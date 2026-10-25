"""RCDA inputs (legacy 18 or explicit POWER 22) and next-day supervision.

This module neither modifies the official WSTS protocol nor infers a wind
convention. Physics remain signed and separate from training-fitted inputs.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from .artifacts import atomic_json
from .manifest import load_manifest, samples_for_role, validate_manifest
from .geography import geographic_wind
from .hourly import EXTRA_CHANNEL_NAMES, ExternalWindStore, HourlyWind, canonical_digest

INPUT_SCHEMA = "fwt_rcda_inputs_v2"
CIRCULAR_WIND_SCHEMA = "fwt_rcda_inputs_v3_circular_wind"
GEOGRAPHIC_COVARIATES_SCHEMA = "fwt_rcda_inputs_v4_geographic_wind_covariates"
GEOGRAPHIC_COVARIATES_CONVENTION = "geographic_towards_ccw_from_east_covariates"
POWER_WIND_SCHEMA = "fwt_rcda_inputs_v5_power_wind_covariates22"
NORMALIZATION_SCHEMA = "fwt_rcda_normalization_v2"
CHANNEL_NAMES = ("current_burn", "dem_m", "blue", "green", "red", "ndvi", "wind_speed",
                 "raw_wind_angle_rad_unused", "temperature", "precipitation", "humidity", "density",
                 "signed_distance", "wind_x", "wind_y", "slope_rad", "upslope_x", "upslope_y")
CIRCULAR_CHANNEL_NAMES = CHANNEL_NAMES[:13] + ("angle_cos", "angle_sin") + CHANNEL_NAMES[15:]
GEOGRAPHIC_COVARIATES_CHANNEL_NAMES = (CHANNEL_NAMES[:13]
                                      + ("wind_unit_image_x", "wind_unit_image_y")
                                      + CHANNEL_NAMES[15:])
SCHEMA_CHANNEL_NAMES = {INPUT_SCHEMA: CHANNEL_NAMES, CIRCULAR_WIND_SCHEMA: CIRCULAR_CHANNEL_NAMES,
                        GEOGRAPHIC_COVARIATES_SCHEMA: GEOGRAPHIC_COVARIATES_CHANNEL_NAMES,
                        POWER_WIND_SCHEMA: GEOGRAPHIC_COVARIATES_CHANNEL_NAMES + EXTRA_CHANNEL_NAMES}
# Physical sign-preserving channels and the unused original circular angle.
FIXED_CHANNELS = (0, 7, 12, 16, 17)


@dataclass(frozen=True)
class DataConfig:
    wind_convention: str
    wind_reference: str
    wind_metadata_review: str
    pixel_size_m: float = 50.0
    distance_scale_m: float = 3200.0
    remove_explicit_distance: bool = False
    nodata_values: tuple = (-9999.0,)
    input_schema_version: str = INPUT_SCHEMA
    external_wind_manifest: str | None = None
    external_wind_manifest_sha256: str | None = None
    external_wind_variant: str | None = None

    def __post_init__(self):
        supported = ("grid_towards_ccw_from_east", "grid_from_clockwise_from_north",
                     "geographic_towards_ccw_from_east", "unverified_circular_covariates",
                     GEOGRAPHIC_COVARIATES_CONVENTION)
        if self.wind_convention not in supported:
            raise ValueError(f"wind_convention must explicitly be one of {supported}")
        circular = self.wind_convention == "unverified_circular_covariates"
        expected = ("unspecified_angle" if circular else "geographic_east_north"
                    if self.wind_convention.startswith("geographic_") else "projected_grid")
        if self.wind_reference != expected:
            raise ValueError(f"wind_reference must match convention: {expected}")
        if not isinstance(self.wind_metadata_review, str) or not self.wind_metadata_review.strip():
            raise ValueError("wind_metadata_review must record the source/convention audit; do not guess")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be finite and positive")
        if not np.isfinite(self.distance_scale_m) or self.distance_scale_m <= 0:
            raise ValueError("distance_scale_m must be finite and positive")
        expected_schema = (CIRCULAR_WIND_SCHEMA if circular else GEOGRAPHIC_COVARIATES_SCHEMA
                           if self.wind_convention == GEOGRAPHIC_COVARIATES_CONVENTION else INPUT_SCHEMA)
        external = (self.external_wind_manifest, self.external_wind_manifest_sha256, self.external_wind_variant)
        if any(value is not None for value in external):
            if self.wind_convention != GEOGRAPHIC_COVARIATES_CONVENTION:
                raise ValueError("external wind requires geographic learned-only covariates")
            if not isinstance(self.external_wind_manifest, str) or not self.external_wind_manifest.strip():
                raise ValueError("external wind requires a manifest path")
            digest = self.external_wind_manifest_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError("external wind requires a lowercase SHA256 binding")
            if self.external_wind_variant not in ('daily', 'hourly'):
                raise ValueError("external wind variant must be daily or hourly")
            expected_schema = POWER_WIND_SCHEMA
        if self.input_schema_version != expected_schema:
            raise ValueError(f"input_schema_version must match wind convention: {expected_schema}")
        object.__setattr__(self, "nodata_values", tuple(self.nodata_values))

    @property
    def covariate_only_wind(self):
        return self.wind_convention in ("unverified_circular_covariates", GEOGRAPHIC_COVARIATES_CONVENTION)

    @property
    def input_channels(self):
        return len(SCHEMA_CHANNEL_NAMES[self.input_schema_version])


def _valid_input(raw, config):
    valid = np.isfinite(raw).all(axis=0)
    for value in config.nodata_values:
        valid &= (raw != value).all(axis=0)
    valid &= (raw[0] >= 0) & (raw[0] <= 1) & (raw[6] >= 0)
    return valid


def _nearest_fill(array, valid):
    if valid.all():
        return array
    if not valid.any():
        return np.zeros_like(array)
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return array[tuple(indices)]


def derive_inputs(raw, config: DataConfig, georeference=None, hourly_wind=None):
    """Return (versioned CHW, signed physics dict, HW input-valid mask), without labels.

    Signed distance is negative within current burn, positive outside; empty/full
    burns saturate to +1/-1. Image x increases right and y increases down. A
    nearest-valid DEM fill is used only for finite local derivatives; invalid
    pixels themselves always stay outside the prediction domain.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 12 or min(raw.shape[1:]) < 2:
        raise ValueError(f"raw input must be [12,H,W] with H,W>=2; received {raw.shape}")
    valid = _valid_input(raw, config)
    clean = np.where(valid[None], raw, 0).astype(np.float32)
    fire = (clean[0] > 0.5) & valid
    if not fire.any():
        distance = np.ones(fire.shape, np.float32)
    elif fire.all():
        distance = -np.ones(fire.shape, np.float32)
    else:
        distance = (distance_transform_edt(~fire, sampling=config.pixel_size_m)
                    - distance_transform_edt(fire, sampling=config.pixel_size_m)) / config.distance_scale_m
        distance = np.clip(distance, -1, 1).astype(np.float32)
    if config.remove_explicit_distance:
        distance = np.zeros_like(distance)
    speed, angle = clean[6], clean[7]
    circular = config.wind_convention == "unverified_circular_covariates"
    geographic_covariates = config.wind_convention == GEOGRAPHIC_COVARIATES_CONVENTION
    if config.covariate_only_wind:
        # These are dimensionless periodic covariates, NOT a physical vector.
        wind_x, wind_y = np.zeros_like(speed), np.zeros_like(speed)
    elif config.wind_convention == "geographic_towards_ccw_from_east":
        wind_x, wind_y = geographic_wind(speed, angle, georeference, config.pixel_size_m)
    elif config.wind_convention == "grid_towards_ccw_from_east":
        wind_x, wind_y = speed * np.cos(angle), -speed * np.sin(angle)
    else:  # clockwise from grid north, FROM direction; convert to TOWARDS.
        wind_x, wind_y = -speed * np.sin(angle), speed * np.cos(angle)
    dem = _nearest_fill(clean[1], valid)
    grad_y, grad_x = np.gradient(dem, config.pixel_size_m, config.pixel_size_m)
    magnitude = np.hypot(grad_x, grad_y)
    slope = np.arctan(magnitude)
    up_x = np.divide(grad_x, magnitude, out=np.zeros_like(grad_x), where=magnitude > 1e-12)
    up_y = np.divide(grad_y, magnitude, out=np.zeros_like(grad_y), where=magnitude > 1e-12)
    priors = np.stack((distance, wind_x, wind_y, slope, up_x, up_y)).astype(np.float32)
    priors[:, ~valid] = 0
    if not np.isfinite(priors).all():
        raise ValueError("nonfinite derived physics; check physical units and input range")
    derived = np.concatenate((clean, priors), axis=0)
    if circular:
        derived[13] = np.where(valid, np.cos(angle), 0)
        derived[14] = np.where(valid, np.sin(angle), 0)
    elif geographic_covariates:
        # Empirical geographic angle convention, learned covariates only. Speed
        # remains in slot 6; exact calm has no direction. No positive-speed gate.
        if config.external_wind_variant is not None:
            if (not isinstance(hourly_wind, HourlyWind)
                    or hourly_wind.manifest_sha256 != config.external_wind_manifest_sha256
                    or hourly_wind.georeference_sha256 != canonical_digest(georeference)):
                raise ValueError("external wind requires a source-bound sample with matching georeference")
            derived[6] = np.where(valid, hourly_wind.mean_speed, 0)
            derived[7] = 0  # The original source angle is not part of this input schema.
            angle = np.full_like(speed, hourly_wind.mean_angle)
            unit_speed = (valid & hourly_wind.direction_present).astype(np.float32)
            extras = np.asarray(hourly_wind.extras if config.external_wind_variant == 'hourly'
                                else (0., 0., 0., 0.), dtype=np.float32)
            derived = np.concatenate((derived, extras[:, None, None] * valid[None]), axis=0)
        else:
            if hourly_wind is not None:
                raise ValueError("hourly wind supplied to a legacy daily input schema")
            unit_speed = (valid & (speed > 0)).astype(np.float32)
        derived[13:15] = np.stack(geographic_wind(unit_speed, angle, georeference, config.pixel_size_m))
    domain = valid & ~fire
    physics = dict(distance=priors[0:1], wind_xy=priors[1:3], slope=priors[3:4],
                   upslope_xy=priors[4:6], domain_mask=domain[None])
    return derived, physics, valid


@dataclass
class NormalizationStats:
    mean: list
    std: list
    fit_sample_ids: list
    data_config: dict
    role: str = "backbone_train"
    schema_version: str = NORMALIZATION_SCHEMA
    input_schema_version: str = INPUT_SCHEMA
    channel_names: tuple = CHANNEL_NAMES
    georeference_sha256: str | None = None

    def __post_init__(self):
        if self.schema_version != NORMALIZATION_SCHEMA:
            raise ValueError("unsupported normalization/input schema")
        config = DataConfig(**self.data_config)
        if self.input_schema_version != config.input_schema_version:
            raise ValueError("normalization input schema differs from data_config")
        if self.role != "backbone_train":
            raise ValueError("normalization must be fitted only on backbone_train")
        means, stds = np.asarray(self.mean), np.asarray(self.std)
        channels = config.input_channels
        if means.shape != (channels,) or stds.shape != (channels,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError(f"normalization requires {channels} finite means and positive standard deviations")
        expected_names = SCHEMA_CHANNEL_NAMES[self.input_schema_version]
        if tuple(self.channel_names) != expected_names:
            raise ValueError("normalization channel names/order disagree with input schema")
        if not self.fit_sample_ids or len(set(self.fit_sample_ids)) != len(self.fit_sample_ids):
            raise ValueError("normalization fit_sample_ids must be nonempty and unique")
        self.data_config = asdict(config)
        if self.data_config['wind_reference'] == 'geographic_east_north':
            if not isinstance(self.georeference_sha256, str) or len(self.georeference_sha256) != 64:
                raise ValueError('Geographic normalization must bind the fitted georeferences')
        self.channel_names = tuple(self.channel_names)

    @classmethod
    def fit(cls, raw_samples: Iterable, config: DataConfig, *, role: str, sample_ids: list,
            sample_georeferences=None, sample_rows=None, external_wind_store=None):
        if role != "backbone_train":
            raise ValueError("normalization may only fit backbone_train inputs")
        if sample_georeferences is not None and len(sample_georeferences) != len(sample_ids):
            raise ValueError("sample_georeferences must match sample_ids exactly")
        if config.external_wind_variant is not None:
            if (external_wind_store is None or sample_rows is None
                    or [row['sample_id'] for row in sample_rows] != list(sample_ids)
                    or any(row['role'] != 'backbone_train' for row in sample_rows)):
                raise ValueError("external wind normalization requires exact backbone_train sample rows and audited store")
        count, mean, m2, number = 0, np.zeros(config.input_channels, np.float64), np.zeros(config.input_channels, np.float64), 0
        for raw in raw_samples:
            if number >= len(sample_ids):
                raise ValueError("More raw inputs than declared sample_ids")
            georef = None if sample_georeferences is None else sample_georeferences[number]
            hourly = None if external_wind_store is None else external_wind_store.for_sample(sample_rows[number])
            derived, _, valid = derive_inputs(raw, config, georef, hourly)
            values = derived[:, valid].astype(np.float64)
            number += 1
            if values.shape[1] == 0:
                continue
            n = values.shape[1]
            batch_mean = values.mean(axis=1)
            batch_m2 = ((values - batch_mean[:, None]) ** 2).sum(axis=1)
            delta = batch_mean - mean
            total = count + n
            m2 += batch_m2 + delta ** 2 * count * n / total
            mean += delta * n / total
            count = total
        if number != len(sample_ids):
            raise ValueError("fit_sample_ids must match the exact number of fitted samples")
        if count == 0:
            raise ValueError("backbone_train has no valid input pixels for normalization")
        std = np.sqrt(np.maximum(m2 / count, 0))
        std[std < 1e-6] = 1
        mean[list(FIXED_CHANNELS)] = 0
        std[list(FIXED_CHANNELS)] = 1
        digest = georeference_digest(sample_ids, sample_georeferences) if config.wind_reference == 'geographic_east_north' else None
        names = SCHEMA_CHANNEL_NAMES[config.input_schema_version]
        return cls(mean.tolist(), std.tolist(), list(sample_ids), asdict(config), role=role,
                   input_schema_version=config.input_schema_version, channel_names=names,
                   georeference_sha256=digest)

    def transform(self, derived, valid):
        derived = np.asarray(derived, dtype=np.float32)
        if derived.ndim != 3 or derived.shape[0] != len(self.mean) or valid.shape != derived.shape[1:]:
            raise ValueError(f"normalization input must be CHW{len(self.mean)} with matching HW validity")
        result = (derived - np.asarray(self.mean, np.float32)[:, None, None]) / np.asarray(self.std, np.float32)[:, None, None]
        result[:, ~valid] = 0
        if not np.isfinite(result).all():
            raise ValueError("nonfinite normalized input")
        return result.astype(np.float32)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def save(self, path):
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def georeference_digest(sample_ids, georeferences):
    if georeferences is None or len(sample_ids) != len(georeferences):
        raise ValueError('Missing exact normalization georeferences')
    entries = sorted(zip(sample_ids, georeferences), key=lambda item: item[0])
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def fit_normalization(manifest, config: DataConfig) -> NormalizationStats:
    """Explicit read of backbone_train input arrays only; never labels or other roles."""
    document = load_manifest(manifest) if isinstance(manifest, (str, Path)) else validate_manifest(manifest)
    rows = samples_for_role(document, "backbone_train")
    external = (ExternalWindStore(config.external_wind_manifest, config.external_wind_manifest_sha256, document)
                if config.external_wind_variant is not None else None)
    return NormalizationStats.fit((np.load(row["input_path"], allow_pickle=False) for row in rows), config,
                                  role="backbone_train", sample_ids=[row["sample_id"] for row in rows],
                                  sample_georeferences=[row.get("georeference") for row in rows],
                                  sample_rows=rows, external_wind_store=external)


class RCDADataset(Dataset):
    """Read-only, single-role dataset; shuffle always moves identity and tensors together."""

    def __init__(self, manifest, role: str, normalization: NormalizationStats, config: DataConfig):
        document = load_manifest(manifest) if isinstance(manifest, (str, Path)) else validate_manifest(manifest)
        if 'sanity_subset' in document and role not in document['sanity_subset'].get('permitted_array_roles', []):
            raise PermissionError('Sanity manifest retains this role as metadata only; array access is forbidden')
        self.samples = samples_for_role(document, role)
        self.role, self.normalization, self.config = role, normalization, config
        if config.external_wind_variant is not None and role not in ('backbone_train', 'residual_train', 'validation'):
            raise PermissionError('External hourly wind is restricted to development roles')
        if asdict(DataConfig(**normalization.data_config)) != asdict(config):
            raise ValueError("normalization physics/configuration differs from dataset configuration")
        tb_ids = {row["sample_id"] for row in document["samples"] if row["role"] == "backbone_train"}
        if set(normalization.fit_sample_ids) != tb_ids:
            raise ValueError("normalization fit_sample_ids must exactly match manifest backbone_train")
        if config.wind_reference == 'geographic_east_north':
            tb = samples_for_role(document, 'backbone_train')
            digest = georeference_digest([s['sample_id'] for s in tb], [s.get('georeference') for s in tb])
            if digest != normalization.georeference_sha256:
                raise ValueError('normalization georeference binding differs from the manifest')
        self.external_wind = (ExternalWindStore(config.external_wind_manifest, config.external_wind_manifest_sha256, document)
                              if config.external_wind_variant is not None else None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        hourly = None if self.external_wind is None else self.external_wind.for_sample(row)
        raw = np.load(row["input_path"], allow_pickle=False)
        label = np.asarray(np.load(row["label_path"], allow_pickle=False), dtype=np.float32)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        derived, physics, input_valid = derive_inputs(raw, self.config, row.get("georeference"), hourly)
        if label.shape != derived.shape[1:]:
            raise ValueError(f"label shape does not match input for {row['sample_id']}")
        label_valid = np.isfinite(label) & (label >= 0) & (label <= 1)
        domain = physics["domain_mask"][0]
        supervised = domain & label_valid
        target = (label > 0.5) & supervised
        return dict(x_norm=torch.from_numpy(self.normalization.transform(derived, input_valid)),
                    physics={key: torch.from_numpy(np.array(value, copy=True)) for key, value in physics.items()},
                    target=torch.from_numpy(target[None].astype(np.float32)),
                    valid_mask=torch.from_numpy(supervised[None]),
                    domain_mask=torch.from_numpy(domain[None].copy()),
                    label_valid_mask=torch.from_numpy(label_valid[None]),
                    sample_id=row["sample_id"], event_id=row["event_id"], date=row["date"],
                    target_date=row["target_date"])
