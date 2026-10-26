"""Strictly previous-input burn changes and a frozen-parent residual branch.

History validity describes common *input* validity, not the unburned supervision
domain: newly burned pixels must remain available as context. This is a
retrospective daily-input experiment, not a certification of acquisition latency.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import _valid_input


HISTORY_SCHEMA = "fwt_rcda_exact_previous_input_burn_change_v1"


def _grid_key(row):
    georef = row.get("georeference")
    required = ("crs", "affine_transform", "shape")
    if not isinstance(georef, Mapping) or any(k not in georef for k in required):
        return None
    shape, affine = georef["shape"], georef["affine_transform"]
    if (not isinstance(georef["crs"], str) or not georef["crs"].strip()
            or not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(v) is not int or v < 2 for v in shape)
            or not isinstance(affine, (tuple, list)) or len(affine) != 6
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) for v in affine)):
        return None
    # Do not infer reprojection, tolerances, neighboring dates, or sensor grids.
    return json.dumps({k: georef[k] for k in required}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def build_history_pairs(samples):
    """Return previous-row indices and audit without accessing any array.

    Only exact t-1, same event/role/spatial group/CRS/affine/shape qualifies.
    Ineligible rows retain None; duplicate identities and malformed dates fail
    closed. Indices always refer to the supplied order, independent of sorting.
    """
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a nonempty ordered sequence")
    lookup, sample_ids, days = {}, set(), []
    for i, row in enumerate(samples):
        if not isinstance(row, Mapping) or any(
                not isinstance(row.get(k), str) or not row[k]
                for k in ("sample_id", "event_id", "role", "date")):
            raise ValueError("history rows require sample_id/event_id/role/date strings")
        day = date.fromisoformat(row["date"])
        if day.isoformat() != row["date"]:
            raise ValueError("history date must be canonical YYYY-MM-DD")
        identity = (row["event_id"], day)
        if row["sample_id"] in sample_ids or identity in lookup:
            raise ValueError("ambiguous duplicate history sample_id or event/date")
        sample_ids.add(row["sample_id"])
        lookup[identity] = i
        days.append(day)

    previous_indices, counts, records = [], Counter(), []
    for i, row in enumerate(samples):
        previous = lookup.get((row["event_id"], days[i] - timedelta(days=1)))
        if previous is None:
            reason = "no_exact_previous_date_in_manifest"
        else:
            prior = samples[previous]
            if prior["role"] != row["role"]:
                reason = "role_mismatch"
            elif (not isinstance(row.get("group_id"), str) or not row["group_id"]
                  or not isinstance(prior.get("group_id"), str) or not prior["group_id"]):
                reason = "missing_spatial_group"
            elif prior["group_id"] != row["group_id"]:
                reason = "spatial_group_mismatch"
            elif _grid_key(row) is None or _grid_key(prior) is None:
                reason = "missing_or_invalid_grid_metadata"
            elif _grid_key(row) != _grid_key(prior):
                reason = "grid_mismatch"
            else:
                reason = "eligible_metadata_pair"
        eligible = reason == "eligible_metadata_pair"
        previous_indices.append(previous if eligible else None)
        counts[reason] += 1
        records.append(dict(sample_id=row["sample_id"], event_id=row["event_id"],
            date=row["date"], role=row["role"], reason=reason,
            previous_sample_id=samples[previous]["sample_id"] if eligible else None))
    audit = dict(schema=HISTORY_SCHEMA, samples=len(samples), counts=dict(counts),
        paired_samples=counts["eligible_metadata_pair"],
        paired_fraction=counts["eligible_metadata_pair"] / len(samples),
        arrays_read=False, records=records,
        pairing="exact t-1; same event, role, explicit spatial group, CRS, affine and shape")
    return previous_indices, audit


def load_history_arrays(dataset):
    """Read raw inputs only and return NumPy caches in dataset.samples order.

    Requires a resolved, validated single-development-role RCDADataset (or an
    equivalent object exposing samples/config/role). Never calls __getitem__ or
    reads labels. has_history denotes an eligible metadata pair even when its
    common valid pixel set is empty; history_valid gives actual pixel validity.
    """
    samples, config, role = dataset.samples, dataset.config, dataset.role
    if role not in ("backbone_train", "residual_train", "validation"):
        raise PermissionError("history loading is restricted to development roles")
    if any(row.get("role") != role for row in samples):
        raise ValueError("history dataset must contain only its declared role")
    previous_indices, pairing = build_history_pairs(samples)
    fires, input_valid, input_paths, expected_shape = [], [], [], None
    for row in samples:
        path = Path(row["input_path"])
        if not path.is_absolute():
            raise ValueError("history input paths must already be manifest-resolved absolute paths")
        raw = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if raw.ndim != 3 or raw.shape[0] != 12 or min(raw.shape[1:]) < 2:
            raise ValueError(f"history raw input must be [12,H,W]: {row['sample_id']}")
        shape = tuple(raw.shape[-2:])
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            raise ValueError("history cache requires one shared actual H,W")
        georef = row.get("georeference", {})
        if "shape" in georef and tuple(georef["shape"]) != shape:
            raise ValueError(f"history input shape disagrees with grid metadata: {row['sample_id']}")
        valid = _valid_input(raw, config)
        fires.append(((raw[0] > .5) & valid).copy())
        input_valid.append(valid.copy())
        input_paths.append(str(path))
    fires, input_valid = np.stack(fires), np.stack(input_valid)
    delta = np.zeros((len(samples), 1, *expected_shape), dtype=np.float32)
    common = np.zeros_like(delta, dtype=np.bool_)
    records = []
    for i, previous in enumerate(previous_indices):
        if previous is not None:
            common[i, 0] = input_valid[i] & input_valid[previous]
            difference = fires[i].astype(np.float32) - fires[previous].astype(np.float32)
            delta[i, 0] = np.where(common[i, 0], difference, 0.)
        positive = int(np.count_nonzero(delta[i] > 0))
        negative = int(np.count_nonzero(delta[i] < 0))
        records.append(dict(sample_id=samples[i]["sample_id"],
            previous_sample_id=samples[previous]["sample_id"] if previous is not None else None,
            paired=previous is not None, current_input_valid_pixels=int(input_valid[i].sum()),
            common_input_valid_pixels=int(common[i].sum()), positive_change_pixels=positive,
            negative_change_pixels=negative,
            unchanged_burned_pixels=int((common[i, 0] & fires[i] & (delta[i, 0] == 0)).sum())))
    totals = {key: sum(row[key] for row in records) for key in (
        "current_input_valid_pixels", "common_input_valid_pixels", "positive_change_pixels",
        "negative_change_pixels", "unchanged_burned_pixels")}
    audit = dict(schema=HISTORY_SCHEMA, role=role, pairing=pairing, shape=list(expected_shape),
        input_arrays_read=len(samples), label_arrays_read=0, input_paths=input_paths,
        total_sample_pixels=int(len(samples) * np.prod(expected_shape)), totals=totals,
        paired_samples_without_valid_pixels=sum(r["paired"] and r["common_input_valid_pixels"] == 0
                                                for r in records),
        sample_records=records,
        burn_definition="raw[0] > 0.5; same validity as data._valid_input",
        delta_definition="current input burn minus exact previous-day input burn; signed {-1,0,1}",
        validity_definition="pair exists AND current input valid AND previous input valid; no label/domain mask",
        availability_limit="Declared past dates only; observational delivery latency is not established")
    return dict(history_delta=delta, history_valid=common,
        has_history=np.asarray([p is not None for p in previous_indices], dtype=np.bool_),
        previous_sample_ids=[samples[p]["sample_id"] if p is not None else None for p in previous_indices],
        audit=audit)


class HistoryRefiner(nn.Module):
    """Small 1/4-grid additive correction; unavailable pixels exactly use parent.

    Two 3x3 convolutions mix a 5x5 low-resolution neighborhood (plus pooling and
    interpolation support). This is not the full receptive field: decoder
    features and GroupNorm already carry broader spatial information.

    Decoder and change values outside common input validity are zero-filled
    before area averaging. The separate averaged validity channel gives the
    valid fraction; values are intentionally not divided by that fraction.
    """

    schema = "fwt_frozen_parent_history_refiner_v1"

    def __init__(self, decoder_channels=64, hidden_channels=16, downsample_factor=4):
        super().__init__()
        if any(type(v) is not int or v < 1
               for v in (decoder_channels, hidden_channels, downsample_factor)):
            raise ValueError("channels and downsample_factor must be positive integers")
        self.decoder_channels = decoder_channels
        self.hidden_channels = hidden_channels
        self.downsample_factor = downsample_factor
        self.features = nn.Sequential(
            nn.Conv2d(decoder_channels + 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels), nn.SiLU())
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def configuration(self):
        return dict(schema=self.schema, history_schema=HISTORY_SCHEMA,
            decoder_channels=self.decoder_channels, hidden_channels=self.hidden_channels,
            downsample_factor=self.downsample_factor, downsample="area",
            invalid_condition_values="zero_filled_before_area_average",
            reduced_validity_channel="fraction_of_common_valid_pixels",
            area_average_denominator="all_pixels_in_area_not_valid_pixels_only",
            convolution_kernels=[3, 3, 1], normalization="group", normalization_groups=1,
            activation="silu", upsample="bilinear_align_corners_false",
            condition_order=["decoder_feature", "signed_input_burn_change", "common_input_valid"],
            conditions_detached=True, output_initialization="zero_final_layer_only",
            output="detached_parent_logit_plus_correction_where_common_input_valid",
            unavailable_pixel_behavior="exact_parent_logit")

    def forward(self, outputs, parent_logits, history_delta, history_valid):
        if not isinstance(outputs, Mapping):
            raise ValueError("outputs must be a mapping")
        logits, feature = parent_logits, outputs.get("decoder_feature")
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or logits.shape[0] < 1 or logits.shape[1] != 1
                or min(logits.shape[-2:]) < self.downsample_factor
                or not logits.is_floating_point()):
            raise ValueError("parent_logits must be floating [B,1,H,W] with H,W >= downsample_factor")
        expected = (logits.shape[0], self.decoder_channels, *logits.shape[-2:])
        if (not isinstance(feature, torch.Tensor) or tuple(feature.shape) != expected
                or feature.device != logits.device or feature.dtype != logits.dtype):
            raise ValueError("decoder_feature must match parent shape/device/dtype")
        if (not isinstance(history_delta, torch.Tensor) or history_delta.shape != logits.shape
                or history_delta.device != logits.device or history_delta.dtype != logits.dtype):
            raise ValueError("history_delta must match parent shape/device/dtype")
        if (not isinstance(history_valid, torch.Tensor) or history_valid.shape != logits.shape
                or history_valid.device != logits.device or history_valid.dtype != torch.bool):
            raise ValueError("history_valid must be bool and match parent shape/device")
        # Shape/type checks avoid repeated GPU value reductions and synchronization.
        # The input-only loader guarantees finite signed binary changes; torch.where
        # also ensures unknown values outside common validity never reach the branch.
        valid = history_valid.detach()
        change = torch.where(valid, history_delta.detach(), torch.zeros_like(history_delta))
        feature = torch.where(valid, feature.detach(), torch.zeros_like(feature))
        conditions = torch.cat((feature, change, valid.to(logits.dtype)), dim=1)
        reduced_size = tuple(v // self.downsample_factor for v in logits.shape[-2:])
        conditions = F.interpolate(conditions, size=reduced_size, mode="area")
        correction = self.output(self.features(conditions))
        correction = F.interpolate(correction, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        parent = logits.detach()
        return torch.where(valid, parent + correction, parent)
