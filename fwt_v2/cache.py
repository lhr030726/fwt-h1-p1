"""Identity-addressed, non-pickle residual caches with strict provenance checks.

Every record carries its own sample ID and complete paired arrays. Batch order
and cross-fit fold concatenation order never determine which target is read.
Callers must pass a project workspace root explicitly; outputs outside it are
rejected before any directory creation. Caching does not confer data access or
training authorization.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .artifacts import WORKSPACE


@dataclass(frozen=True)
class CacheMetadata:
    backbone_sha256: str
    schema_version: str
    manifest_sha256: str
    normalizer_sha256: str
    role: str = "residual_train"
    label_definition: str = "next-minus-current-v1"
    augmentation_version: str = "none-v1"
    cache_version: int = 1

    def __post_init__(self):
        for name in ("backbone_sha256", "manifest_sha256", "normalizer_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(name + " must be a lowercase SHA256 digest")
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
               for name in ("schema_version", "role", "label_definition", "augmentation_version")):
            raise ValueError("cache semantic versions and role must be explicit nonempty strings")
        if self.cache_version != 1:
            raise ValueError("unsupported residual cache version")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "bifu" or array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(name + " must be a numeric nonempty CHW array")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(name + " must contain only finite values")
    return array


def _validate_arrays(target, p_det, residual, conditions):
    arrays = {name: _array(value, name)
              for name, value in [("target", target), ("p_det", p_det), ("residual", residual)]}
    shape = arrays["target"].shape
    if shape[0] != 1 or any(value.shape != shape for value in arrays.values()):
        raise ValueError("target, p_det, residual must have identical [1,H,W] shapes")
    if not np.isin(arrays["target"], [0.0, 1.0]).all():
        raise ValueError("target must be binary")
    if ((arrays["p_det"] < 0) | (arrays["p_det"] > 1)).any():
        raise ValueError("p_det must lie in [0,1]")
    if ((arrays["residual"] < -1) | (arrays["residual"] > 1)).any():
        raise ValueError("probability residual must lie in [-1,1]")
    if not np.allclose(arrays["residual"] + arrays["p_det"], arrays["target"], atol=2e-6, rtol=0):
        raise ValueError("residual + p_det must equal target")
    if not isinstance(conditions, Mapping) or "distance" not in conditions:
        raise ValueError("conditions must include distance")
    condition_arrays = {}
    for name, value in conditions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("invalid condition name")
        array = _array(value, "condition " + name)
        if name == "distance" and (array.shape != shape or np.abs(array).max() > 1.000001):
            raise ValueError("distance must match target with values in [-1,1]")
        if array.shape[-2] > shape[-2] or array.shape[-1] > shape[-1]:
            raise ValueError("condition resolution must not exceed target resolution")
        condition_arrays[name] = array
    return arrays, condition_arrays


class ResidualCache:
    def __init__(self, root, metadata: CacheMetadata, *, workspace_root):
        self.root = Path(root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        if self.workspace_root != WORKSPACE and WORKSPACE not in self.workspace_root.parents:
            raise ValueError("workspace_root must remain within the canonical innovation workspace")
        try:
            self.root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("cache output must remain inside explicit workspace_root") from exc
        if not isinstance(metadata, CacheMetadata):
            raise TypeError("metadata must be CacheMetadata")
        self.metadata = asdict(metadata)
        self.metadata_json = _canonical(self.metadata)
        self.metadata_path = self.root / "metadata.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            self._verify_metadata()
        else:
            self._atomic_write(self.metadata_path, lambda handle: handle.write(self.metadata_json.encode("utf-8")))

    def _verify_metadata(self):
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != self.metadata:
            raise ValueError("cache metadata mismatch; use a new cache for different provenance")

    def _atomic_write(self, destination, writer):
        # The temporary file is in the same directory/volume for atomic replace.
        fd, temporary = tempfile.mkstemp(prefix=".partial-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _path(self, sample_id):
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        return self.root / (digest + ".npz")

    def write(self, sample_id, *, target, p_det, residual, conditions):
        destination = self._path(sample_id)
        self._verify_metadata()
        arrays, condition_arrays = _validate_arrays(target, p_det, residual, conditions)
        payload = dict(arrays)
        payload.update({"condition_" + name: value for name, value in condition_arrays.items()})
        record = {
            "sample_id": sample_id, "metadata": self.metadata,
            "shapes": {key: list(value.shape) for key, value in payload.items()},
            "condition_names": sorted(condition_arrays),
        }
        payload["record_json"] = np.asarray(_canonical(record))
        self._atomic_write(destination, lambda handle: np.savez_compressed(handle, **payload))

    def read(self, sample_id):
        path = self._path(sample_id)
        self._verify_metadata()
        with np.load(path, allow_pickle=False) as stored:
            record = json.loads(str(stored["record_json"].item()))
            if record["sample_id"] != sample_id or record["metadata"] != self.metadata:
                raise ValueError("cached sample identity or metadata does not match")
            expected_keys = {"target", "p_det", "residual", "record_json"}
            expected_keys.update("condition_" + name for name in record["condition_names"])
            if set(stored.files) != expected_keys:
                raise ValueError("cached array names do not match record schema")
            arrays = {name: stored[name].copy() for name in expected_keys if name != "record_json"}
            if {name: list(array.shape) for name, array in arrays.items()} != record["shapes"]:
                raise ValueError("cached shapes do not match record schema")
        conditions = {name: arrays["condition_" + name] for name in record["condition_names"]}
        core, conditions = _validate_arrays(arrays["target"], arrays["p_det"], arrays["residual"], conditions)
        return {"sample_id": sample_id, **core, "conditions": conditions}
