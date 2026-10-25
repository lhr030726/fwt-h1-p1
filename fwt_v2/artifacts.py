"""Versioned, atomic artifacts with workspace and model/sampling bindings."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
BINDING_KEYS = ('model_sha256', 'backbone_sha256', 'normalizer_sha256',
                'manifest_sha256', 'family', 'sampling')


def output_path(path):
    path = Path(path).resolve()
    if WORKSPACE not in path.parents:
        raise ValueError('All FWT outputs must stay under the innovation workspace')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_checkpoint(path, bundle):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(handle)
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    # Only tensor/primitive bundles, no arbitrary object deserialization.
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(bundle, dict) or bundle.get('schema') != 'fwt_checkpoint_v2':
        raise ValueError('Expected a versioned FWT-v2 checkpoint')
    return bundle


def verify_binding(actual, expected):
    for key in BINDING_KEYS:
        if key not in actual or actual[key] != expected.get(key):
            raise ValueError('Artifact binding mismatch: ' + key)


class PredictionStore:
    """One NPZ per identity plus a completed index; never a shuffled flat cache."""
    def __init__(self, root, metadata=None, create=False):
        self.root = Path(root).resolve()
        self.index_path = self.root / 'predictions.json'
        if create:
            output_path(self.root / 'predictions.json')
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError('Prediction output must be a new empty directory')
            self.root.mkdir(parents=True, exist_ok=True)
            self.index = dict(schema='fwt_predictions_v2', complete=False,
                              metadata=dict(metadata), samples=[])
            atomic_json(self.index_path, self.index)
        else:
            self.index = read_json(self.index_path)
            if self.index.get('schema') != 'fwt_predictions_v2' or not self.index.get('complete'):
                raise ValueError('Prediction store is incomplete or has the wrong schema')
        self.writable = create
        self.identities = {s['sample_id'] for s in self.index['samples']}

    @property
    def metadata(self):
        return self.index['metadata']

    def write(self, sample_id, event_id, probability, target, valid_mask, domain_mask, calibration_logit=None):
        if not self.writable or sample_id in self.identities:
            raise ValueError('Read-only store or duplicate sample identity')
        p, y = np.asarray(probability), np.asarray(target)
        valid, domain = np.asarray(valid_mask), np.asarray(domain_mask)
        if not (p.shape == y.shape == valid.shape == domain.shape) or not p.size:
            raise ValueError('Prediction, supervision and masks need the same nonempty shape')
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Prediction probabilities must be finite and within [0,1]')
        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Targets must be finite and binary before serialization')
        if not np.isin(valid, [0, 1]).all() or not np.isin(domain, [0, 1]).all():
            raise ValueError('Masks must be binary')
        if np.any(valid.astype(bool) & ~domain.astype(bool)) or np.any((y > 0) & ~valid.astype(bool)):
            raise ValueError('Supervision must lie within the prediction domain')
        if calibration_logit is not None:
            z = np.asarray(calibration_logit)
            if z.shape != p.shape or not np.isfinite(z).all():
                raise ValueError('Calibration logits must be finite and match prediction shape')
        filename = hashlib.sha256(sample_id.encode()).hexdigest() + '.npz'
        path = self.root / filename
        handle, temporary = tempfile.mkstemp(dir=self.root, suffix='.tmp')
        try:
            with os.fdopen(handle, 'wb') as stream:
                extra = {} if calibration_logit is None else dict(calibration_logit=np.asarray(calibration_logit, dtype=np.float32))
                np.savez_compressed(stream, probability=np.asarray(probability, dtype=np.float32),
                                    target=np.asarray(target, dtype=np.uint8),
                                    valid_mask=np.asarray(valid_mask, dtype=bool),
                                    domain_mask=np.asarray(domain_mask, dtype=bool), **extra)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.index['samples'].append(dict(sample_id=sample_id, event_id=event_id,
                                          file=filename, sha256=sha256(path)))
        self.identities.add(sample_id)

    def finish(self):
        if not self.writable or not self.index['samples']:
            raise ValueError('Cannot finish an empty or read-only prediction store')
        self.index['complete'] = True
        atomic_json(self.index_path, self.index)
        self.writable = False

    def records(self):
        for sample in self.index['samples']:
            path = (self.root / sample['file']).resolve()
            if path.parent != self.root or sha256(path) != sample['sha256']:
                raise ValueError('Prediction file path or content mismatch')
            with np.load(path, allow_pickle=False) as content:
                yield dict(sample_id=sample['sample_id'], event_id=sample['event_id'],
                           **{name: content[name] for name in content.files})
