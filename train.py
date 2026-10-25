"""Portable H1/P1 training. Default is a read-only metadata preflight.

Run --phase det --execute, then --phase diff --execute explicitly. There is
no held-out evaluation command, automatic second stage, or resume shortcut.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from fwt_v2.portable import (PortableSources, bound_file, contained_path,
                             file_sha256, read_json)

REPOSITORY = Path(__file__).resolve().parent
DEVELOPMENT_ROLES = ('backbone_train', 'residual_train', 'validation')
FULL_COUNTS = dict(backbone_train=3197, residual_train=811, validation=568)
MODEL = dict(widths=[16, 32, 64, 128], decoder_widths=[64, 128, 256, 512],
             prop_channels=64, use_gate=True, use_distance=True, use_propagation=True,
             use_physics_prior=True, drop_modal=.15, propagation_steps=3,
             propagation_radius=3, normalization='group', norm_max_groups=8,
             use_wind_prior=False, schema_version='rcda22-semantic21-v3',
             extra_weather_channels=4)
DIFFUSION = dict(timesteps=200, hidden_channels=[32, 64, 96, 128],
                 condition_channels=16, condition_mode='multiscale',
                 prediction_type='clean_probability', training_timestep_sampling='antithetic')
RECIPE = dict(seed=0, batch_size=1, learning_rate=.001, optimizer='AdamW',
              weight_decay=.01, det_loss='focal_alpha_positive_0.75_gamma_2',
              diff_loss='clean_probability_bce', K=8, steps=4, sampler='ddim',
              checkpoint='maximum_full_validation_pooled_AP_earliest_tie')


def source_digest():
    return hashlib.sha256(''.join(p.name + file_sha256(p) for p in
        sorted((REPOSITORY / 'fwt_v2').glob('*.py'))).encode()).hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--phase', choices=('det', 'diff'), required=True)
    result.add_argument('--data-root', type=Path, default=REPOSITORY / 'data')
    result.add_argument('--run-dir', type=Path, default=REPOSITORY / 'runs' / 'h1p1')
    result.add_argument('--config', type=Path, help='Optional exact copy of bundle config.h1.json')
    result.add_argument('--device', default='cpu', help='cpu, cuda or cuda:N; selected by the user')
    result.add_argument('--cpu-threads', type=int, default=2)
    result.add_argument('--batch-size', type=int, default=1,
                        help='Explicit training recipe variant; use the same value for H1 and P1')
    result.add_argument('--p1-zero-init', action='store_true',
                        help='Zero P1 final correction layer after normal construction; set for BOTH H1 and P1')
    result.add_argument('--validation-batch-size', type=int, default=None,
                        help='Validation batching; defaults to training batch size')
    result.add_argument('--sample-cache-gib', type=float, default=0.,
                        help='Shared CPU tensor-cache budget for training and validation, 0..64 GiB')
    result.add_argument('--epochs', type=int, choices=(2, 30), default=30)
    result.add_argument('--execute', action='store_true', help='Explicitly start this one training stage')
    return result


def _expanded_data(data):
    result = dict(data)
    result.setdefault('remove_explicit_distance', False)
    result.setdefault('nodata_values', [-9999.0])
    return result


def _check_manifest(document, original, data_root):
    """Metadata-only: original identities, role separation, next-day and stat."""
    from datetime import date, timedelta
    if document.get('schema_version') != 'fwt_rcda_manifest_v2' or document.get('data_root') != '.':
        raise ValueError('Expected a bundle-relative development manifest')
    audit = document.get('audit', {})
    if (audit.get('label_semantics') != 'next_day_cumulative_burn'
            or audit.get('dates_verified') is not True or audit.get('roles_frozen') is not True
            or not audit.get('exposure_review')):
        raise ValueError('Missing original label/date/role audit')
    original_rows = {row['sample_id']: row for row in original['samples']}
    ids, paths, inodes, event_roles, group_roles = set(), set(), set(), {}, {}
    counts = Counter()
    for row in document['samples']:
        sid, role = row['sample_id'], row['role']
        if sid in ids or role not in DEVELOPMENT_ROLES:
            raise ValueError('Duplicate identity or held-out role in runtime data')
        ids.add(sid)
        counts[role] += 1
        fields = lambda value: {k: v for k, v in value.items() if k not in ('input_path', 'label_path')}
        if sid not in original_rows or fields(row) != fields(original_rows[sid]):
            raise ValueError('Runtime sample identity/role differs from original full manifest')
        if date.fromisoformat(row['target_date']) - date.fromisoformat(row['date']) != timedelta(days=1):
            raise ValueError('Target must be the exact next day')
        for key, mapping in ((row['event_id'], event_roles),
                             (row.get('group_id', row['event_id']), group_roles)):
            if key in mapping and mapping[key] != role:
                raise ValueError('Event/spatial group crosses development roles')
            mapping[key] = role
        for field in ('input_path', 'label_path'):
            path = contained_path(data_root, row[field])
            if path.suffix != '.npy' or not path.is_file() or path in paths:
                raise ValueError('Missing/aliased development array path')
            paths.add(path)
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if stat.st_ino and identity in inodes:
                raise ValueError('Hard-linked array aliases are forbidden')
            inodes.add(identity)
    if set(counts) != set(DEVELOPMENT_ROLES):
        raise ValueError('All three development roles are required')
    return dict(counts)


def _check_normalizer(normalizer, original, config, manifest, weather_hash):
    migrated = json.loads(json.dumps(original))
    migrated['data_config']['external_wind_manifest'] = 'weather_manifest.json'
    migrated['data_config']['external_wind_manifest_sha256'] = weather_hash
    if normalizer != migrated:
        raise ValueError('Normalization statistics changed beyond explicit weather relocation')
    data = _expanded_data(config['data'])
    if data != _expanded_data(normalizer['data_config']):
        raise ValueError('Config differs from frozen normalizer data configuration')
    if (data.get('input_schema_version') != 'fwt_rcda_inputs_v5_power_wind_covariates22'
            or data.get('external_wind_variant') != 'hourly'
            or data.get('external_wind_manifest') != 'weather_manifest.json'
            or data.get('external_wind_manifest_sha256') != weather_hash
            or data.get('wind_reference') != 'geographic_east_north'
            or data.get('wind_convention') != 'geographic_towards_ccw_from_east_covariates'
            or data.get('pixel_size_m') != 50 or data.get('distance_scale_m') != 1600
            or data.get('remove_explicit_distance') is not False):
        raise ValueError('Only the frozen H1 learned hourly-covariate configuration is supported')
    tb = [row for row in manifest['samples'] if row['role'] == 'backbone_train']
    ids = [row['sample_id'] for row in tb]
    if (normalizer.get('role') != 'backbone_train'
            or normalizer.get('fit_sample_ids') != ids):
        raise ValueError('Normalizer must preserve the exact ordered Tb fitting identities')
    geo = sorted(zip(ids, [row['georeference'] for row in tb]), key=lambda item: item[0])
    digest = hashlib.sha256(json.dumps(geo, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if normalizer.get('georeference_sha256') != digest:
        raise ValueError('Normalizer georeferences differ from Tb')
    for name in ('mean', 'std'):
        values = normalizer.get(name, [])
        if len(values) != 22 or any(not math.isfinite(v) or (name == 'std' and v <= 0) for v in values):
            raise ValueError('Expected 22 finite frozen normalization statistics')


def _check_array_index(index, manifest, data_root):
    if index.get('schema_version') != 'fwt_portable_array_index_v1':
        raise ValueError('Expected the published development array byte index')
    expected = {(row['sample_id'], field): (row['role'], 'data/' + row[field])
                for row in manifest['samples'] for field in ('input_path', 'label_path')}
    found = set()
    for item in index.get('files', []):
        key = (item['sample_id'], item['field'])
        if (key in found or key not in expected
                or (item['role'], item['path']) != expected[key]
                or type(item.get('size')) is not int or item['size'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))):
            raise ValueError('Array index identity/path differs from runtime sample pairing')
        found.add(key)
        # The archive root is named data; data_root itself may be relocated/renamed.
        path = contained_path(data_root, item['path'][len('data/'):])
        if path.stat().st_size != item['size']:
            raise ValueError('Array size differs from published byte index')
    if found != set(expected):
        raise ValueError('Array index must cover every input/label exactly once')


def _check_own_det(directory, binding):
    if not (directory / 'portable_completed.json').is_file():
        raise ValueError('Diff preflight waits for the completed Det in the SAME run-dir/profile')
    receipt = read_json(directory / 'portable_completed.json')
    if (receipt.get('schema_version') != 'fwt_portable_training_completed_v1'
            or receipt.get('status') != 'completed' or receipt.get('phase') != 'det'
            or receipt.get('binding') != binding):
        raise ValueError('Diff requires its own completed Det with identical data/source/budget')
    files = receipt.get('files', {})
    for name in ('best.pth', 'selection.json', 'diagnostics.json', 'history.json', 'configuration.json'):
        if files.get(name) != file_sha256(directory / name):
            raise ValueError('Own Det completion artifact changed: ' + name)
    diagnostic = read_json(directory / 'diagnostics.json')
    selection = read_json(directory / 'selection.json')
    if (diagnostic.get('status') != 'completed'
            or diagnostic.get('completed_epochs') != binding['epochs']
            or selection.get('selection_frozen') is not True):
        raise ValueError('Own Det selection is not fully completed and frozen')
    return directory / 'best.pth'


def preflight(args):
    if args.cpu_threads < 1 or not re.fullmatch(r'cpu|cuda(?::\d+)?', args.device):
        raise ValueError('Use positive CPU threads and cpu/cuda/cuda:N')
    validation_batch_size = args.batch_size if args.validation_batch_size is None else args.validation_batch_size
    if (not 1 <= args.batch_size <= 128 or
            (args.validation_batch_size is not None and not 1 <= args.validation_batch_size <= 128)):
        raise ValueError('Training and validation batch sizes must be between 1 and 128')
    if not math.isfinite(args.sample_cache_gib) or not 0 <= args.sample_cache_gib <= 64:
        raise ValueError('Sample cache must be a finite budget from 0 to 64 GiB')
    if args.epochs not in (2, 30) or args.phase not in ('det', 'diff'):
        raise ValueError('Only the fixed 30-epoch recipe or isolated 2-epoch smoke is supported')
    root, run_root = args.data_root.resolve(), args.run_dir.resolve()
    if root == run_root or root in run_root.parents or run_root in root.parents:
        raise ValueError('Data and output directories must be disjoint')
    bundle_path = root / 'bundle.json'
    bundle = read_json(bundle_path)
    if (bundle.get('schema_version') != 'fwt_portable_data_bundle_v1'
            or bundle.get('data_scope') != 'development_only'
            or bundle.get('heldout_arrays_included') is not False):
        raise ValueError('A development-only portable bundle is required')
    files = {name: bound_file(root, bundle['files'][name])
             for name in ('manifest', 'normalizer', 'weather', 'config', 'arrays_index')}
    if files['weather'] != root / 'weather_manifest.json':
        raise ValueError('Bundle weather path must match the portable configuration')
    config_path = args.config.resolve() if args.config else files['config']
    if file_sha256(config_path) != file_sha256(files['config']):
        raise ValueError('Config must be an exact copy of the bound bundle config')
    config = read_json(config_path)
    if config.get('model') != MODEL or config.get('diffusion') != DIFFUSION:
        raise ValueError('Only the frozen H1/P1 architecture and diffusion recipe are supported')
    manifest = read_json(files['manifest'])
    original = read_json(bound_file(root, bundle['original_full_manifest']))
    counts = _check_manifest(manifest, original, root)
    if counts != bundle['role_counts']:
        raise ValueError('Bundle role counts differ from actual metadata')
    _check_array_index(read_json(files['arrays_index']), manifest, root)
    if args.epochs == 30:
        if counts != FULL_COUNTS or bundle.get('purpose') == 'smoke':
            raise ValueError('Formal training requires the complete 3197/811/568 development bundle')
        expected_ids = {r['sample_id'] for r in original['samples'] if r['role'] in DEVELOPMENT_ROLES}
        if expected_ids != {r['sample_id'] for r in manifest['samples']}:
            raise ValueError('Formal training must preserve all original development samples')
    _check_normalizer(read_json(files['normalizer']),
        read_json(bound_file(root, bundle['original_normalizer'])), config, manifest,
        bundle['files']['weather']['sha256'])
    sources = PortableSources(files['weather'])
    profile = 'formal' if args.epochs == 30 else 'smoke-2ep'
    output = run_root / profile / args.phase
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh phase output is required; choose a new --run-dir')
    binding = dict(source_sha256=source_digest(), entry_sha256=file_sha256(__file__),
        bundle_sha256=file_sha256(bundle_path),
        files={name: file_sha256(path) for name, path in files.items()},
        epochs=args.epochs, profile=profile, recipe=dict(RECIPE, batch_size=args.batch_size,
            p1_output_initialization='zero' if args.p1_zero_init else 'random'),
        performance=dict(sample_cache_gib=args.sample_cache_gib,
                         validation_batch_size=validation_batch_size, num_workers=0))
    backbone = _check_own_det(run_root / profile / 'det', binding) if args.phase == 'diff' else None
    return dict(schema_version='fwt_portable_preflight_v1', status='passed',
        phase=args.phase, profile=profile, training_started=False, heldout_evaluation=False,
        arrays_read=0, weights_deserialized=0, gpu_initialized=False,
        data_root=str(root), run_root=str(run_root), output=str(output),
        config_path=str(config_path), manifest=str(files['manifest']),
        normalizer=str(files['normalizer']), backbone=str(backbone) if backbone else None,
        role_counts=counts, mapped_weather_evidence_files=len(sources.mapping), binding=binding,
        note='File hashes/identity/stat checks only; original POWER semantic verification runs before training arrays.')


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute(args, checked):
    """One explicit fresh stage using the preserved engine, never an auto chain."""
    # Recheck immediately before importing ML libraries; no preflight receipt is an authorization.
    if not args.execute or preflight(args) != checked:
        raise ValueError('Explicit execution and unchanged preflight are required')
    import torch
    from fwt_v2 import artifacts, engine
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; choose a compatible torch build or --device cpu')
    previous_root = artifacts.WORKSPACE
    artifacts.WORKSPACE = Path(checked['run_root'])
    train_args = SimpleNamespace(manifest=checked['manifest'], normalizer=checked['normalizer'],
        output=checked['output'], epochs=args.epochs, batch_size=args.batch_size, learning_rate=.001,
        validation_batch_size=checked['binding']['performance']['validation_batch_size'],
        sample_cache_gib=args.sample_cache_gib,
        seed=0, device=args.device, loss='focal', samples=8, sampling_steps=4,
        sampler='ddim', architecture='fwt', backbone=checked['backbone'],
        resume=None, stop_after_epoch=None)
    try:
        # The on-disk bundle remains immutable and was checked twice above. Bind
        # this opt-in variant in both stages, then modify only this in-memory copy.
        configuration = read_json(checked['config_path'])
        if checked['binding']['recipe']['p1_output_initialization'] == 'zero':
            configuration['diffusion'] = dict(configuration['diffusion'], output_initialization='zero')
        with _working_directory(checked['data_root']):
            result = engine.train(train_args, configuration, args.phase)
        output = Path(checked['output'])
        diagnostic = read_json(output / 'diagnostics.json')
        selection = read_json(output / 'selection.json')
        if (diagnostic.get('status') != 'completed'
                or diagnostic.get('completed_epochs') != args.epochs
                or selection.get('selection_frozen') is not True):
            raise RuntimeError('Engine did not finish the requested frozen selection')
        names = ('best.pth', 'last.pth', 'selection.json', 'diagnostics.json',
                 'history.json', 'configuration.json', 'validation_selected.json')
        receipt = dict(schema_version='fwt_portable_training_completed_v1', status='completed',
            phase=args.phase, binding=checked['binding'],
            smoke_only=args.epochs != 30, heldout_evaluation=False,
            files={name: file_sha256(output / name) for name in names}, result=result)
        artifacts.atomic_json(output / 'portable_completed.json', receipt)
        return receipt
    finally:
        artifacts.WORKSPACE = previous_root


def main(argv=None):
    args = parser().parse_args(argv)
    checked = preflight(args)
    result = execute(args, checked) if args.execute else checked
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
