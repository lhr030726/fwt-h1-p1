#!/usr/bin/env python3
"""Paired best-49 compact BCE continuation; read-only dry-run by default.

Both arms inherit the same full AdamW/RNG state, then explicitly set constant
learning rates 1e-4 / 1e-5. Ten child epochs are absolute epochs 50--59. The
frozen pilot and extension schemas remain unchanged and are never patched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy, deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import extend_literature_ablation as extension

pilot = extension.pilot
SCHEMA = 'fwt_compact_continuation_state_v1'
TRANSACTION_SCHEMA = 'fwt_compact_continuation_transaction_v1'
PLAN_SCHEMA = 'fwt_compact_continuation_plan_v1'
SEED, PARENT_EPOCHS, CHILD_EPOCHS, TOTAL_EPOCHS = 4, 49, 10, 59
UPDATES_PER_EPOCH, PARENT_UPDATES = 51, 2499
ARMS = ('constant_lr', 'lower_lr')
LEARNING_RATES = {'constant_lr': 1e-4, 'lower_lr': 1e-5}
PARENT_COMMIT = '40417b28651b811bb359e5040706782d6648523a'
PARENT_BEST_SHA = '28855343cc3840f523a779860ba913bb4c35822868336ef2fa928b9598c23f17'
PARENT_AP = .26120831547850837
RESERVE_BYTES = 256 * 1024**2
read_json, sha, canonical_sha = pilot.read_json, pilot.sha, pilot.canonical_sha
serialize_state, deserialize_state = pilot.serialize_state, pilot.deserialize_state


def tensor_cpu_tree(value):
    """Own every saved tensor, including AdamW's already-CPU step scalars.

    This matches the frozen helper's cloning semantics explicitly. Moving a
    tensor to CPU alone is not a snapshot when that tensor already lives there.
    """
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensor_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_cpu_tree(item) for item in value)
    return deepcopy(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_files():
    return extension.source_files() | {'tools/continue_compact_refiner.py': sha(__file__)}


def continuation_order(train_ids, parent_orders):
    import torch
    require(len(train_ids) == len(set(train_ids)) == 811, 'Require exactly 811 unique Tr samples')
    require(len(parent_orders) == 50, 'Require all fifty parent batch orders')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    result = []
    for _ in range(TOTAL_EPOCHS):
        ids = [train_ids[i] for i in torch.randperm(811, generator=generator).tolist()]
        result.append([ids[i:i + 16] for i in range(0, 811, 16)])
    require(result[:50] == parent_orders, 'Regenerated first fifty orders differ from parent')
    for epoch in result:
        require([len(b) for b in epoch] == [16] * 50 + [11]
                and sorted(x for b in epoch for x in b) == sorted(train_ids), 'Invalid full Tr order')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run', 'output', 'diagnostic-report', 'experiment-plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def validate_diagnostic(path, parent_best):
    report = read_json(path)
    require(report.get('schema') == 'fwt_degradation_completed_v1'
            and report.get('status') == 'completed' and report.get('source_commit') == PARENT_COMMIT,
            'Require completed original-source Tr/V diagnostic report')
    preservation = report.get('preservation', {})
    for key in ('old_artifacts_sha256_before_after_equal', 'Tr_V_array_hashes_verified_before',
                'Tr_V_file_identity_before_after_equal', 'model_state_sha256_before_after_equal'):
        require(preservation.get(key) is True, 'Diagnostic preservation failed: ' + key)
    require(preservation.get('optimizer_steps_performed') == 0 and preservation.get('old_files_written') == 0,
            'Diagnostic must not train or change old files')
    checkpoint = report.get('checkpoint_set', {}).get('compact_bce/best', {})
    require(Path(checkpoint.get('path', '')).resolve() == parent_best
            and checkpoint.get('checkpoint_sha256') == sha(parent_best) == PARENT_BEST_SHA
            and checkpoint.get('completed_epochs') == PARENT_EPOCHS
            and checkpoint.get('updates') == PARENT_UPDATES, 'Diagnostic parent best identity differs')
    require(set(report.get('roles', {})) == {'residual_train', 'validation'}
            and all(v.get('status') == 'completed' for v in report['roles'].values()),
            'Diagnostic must complete both full Tr/V roles')
    diagnostic_plan = path.parent / 'diagnostic-plan.md'
    require(sha(diagnostic_plan) == report.get('frozen_diagnostic_plan_sha256'),
            'Frozen diagnostic plan identity differs')
    files = {str(path): sha(path), str(diagnostic_plan): sha(diagnostic_plan)}
    for name, digest in report.get('files', {}).items():
        target = (path.parent / name).resolve()
        require(path.parent in target.parents and target.suffix == '.json', 'Diagnostic file escapes directory')
        require(sha(target) == digest, 'Diagnostic file digest differs')
        files[str(target)] = digest
    return report, files


def validate_parent_source_files(expected, current, additional_source_files=()):
    """Preserve every frozen source digest while allowing only named additions.

    Existing continuation callers retain exact-set comparison by default. A
    downstream experiment may explicitly declare new modules discovered by the
    shared source inventory; it may not omit or change any historical file.
    """
    allowed = set(additional_source_files)
    require(len(allowed) == len(additional_source_files) and not allowed.intersection(expected),
            'Additional source allowlist must contain unique new filenames')
    require(set(current) == set(expected) | allowed,
            'Frozen parent source missing or unapproved additional source present')
    require(all(current[name] == digest for name, digest in expected.items()),
            'Frozen parent source digest differs')


def make_plan(args, *, additional_source_files=()):
    parent_path = args.parent_run / 'plan.json'
    parent = read_json(parent_path)
    receipt = read_json(args.parent_run / 'receipt.json')
    require(parent.get('schema') == 'fwt_literature_extension_plan_v1'
            and parent.get('stage') == 'extension' and parent.get('seeds') == [SEED]
            and parent.get('arms') == list(extension.ARMS)
            and parent.get('recipe') == dict(pilot.BASE_RECIPE, epochs=50)
            and receipt.get('schema') == 'fwt_literature_extension_completed_v1'
            and receipt.get('status') == 'completed' and receipt.get('completed_epochs') == 50
            and receipt.get('plan_sha256') == sha(parent_path), 'Require completed original fifty-epoch parent')
    validate_parent_source_files(parent['source_files'], extension.source_files(), additional_source_files)
    parent_root = args.parent_run.parent.parent
    files = dict(parent['input_files_sha256'])
    files.update({str(parent_root / name): value for name, value in parent['source_files'].items()})
    for name in ('plan.json', 'receipt.json', 'input-binding.json', 'migration.json', 'seed-004/transaction.pth'):
        files[str(args.parent_run / name)] = sha(args.parent_run / name)
    require(receipt['input_binding_sha256'] == files[str(args.parent_run / 'input-binding.json')]
            and receipt['transaction_sha256'] == files[str(args.parent_run / 'seed-004/transaction.pth')]
            and receipt['migration_sha256'] == files[str(args.parent_run / 'migration.json')],
            'Parent receipt identity differs')
    require(set(receipt['seeds']) == {'4'} and set(receipt['seeds']['4']) == set(extension.ARMS),
            'Parent receipt arm set differs')
    for arm, item in receipt['seeds']['4'].items():
        directory = args.parent_run / 'seed-004' / arm
        require(read_json(directory / 'receipt.json') == item and item['status'] == 'completed'
                and item['completed_epochs'] == 50 and item['total_updates'] == 2550,
                'Incomplete parent arm: ' + arm)
        files[str(directory / 'receipt.json')] = sha(directory / 'receipt.json')
        for name, digest in item['files'].items():
            require(Path(name).name == name, 'Unsafe parent artifact name')
            files[str(directory / name)] = digest
    selected_dir = args.parent_run / 'seed-004/compact_bce'
    best_path = selected_dir / 'best.pth'
    item = receipt['seeds']['4']['compact_bce']
    require(item['best_epoch'] == 48 and item['files']['best.pth'] == PARENT_BEST_SHA
            and abs(item['best_validation_ap'] - PARENT_AP) < 1e-12, 'Unexpected compact parent best')
    report, diagnostic_files = validate_diagnostic(args.diagnostic_report, best_path)
    files.update(diagnostic_files)
    files[str(args.experiment_plan)] = sha(args.experiment_plan)
    for path, digest in files.items():
        require(sha(path) == digest, 'Original source/input/artifact changed: ' + path)
    raw_manifest = read_json(parent['input_paths']['manifest'])
    require({r['role'] for r in raw_manifest['samples']} == {'backbone_train', 'residual_train', 'validation'},
            'Only original development manifest is accepted')
    for role, key, count in (('residual_train', 'training_sample_ids', 811),
                             ('validation', 'validation_sample_ids', 568)):
        ids = [r['sample_id'] for r in raw_manifest['samples'] if r['role'] == role]
        require(ids == parent[key] and len(ids) == len(set(ids)) == count, 'Original Tr/V identities differ')
    orders = continuation_order(parent['training_sample_ids'], parent['batch_orders']['4'])
    recipe = dict(parent_completed_epochs=49, child_epochs=10, absolute_final_epoch=59,
        batch_size=16, validation_batch_size=8, optimizer='AdamW', learning_rates=LEARNING_RATES,
        weight_decay=.01, precision='float32', cpu_threads=2, sample_cache_gib=16,
        loss='BCE_mean_all_valid_pixels', auxiliary_weight=0., augmentation='none',
        backbone='fixed_original_H1_no_grad_eval', checkpoint_selection='max_sigmoid_pooled_AP_earliest_tie',
        batch_order='seed4_regenerate59_verify50_use_indices49_to58',
        metrics='sigmoid_AP_and_raw_AP_event_equal_and_pooled_NLL_Brier_parent_logit_drift_v1',
        ema=False, gradient_clipping=False, schedule='two_fixed_learning_rates_after_exact_parent_restore',
        drift_scope='full_V_valid_pixels_all_positive_negative_against_parent_best49',
        replacement_gate=dict(ap_gain=.001, proper_score_max_ratio=1.05, positive_event_ap_max_drop=.01),
        comparison='last_three_child_epoch_mean_AP_lower_minus_constant', practical_difference=.001)
    plan = dict(schema=PLAN_SCHEMA, stage='compact_continuation', seed=SEED, arms=list(ARMS), recipe=recipe,
        parent_run=str(args.parent_run), parent_commit=PARENT_COMMIT,
        parent_plan_sha256=sha(parent_path), parent_receipt_sha256=sha(args.parent_run / 'receipt.json'),
        parent_transaction_sha256=files[str(args.parent_run / 'seed-004/transaction.pth')],
        parent_best_sha256=PARENT_BEST_SHA,
        parent_model_sha256=report['checkpoint_set']['compact_bce/best']['model_state_sha256'],
        parent_binding_sha256=canonical_sha(read_json(selected_dir / 'configuration.json')),
        parent_selected_validation=read_json(selected_dir / 'validation_selected.json')['validation'],
        parent_source_files=parent['source_files'], source_files=source_files(), input_files_sha256=files,
        diagnostic_report=str(args.diagnostic_report), diagnostic_report_sha256=sha(args.diagnostic_report),
        experiment_plan=str(args.experiment_plan), experiment_plan_sha256=sha(args.experiment_plan),
        parent_completed_epochs=49, child_epochs=10, parent_updates=2499, child_updates=510,
        lifetime_updates=3009, absolute_final_epoch=59,
        all_batch_orders=orders, batch_orders=orders[49:59],
        batch_order_sha256=canonical_sha(orders[49:59]), all_batch_order_sha256=canonical_sha(orders),
        parent_batch_order_sha256=parent['batch_order_sha256']['4'], input_paths=parent['input_paths'],
        output=str(args.output), resume_run=str(args.resume_run) if args.resume_run else None,
        dry_run=not args.execute,
        limitations=['Single-seed Tr/V development intervention, not independent validation.',
            'Fork from best49; original last50 and all historical artifacts remain unchanged.',
            'Inherited best is a fallback, not a new improvement; no automatic next experiment.'])
    for key in ('training_sample_ids', 'validation_sample_ids', 'diffusion_config', 'data_config',
                'model_config', 'manifest_sha256', 'normalizer_sha256', 'bound_data_metadata_paths'):
        plan[key] = parent[key]
    if args.resume_run:
        previous = read_json(args.resume_run / 'plan.json')
        ignore = {'output', 'resume_run', 'dry_run'}
        require(canonical_sha({k: v for k, v in previous.items() if k not in ignore}) == canonical_sha(
                {k: v for k, v in plan.items() if k not in ignore}), 'Resume plan/source/data/budget differs')
    return plan


def migrate_optimizer_lr(optimizer, arm):
    """Only LR may change after original-state verification; preserve every moment."""
    from fwt_v2.training_state import optimizer_state_integrity
    require(arm in ARMS, 'Unknown branch')
    before = tensor_cpu_tree(optimizer.state_dict())
    require(all(group['lr'] == 1e-4 and group['weight_decay'] == .01
                for group in before['param_groups']), 'LR migration requires original AdamW settings')
    original_integrity = optimizer_state_integrity(before)
    for group in optimizer.param_groups:
        group['lr'] = LEARNING_RATES[arm]
    after = tensor_cpu_tree(optimizer.state_dict())
    restored_groups = deepcopy(after['param_groups'])
    for group in restored_groups:
        group['lr'] = 1e-4
    require(optimizer_state_integrity(dict(after, param_groups=restored_groups)) == original_integrity,
            'Migration changed AdamW fields other than LR')
    return dict(source_optimizer_integrity=original_integrity,
        migrated_optimizer_integrity=optimizer_state_integrity(after),
        changed_fields=[] if arm == 'constant_lr' else ['param_groups[*].lr'],
        source_lr=1e-4, target_lr=LEARNING_RATES[arm])


def restore_parent_state(state, model, optimizer, binding, device, parent_history):
    """Verify the frozen schema, giving AdamW exclusively owned tensor copies.

    Optimizer.load_state_dict may retain input tensor references (notably in
    PyTorch 2.7). Never hand it the shared parent checkpoint or an audit snapshot.
    """
    extension.restore_state(deepcopy(state), model, optimizer, binding, device, parent_history)


def make_state(model, optimizer, binding, history, baseline, device):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import capture_rng, optimizer_state_integrity
    n = len(history)
    state = dict(schema=SCHEMA, checkpoint_kind='full_training_state', binding=deepcopy(binding),
        completed_child_epochs=n, child_updates=n * 51, parent_updates=PARENT_UPDATES,
        parent_completed_epochs=PARENT_EPOCHS, absolute_completed_epochs=PARENT_EPOCHS + n,
        lifetime_updates=PARENT_UPDATES + n * 51, updates=PARENT_UPDATES + n * 51,
        learning_rate=LEARNING_RATES[binding['branch_id']],
        model_state_dict=tensor_cpu_tree(model.state_dict()), model_state_sha256=model_state_digest(model),
        optimizer_state_dict=tensor_cpu_tree(optimizer.state_dict()), rng_state=capture_rng(device),
        history=deepcopy(history), baseline_validation=deepcopy(baseline),
        validation=deepcopy(history[-1]['validation'] if history else baseline),
        origin='new_child_epoch' if n else 'inherited_best49')
    state['optimizer_integrity'] = optimizer_state_integrity(state['optimizer_state_dict'])
    return state


def check_state(state, arm, binding=None):
    from fwt_v2.training_state import verify_optimizer_integrity
    n = state.get('completed_child_epochs')
    require(state.get('schema') == SCHEMA and state.get('binding', {}).get('branch_id') == arm
            and state['binding'].get('seed') == SEED and state['binding'].get('architecture_arm') == 'compact_bce',
            'Continuation state schema/branch differs')
    require(binding is None or state['binding'] == binding, 'Continuation state binding differs')
    require(type(n) is int and 0 <= n <= CHILD_EPOCHS and state.get('child_updates') == n * 51
            and state.get('parent_updates') == PARENT_UPDATES and state.get('parent_completed_epochs') == 49
            and state.get('lifetime_updates') == state.get('updates') == PARENT_UPDATES + n * 51
            and state.get('absolute_completed_epochs') == PARENT_EPOCHS + n
            and len(state.get('history', [])) == n, 'Continuation progress/lifetime updates differ')
    require(state.get('learning_rate') == LEARNING_RATES[arm]
            and state['optimizer_state_dict']['param_groups'] == state['binding']['optimizer_param_groups']
            and all(g['lr'] == LEARNING_RATES[arm] and g['weight_decay'] == .01
                    for g in state['optimizer_state_dict']['param_groups']), 'Bound AdamW groups/LR differ')
    verify_optimizer_integrity(state)
    require([r['child_epoch'] for r in state['history']] == list(range(1, n + 1))
            and all(r['absolute_epoch'] == PARENT_EPOCHS + r['child_epoch']
                    and r['lifetime_updates'] == PARENT_UPDATES + 51 * r['child_epoch']
                    and r['learning_rate'] == LEARNING_RATES[arm] for r in state['history']),
            'Child history progress differs')
    expected = state['history'][-1]['validation'] if n else state['baseline_validation']
    require(state['validation'] == expected and math.isfinite(expected['pooled_ap']), 'State validation differs')
    require(state['origin'] == ('new_child_epoch' if n else 'inherited_best49'), 'State origin differs')


def restore_state(state, model, optimizer, binding, device):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import restore_training_parameters, validate_rng, restore_rng
    check_state(state, binding['branch_id'], binding)
    validate_rng(state['rng_state'], device)
    owned = deepcopy(state)
    restore_training_parameters(dict(owned, selected={'state_dict': owned['model_state_dict']}), model, optimizer)
    require(model_state_digest(model) == state['model_state_sha256'], 'Restored model digest differs')
    restore_rng(state['rng_state'], device)


def build_transaction(states, best_bytes, new_best_bytes, rng_state):
    import torch
    require(set(states) == set(best_bytes) == set(new_best_bytes) == set(ARMS), 'Paired transaction requires both arms')
    counts = {state['completed_child_epochs'] for state in states.values()}
    require(len(counts) == 1, 'Paired transaction child epochs differ')
    arms = {}
    for arm in ARMS:
        last = deepcopy(states[arm])
        last['selected_checkpoint_sha256'] = hashlib.sha256(best_bytes[arm]).hexdigest()
        last['new_selected_checkpoint_sha256'] = (hashlib.sha256(new_best_bytes[arm]).hexdigest()
                                                  if new_best_bytes[arm] is not None else None)
        entry = {}
        for kind, data in (('last', serialize_state(last)), ('best', best_bytes[arm]), ('new_best', new_best_bytes[arm])):
            if data is not None:
                entry[kind + '_data'] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                entry[kind + '_sha256'] = hashlib.sha256(data).hexdigest()
        arms[arm] = entry
    result = dict(schema=TRANSACTION_SCHEMA, seed=SEED, completed_child_epochs=counts.pop(), arms=arms, rng_state=rng_state)
    decode_transaction(result)
    return result


def decode_transaction(transaction, bindings=None):
    require(transaction.get('schema') == TRANSACTION_SCHEMA and transaction.get('seed') == SEED
            and set(transaction.get('arms', {})) == set(ARMS)
            and type(transaction.get('completed_child_epochs')) is int
            and 0 <= transaction['completed_child_epochs'] <= CHILD_EPOCHS, 'Transaction schema/arms/progress differs')
    n = transaction['completed_child_epochs']
    decoded = {}
    for arm, entry in transaction['arms'].items():
        require(set(entry) == ({'last_data', 'last_sha256', 'best_data', 'best_sha256'} |
                ({'new_best_data', 'new_best_sha256'} if n else set())), 'Transaction payload set differs')
        states = {kind: deserialize_state(pilot.transaction_checkpoint_bytes(entry, kind))
                  for kind in ('last', 'best', 'new_best') if kind + '_data' in entry}
        last, best = states['last'], states['best']
        for state in states.values():
            check_state(state, arm, bindings[arm] if bindings else last['binding'])
            count = state['completed_child_epochs']
            require(count <= n and state['history'] == last['history'][:count]
                    and state['baseline_validation'] == last['baseline_validation'], 'Checkpoint history prefix differs')
        require(last['completed_child_epochs'] == n
                and last['selected_checkpoint_sha256'] == entry['best_sha256'], 'Last progress/best identity differs')
        aps = [last['baseline_validation']['pooled_ap']] + [r['validation']['pooled_ap'] for r in last['history']]
        require(best['completed_child_epochs'] == max(range(len(aps)), key=lambda i: aps[i]), 'Inherited/new best selection differs')
        if n:
            new = states['new_best']
            require(new['completed_child_epochs'] == 1 + max(range(n), key=lambda i: aps[i + 1])
                    and last['new_selected_checkpoint_sha256'] == entry['new_best_sha256'], 'New-only best selection differs')
        else:
            require(last.get('new_selected_checkpoint_sha256') is None, 'Initial transaction cannot have a new best')
        decoded[arm] = states
    return decoded


def publish_transaction(directory, transaction):
    from fwt_v2.artifacts import atomic_checkpoint
    decode_transaction(transaction)
    atomic_checkpoint(directory / 'transaction.pth', transaction)
    for arm, entry in transaction['arms'].items():
        for kind in ('best', 'last', 'new_best'):
            if kind + '_data' in entry:
                pilot.atomic_bytes(directory / arm / (kind + '.pth'), pilot.transaction_checkpoint_bytes(entry, kind))


def restore_transaction(transaction, heads, optimizers, bindings, device):
    from fwt_v2.training_state import validate_rng, restore_rng
    decoded = decode_transaction(transaction, bindings)
    validate_rng(transaction['rng_state'], device)
    for arm, states in decoded.items():
        # Validate complete optimizer/model states for selected payloads as well.
        for kind in ('best', 'new_best', 'last'):
            if kind in states:
                restore_state(states[kind], heads[arm], optimizers[arm], bindings[arm], device)
    restore_rng(transaction['rng_state'], device)
    return decoded


def paired_validation_records(backbone, heads, parent_head, loader, device):
    """One frozen H1 forward per V batch; parent head supplies a fixed anchor."""
    import torch
    from fwt_v2.engine import records_from_batch, to_device
    backbone.eval()
    parent_head.eval()
    for head in heads.values():
        head.eval()
    result = {name: [] for name in heads}
    result['parent'] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = to_device(cpu_batch, device)
            outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
            parent_logits = parent_head(outputs, batch['physics'])
            parent_records = list(records_from_batch(batch, parent_logits.sigmoid(), parent_logits))
            result['parent'].extend(parent_records)
            for arm, head in heads.items():
                logits = head(outputs, batch['physics'])
                rows = list(records_from_batch(batch, logits.sigmoid(), logits))
                for row, anchor in zip(rows, parent_records):
                    row['parent_calibration_logit'] = anchor['calibration_logit']
                result[arm].extend(rows)
    return result


def extended_metrics(records):
    """Retain historical sigmoid AP/proper scores; add explicitly named probes."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from fwt_v2.protocol import _validated
    rows = _validated(records)
    result = pilot.epoch_metrics(rows)
    zs, ys = [], []
    nll_sum = brier_sum = 0.
    stats = {name: dict(count=0, logit_sum=0., logit_square=0., nll_sum=0., brier_sum=0., drift_sum=0.,
                       drift_abs=0., drift_square=0.) for name in ('all', 'positive', 'negative')}
    has_anchor = ['parent_calibration_logit' in r for r in rows]
    require(all(has_anchor) or not any(has_anchor), 'Mixed anchor availability')
    for row in rows:
        valid = np.asarray(row['valid_mask'], bool)
        z = np.asarray(row['calibration_logit'], np.float64)[valid]
        y = np.asarray(row['target'])[valid] > .5
        p = np.asarray(row['probability'], np.float64)[valid]
        require(np.isfinite(z).all(), 'Nonfinite raw validation logit')
        zs.append(z); ys.append(y)
        pixel_nll, pixel_brier = np.logaddexp(0, z) - y * z, (p - y) ** 2
        nll_sum += float(pixel_nll.sum(dtype=np.float64))
        brier_sum += float(pixel_brier.sum(dtype=np.float64))
        delta = None
        if all(has_anchor):
            anchor = np.asarray(row['parent_calibration_logit'], np.float64)
            require(anchor.shape == valid.shape and np.isfinite(anchor).all(), 'Invalid parent logit anchor')
            delta = z - anchor[valid]
        for name, selected in (('all', np.ones(len(y), dtype=bool)), ('positive', y), ('negative', ~y)):
            values, current = z[selected], stats[name]
            current['count'] += len(values)
            current['logit_sum'] += float(values.sum(dtype=np.float64))
            current['logit_square'] += float(np.square(values).sum(dtype=np.float64))
            current['nll_sum'] += float(pixel_nll[selected].sum(dtype=np.float64))
            current['brier_sum'] += float(pixel_brier[selected].sum(dtype=np.float64))
            if delta is not None:
                d = delta[selected]
                current['drift_sum'] += float(d.sum(dtype=np.float64))
                current['drift_abs'] += float(np.abs(d).sum(dtype=np.float64))
                current['drift_square'] += float(np.square(d).sum(dtype=np.float64))
    z, y = np.concatenate(zs), np.concatenate(ys)
    require(len(y) > 0 and y.any(), 'Validation must contain valid positives')
    result.update(pooled_raw_logit_ap=float(average_precision_score(y, z)),
                  pooled_nll=nll_sum / len(y), pooled_brier=brier_sum / len(y),
                  ranking_selection_score='pooled_ap_sigmoid_float32', raw_logit_statistics={},
                  parent_logit_drift={} if all(has_anchor) else None)
    for name, current in stats.items():
        count = current['count']
        result['raw_logit_statistics'][name] = dict(count=count,
            mean=current['logit_sum'] / count if count else None,
            rms=math.sqrt(current['logit_square'] / count) if count else None,
            nll=current['nll_sum'] / count if count else None,
            brier=current['brier_sum'] / count if count else None)
        if all(has_anchor):
            result['parent_logit_drift'][name] = dict(count=count,
                mean=current['drift_sum'] / count if count else None,
                mean_absolute=current['drift_abs'] / count if count else None,
                rms=math.sqrt(current['drift_square'] / count) if count else None)
    return result


def check_records(records, plan, domain_sha):
    require([r['sample_id'] for r in records] == plan['validation_sample_ids']
            and pilot.records_identity(records) == domain_sha, 'Full-V sample/target/mask identity changed')


def check_metric_replay(actual, expected):
    for key in ('pooled_ap', 'nll', 'brier'):
        require(abs(actual[key] - expected[key]) <= 1e-7, 'Parent/selected V replay failed: ' + key)


def select_development_candidate(arm_receipts, parent_best_sha=PARENT_BEST_SHA):
    """Apply the frozen gate/tie rule only to newly trained selected states."""
    require(set(arm_receipts) == set(ARMS), 'Selection requires both completed arms')
    eligible = [arm for arm in ARMS if
                arm_receipts[arm]['new_best']['qualifies_development_replacement_gate'] is True]
    common = dict(development_only=True, independent_evaluation=False, confirmation_required=True,
                  practical_ap_margin=.001, eligible_new_best_arms=eligible)
    if not eligible:
        return dict(common, decision='retained_parent', candidate=None,
                    checkpoint_sha256=parent_best_sha, reason='No new-only best passes all development gates')
    if len(eligible) == 1:
        selected, reason = eligible[0], 'Only one new-only best passes all development gates'
    else:
        ap = {a: arm_receipts[a]['new_best']['validation']['pooled_ap'] for a in ARMS}
        if abs(ap['lower_lr'] - ap['constant_lr']) >= .001:
            selected, reason = max(ARMS, key=ap.get), 'Eligible new-best AP difference is at least 0.001'
        else:
            means = {a: arm_receipts[a]['last_three_mean_ap'] for a in ARMS}
            selected = 'lower_lr' if means['lower_lr'] >= means['constant_lr'] else 'constant_lr'
            reason = ('Eligible new-best AP difference below 0.001; last-three mean decides, '
                      'exact mean tie prefers lower_lr')
    return dict(common, decision='new_development_candidate', candidate=selected,
                checkpoint_kind='new_best', checkpoint_sha256=arm_receipts[selected]['new_best']['checkpoint_sha256'],
                reason=reason)


def write_status(args, phase, **details):
    from fwt_v2.artifacts import atomic_json
    value = dict(schema='fwt_compact_continuation_status_v1', phase=phase,
        total_child_epochs=10, parent_completed_epochs=49, arms=list(ARMS),
        started_at=args._run_started_at, updated_at=time.time(),
        resources=pilot.old.resource_snapshot(), **details)
    atomic_json(args.output / 'status.json', value)
    with (args.output / 'resource-history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def check_disk(path, transaction=None):
    while not path.exists():
        path = path.parent
    required = RESERVE_BYTES
    if transaction is not None:
        # Peak new transaction + mirrors, with headroom for longer small history.
        required += 4 * len(serialize_state(transaction))
    require(shutil.disk_usage(path).free >= required, 'Insufficient disk for paired transaction and 256 MiB reserve')


def execute(args, plan):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.artifacts import atomic_json, output_path
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import audited_manifest, load_det, model_state_digest, seed_everything, to_device
    from fwt_v2.gradient_diagnostics import gradients_finite, gradient_l2_norm
    from fwt_v2.literature_refiner import masked_refinement_loss
    from fwt_v2.manifest import manifest_digest, samples_for_role
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    from fwt_v2.training_state import capture_rng, restore_rng, sample_file_identity, optimizer_state_integrity

    output_path(args.output / 'plan.json')
    require(not args.output.exists(), 'Output must be a fresh directory, including resume')
    require(args.parent_run not in args.output.parents and args.output != args.parent_run,
            'Continuation output must not be inside parent run')
    pilot.old.assert_gpu_idle()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA GPU required')
    check_disk(args.output)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    device = torch.device('cuda:0')
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_created, args._run_started_at = True, time.time()
    atomic_json(args.output / 'plan.json', plan)
    write_status(args, 'verify_data', step='full_original_Tr_V_hashes')
    paths = {k: Path(v) for k, v in plan['input_paths'].items()}
    os.chdir(paths['manifest'].parent)
    manifest = audited_manifest(paths['manifest'])
    require(manifest_digest(manifest) == plan['manifest_sha256']
            and sha(paths['normalizer']) == plan['normalizer_sha256'], 'Data manifest/normalizer differs')
    train_rows, val_rows = (samples_for_role(manifest, role) for role in ('residual_train', 'validation'))
    require([r['sample_id'] for r in train_rows] == plan['training_sample_ids']
            and [r['sample_id'] for r in val_rows] == plan['validation_sample_ids'], 'Tr/V sample identities differ')
    samples = train_rows + val_rows
    sample_identity = sample_file_identity(samples)
    array_hashes = {str(Path(row[k]).resolve()): sha(row[k]) for row in samples for k in ('input_path', 'label_path')}
    pilot.old.verify_selected_array_index(read_json(plan['bound_data_metadata_paths']['arrays_index']),
                                         samples, array_hashes, paths['manifest'].parent)
    dataset = RCDADataset(manifest, 'residual_train', NormalizationStats.load(paths['normalizer']),
                          DataConfig(**plan['data_config']))
    val_dataset = copy(dataset)
    val_dataset.role, val_dataset.samples = 'validation', val_rows
    cache = SharedTensorCache(16 * 1024**3)
    training, validation = CachedDataset(dataset, cache), CachedDataset(val_dataset, cache)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
    seed_everything(0)
    backbone, bundle = load_det(paths['backbone'], device)
    require(bundle.get('selection_frozen') and bundle['family'] == 'det'
            and all(canonical_sha(bundle[k]) == canonical_sha(plan[k]) for k in
                    ('data_config', 'model_config', 'manifest_sha256', 'normalizer_sha256')), 'Fixed H1 metadata differs')
    backbone.requires_grad_(False).eval()
    h1_sha = model_state_digest(backbone)
    original_common = read_json(args.parent_run / 'input-binding.json')
    runtime = dict(python=sys.version, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                   cpu_threads=2, interop_threads=2, matmul_tf32=False, cudnn_tf32=True)
    for key, value in dict(array_files_sha256=array_hashes, array_stat_identity=sample_identity,
            frozen_backbone_state_sha256=h1_sha, runtime=runtime).items():
        require(canonical_sha(value) == canonical_sha(original_common[key]), 'Parent data/runtime/H1 differs: ' + key)
    parent_plan = read_json(args.parent_run / 'plan.json')
    original = read_json(args.parent_run / 'seed-004/compact_bce/configuration.json')
    require(canonical_sha(original) == plan['parent_binding_sha256']
            and all(canonical_sha(original[k]) == canonical_sha(v) for k, v in original_common.items()),
            'Parent configuration binding differs')
    transaction = torch.load(args.parent_run / 'seed-004/transaction.pth', map_location='cpu', weights_only=True)
    parent_decoded = extension.decode_transaction(transaction)
    require(transaction['completed_epochs'] == 50, 'Parent transaction incomplete')
    parent_state = parent_decoded['compact_bce'][1]
    require(transaction['arms']['compact_bce']['best_sha256'] == PARENT_BEST_SHA
            and parent_state['completed_epochs'] == 49 and parent_state['updates'] == 2499
            and parent_state['model_state_sha256'] == plan['parent_model_sha256'], 'Parent best full state differs')
    parent_history = read_json(Path(parent_plan['pilot_run']) / 'seed-004/compact_bce/history.json')
    config = DiffusionConfig(**dict(plan['diffusion_config'], output_initialization='zero', prediction_type='clean_probability'))
    template, initial_sha, _ = pilot.initialize_head(SEED, 'compact_bce', paths['backbone'], device, config, h1_sha)
    require(initial_sha == original['initial_head_state_sha256']
            and template.configuration() == original['head_configuration']
            and sum(p.numel() for p in template.parameters()) == original['head_parameters']
            and original['auxiliary_weight'] == 0., 'Compact head architecture/initialization differs')
    template_optimizer = torch.optim.AdamW(template.parameters(), lr=1e-4, weight_decay=.01)
    require(canonical_sha(template_optimizer.defaults) == canonical_sha(original['optimizer_defaults']),
            'Original AdamW defaults differ')
    restore_parent_state(parent_state, template, template_optimizer, original, device, parent_history)
    parent_head = deepcopy(template).requires_grad_(False).eval()
    heads, optimizers, migrations = {}, {}, {}
    for arm in ARMS:
        heads[arm] = deepcopy(template)
        optimizers[arm] = torch.optim.AdamW(heads[arm].parameters(), lr=1e-4, weight_decay=.01)
        restore_parent_state(parent_state, heads[arm], optimizers[arm], original, device, parent_history)
        migrations[arm] = migrate_optimizer_lr(optimizers[arm], arm)
        require(model_state_digest(heads[arm]) == plan['parent_model_sha256'], 'Branch failed exact model restoration')
    del template, template_optimizer, parent_decoded, transaction
    restore_rng(parent_state['rng_state'], device)
    write_status(args, 'parent_replay', step='full_V_parent_best49_and_identical_branches')
    records = paired_validation_records(backbone, heads, parent_head, val_loader, device)
    domain_sha = pilot.records_identity(records['parent'])
    require(domain_sha == original_common['target_and_masks_sha256'], 'Parent validation masks/targets differ')
    baseline = extended_metrics(records['parent'])
    baseline.update(pilot.selected_positive_event_ap(records['parent']))
    check_metric_replay(baseline, plan['parent_selected_validation'])
    for name, rows in records.items():
        check_records(rows, plan, domain_sha)
        if name != 'parent':
            metrics = extended_metrics(rows)
            check_metric_replay(metrics, baseline)
            require(metrics['parent_logit_drift']['all']['rms'] == 0., 'Initial branch is not identical to parent')
    del records
    # A no-update replay must not alter optimizer state; recheck after all initialization.
    for arm in ARMS:
        require(optimizer_state_integrity(tensor_cpu_tree(optimizers[arm].state_dict())) ==
                migrations[arm]['migrated_optimizer_integrity'], 'Replay altered AdamW state')
    restore_rng(parent_state['rng_state'], device)
    common = deepcopy(original_common)
    common.update(recipe=plan['recipe'], stage='compact_continuation', source_files=plan['source_files'],
        input_files_sha256=plan['input_files_sha256'], parent_best_sha256=PARENT_BEST_SHA,
        parent_model_sha256=plan['parent_model_sha256'], diagnostic_report_sha256=plan['diagnostic_report_sha256'],
        experiment_plan_sha256=plan['experiment_plan_sha256'])
    atomic_json(args.output / 'input-binding.json', common)
    atomic_json(args.output / 'baseline.json', dict(schema='fwt_compact_continuation_baseline_v1',
        parent_best_sha256=PARENT_BEST_SHA, validation=baseline, target_and_masks_sha256=domain_sha))
    bindings, histories, states, best_bytes, new_best_bytes = {}, {}, {}, {}, {}
    for arm in ARMS:
        (args.output / arm).mkdir()
        binding = deepcopy(original)
        binding.update(common)
        binding.update(branch_id=arm, architecture_arm='compact_bce', seed=SEED,
            batch_order_sha256=plan['batch_order_sha256'], parent_binding_sha256=plan['parent_binding_sha256'],
            optimizer_param_groups=deepcopy(optimizers[arm].state_dict()['param_groups']),
            intervention=dict(parent_epochs=49, parent_updates=2499, child_epochs=10, child_updates=510,
                lifetime_updates=3009, learning_rate=LEARNING_RATES[arm], optimizer_migration=migrations[arm]))
        bindings[arm], histories[arm] = binding, []
        atomic_json(args.output / arm / 'configuration.json', binding)
        state = make_state(heads[arm], optimizers[arm], binding, [], baseline, device)
        states[arm], best_bytes[arm], new_best_bytes[arm] = state, serialize_state(state), None
    transaction = build_transaction(states, best_bytes, new_best_bytes, capture_rng(device))
    resume_sha = None
    if args.resume_run:
        previous = args.resume_run / 'transaction.pth'
        resume_sha = sha(previous)
        transaction = torch.load(previous, map_location='cpu', weights_only=True)
    decoded = restore_transaction(transaction, heads, optimizers, bindings, device)
    start_epoch = transaction['completed_child_epochs']
    for arm in ARMS:
        histories[arm] = deepcopy(decoded[arm]['last']['history'])
        best_bytes[arm] = pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
        new_best_bytes[arm] = (pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'new_best') if start_epoch else None)
    atomic_json(args.output / 'migration.json', dict(schema='fwt_compact_continuation_migration_v1',
        parent_checkpoint_sha256=PARENT_BEST_SHA, parent_model_sha256=plan['parent_model_sha256'],
        parent_optimizer_integrity=parent_state['optimizer_integrity'],
        parent_rng_sha256=hashlib.sha256(serialize_state(parent_state['rng_state'])).hexdigest(),
        parent_updates=2499, parent_completed_epochs=49, branch_migrations=migrations,
        resume_transaction_sha256=resume_sha, restored_child_epochs=start_epoch,
        preserved_parent_history_sha256=canonical_sha(parent_state['history']),
        allowed_changes=['new_schema_and_provenance', 'branch_id', 'child_budget_and_order_extension', 'declared_lr_only']))
    del decoded, parent_state
    check_disk(args.output, transaction)
    publish_transaction(args.output, transaction)
    for arm in ARMS:
        atomic_json(args.output / arm / 'history.json', histories[arm])

    def verify_unchanged():
        require(source_files() == plan['source_files'], 'Continuation source changed')
        require(all(sha(path) == digest for path, digest in plan['input_files_sha256'].items()),
                'Bound original source/checkpoint/metadata changed')
        require(sample_file_identity(samples) == sample_identity, 'Tr/V file identity changed')
        require(model_state_digest(backbone) == h1_sha
                and all(not p.requires_grad and p.grad is None for p in backbone.parameters()), 'Frozen H1 changed/acquired gradients')
        require(model_state_digest(parent_head) == plan['parent_model_sha256']
                and all(not p.requires_grad and p.grad is None for p in parent_head.parameters()), 'Parent anchor changed/acquired gradients')

    id_index = {r['sample_id']: i for i, r in enumerate(train_rows)}
    for child_index in range(start_epoch, CHILD_EPOCHS):
        verify_unchanged()
        check_disk(args.output, transaction)
        for head in heads.values():
            head.train()
        batches = [[id_index[sid] for sid in batch] for batch in plan['batch_orders'][child_index]]
        loader = DataLoader(training, batch_sampler=batches, num_workers=0, generator=torch.Generator().manual_seed(0))
        write_status(args, 'training', child_epoch=child_index + 1, absolute_epoch=50 + child_index,
                     completed_child_epochs=child_index, batch=0, batches=51)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses, norms, intervals, observed = {a: [] for a in ARMS}, {a: [] for a in ARMS}, defaultdict(list), []
        for batch_index, cpu_batch in enumerate(loader):
            require(list(cpu_batch['sample_id']) == plan['batch_orders'][child_index][batch_index], 'Actual shared batch differs')
            batch = to_device(cpu_batch, device)
            with torch.no_grad(), pilot.cuda_interval(intervals, 'shared_H1'):
                outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
            for arm in ARMS:
                optimizer, head = optimizers[arm], heads[arm]
                require(all(g['lr'] == LEARNING_RATES[arm] and g['weight_decay'] == .01
                        for g in optimizer.param_groups), 'Unexpected training LR/weight decay')
                optimizer.zero_grad(set_to_none=True)
                with pilot.cuda_interval(intervals, arm + '/forward_and_loss'):
                    logits = head(outputs, batch['physics'])
                    loss = masked_refinement_loss(logits, batch['target'], batch['valid_mask'], aux_weight=0.)['bce']
                require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Nonfinite/nonscalar BCE')
                with pilot.cuda_interval(intervals, arm + '/backward'):
                    loss.backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                require(bool(gradients) and gradients_finite(gradients), 'Missing/nonfinite gradients')
                norm = gradient_l2_norm(gradients)
                require(math.isfinite(norm) and norm > 0, 'Invalid gradient norm')
                with pilot.cuda_interval(intervals, arm + '/optimizer'):
                    optimizer.step()
                losses[arm].append(float(loss.detach())); norms[arm].append(norm)
            observed.append(list(cpu_batch['sample_id']))
            if (batch_index + 1) % 10 == 0:
                write_status(args, 'training', child_epoch=child_index + 1, absolute_epoch=50 + child_index,
                    completed_child_epochs=child_index, batch=batch_index + 1, batches=51,
                    lifetime_updates=2499 + child_index * 51 + batch_index + 1)
        require(observed == plan['batch_orders'][child_index], 'Incomplete shared epoch')
        torch.cuda.synchronize()
        train_seconds, gpu_seconds = time.perf_counter() - started, pilot.cuda_elapsed_seconds(intervals)
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        write_status(args, 'validation', child_epoch=child_index + 1, absolute_epoch=50 + child_index,
                     completed_child_epochs=child_index)
        validation_started = time.perf_counter()
        records = paired_validation_records(backbone, heads, parent_head, val_loader, device)
        for rows in records.values():
            check_records(rows, plan, domain_sha)
        # Parent weights/identity are checked separately; avoid another full AP
        # sort each epoch solely for a frozen anchor. Its logits supply drift.
        states = {}
        for arm in ARMS:
            metrics = extended_metrics(records[arm])
            histories[arm].append(dict(child_epoch=child_index + 1, absolute_epoch=50 + child_index,
                lifetime_updates=2499 + (child_index + 1) * 51, child_updates=(child_index + 1) * 51,
                learning_rate=LEARNING_RATES[arm], training_loss=sum(losses[arm]) / 51, validation=metrics,
                training_checks=dict(updates=51, loss_finite=True, gradients_finite=True,
                    gradient_l2_min=min(norms[arm]), gradient_l2_max=max(norms[arm]),
                    shared_batch_order_sha256=canonical_sha(observed), shared_training_seconds=train_seconds,
                    shared_training_samples_per_second=811 / train_seconds,
                    shared_H1_cuda_stream_seconds=gpu_seconds['shared_H1'],
                    head_cuda_stream_seconds={phase: gpu_seconds[arm + '/' + phase]
                        for phase in ('forward_and_loss', 'backward', 'optimizer')},
                    timing_scope='CUDA stream elapsed includes launch gaps, not GPU busy time',
                    gpu_peak_allocated_bytes=peak_allocated, gpu_peak_reserved_bytes=peak_reserved)))
            state = make_state(heads[arm], optimizers[arm], bindings[arm], histories[arm], baseline, device)
            states[arm] = state
            if metrics['pooled_ap'] > deserialize_state(best_bytes[arm])['validation']['pooled_ap']:
                best_bytes[arm] = serialize_state(state)
            if new_best_bytes[arm] is None or metrics['pooled_ap'] > deserialize_state(new_best_bytes[arm])['validation']['pooled_ap']:
                new_best_bytes[arm] = serialize_state(state)
        del records
        verify_unchanged()
        transaction = build_transaction(states, best_bytes, new_best_bytes, capture_rng(device))
        check_disk(args.output, transaction)
        publish_transaction(args.output, transaction)
        for arm in ARMS:
            atomic_json(args.output / arm / 'history.json', histories[arm])
        write_status(args, 'training', step='paired_epoch_committed', child_epoch=child_index + 1,
            absolute_epoch=50 + child_index, completed_child_epochs=child_index + 1,
            lifetime_updates=2499 + (child_index + 1) * 51,
            validation_seconds=time.perf_counter() - validation_started,
            last_validation_ap={a: histories[a][-1]['validation']['pooled_ap'] for a in ARMS})
        print(json.dumps(dict(event='paired_epoch_completed', child_epoch=child_index + 1,
            absolute_epoch=50 + child_index, last_validation_ap={a: histories[a][-1]['validation']['pooled_ap'] for a in ARMS})), flush=True)

    # Re-evaluate both overall (parent-eligible) and new-only selected models.
    selected_reports = {arm: {} for arm in ARMS}
    for kind, payloads in (('overall_best', best_bytes), ('new_best', new_best_bytes)):
        selected = {}
        for arm in ARMS:
            selected[arm] = deserialize_state(payloads[arm])
            heads[arm].load_state_dict(selected[arm]['model_state_dict'], strict=True)
            require(model_state_digest(heads[arm]) == selected[arm]['model_state_sha256'], 'Selected model digest differs')
        records = paired_validation_records(backbone, heads, parent_head, val_loader, device)
        for arm in ARMS:
            check_records(records[arm], plan, domain_sha)
            metrics = extended_metrics(records[arm])
            check_metric_replay(metrics, selected[arm]['validation'])
            metrics.update(pilot.selected_positive_event_ap(records[arm]))
            state = selected[arm]
            qualifies = (state['completed_child_epochs'] > 0 and metrics['pooled_ap'] >= baseline['pooled_ap'] + .001
                and metrics['nll'] <= baseline['nll'] * 1.05 and metrics['brier'] <= baseline['brier'] * 1.05
                and metrics['positive_event_ap'] >= baseline['positive_event_ap'] - .01)
            selected_reports[arm][kind] = dict(origin=state['origin'], child_epoch=state['completed_child_epochs'],
                absolute_epoch=state['absolute_completed_epochs'], lifetime_updates=state['lifetime_updates'],
                checkpoint_sha256=hashlib.sha256(payloads[arm]).hexdigest(), model_state_sha256=state['model_state_sha256'],
                validation=metrics, operating_points=pilot.selected_operating_points(records[arm]),
                qualifies_development_replacement_gate=qualifies)
        del records
    arm_receipts = {}
    for arm in ARMS:
        require(len(histories[arm]) == 10, 'Continuation did not complete ten child epochs')
        report = dict(schema='fwt_compact_continuation_selected_v1', branch_id=arm, selection_frozen=True,
            independent_evaluation=False, selected_on='V_only', inherited_parent=baseline, **selected_reports[arm])
        atomic_json(args.output / arm / 'validation_selected.json', report)
        arm_receipts[arm] = dict(status='completed', branch_id=arm, learning_rate=LEARNING_RATES[arm],
            child_epochs=10, child_updates=510, lifetime_updates=3009, absolute_completed_epochs=59,
            last_validation=histories[arm][-1]['validation'],
            last_three_mean_ap=sum(r['validation']['pooled_ap'] for r in histories[arm][-3:]) / 3,
            overall_best=selected_reports[arm]['overall_best'], new_best=selected_reports[arm]['new_best'],
            binding_sha256=canonical_sha(bindings[arm]),
            files={name: sha(args.output / arm / name) for name in ('best.pth', 'last.pth', 'new_best.pth',
                'history.json', 'configuration.json', 'validation_selected.json')})
        atomic_json(args.output / arm / 'receipt.json', arm_receipts[arm])
    verify_unchanged()
    difference = arm_receipts['lower_lr']['last_three_mean_ap'] - arm_receipts['constant_lr']['last_three_mean_ap']
    atomic_json(args.output / 'receipt.json', dict(schema='fwt_compact_continuation_completed_v1', status='completed',
        arms=arm_receipts, child_epochs=10, completed_child_epochs=10, child_updates_per_arm=510,
        lifetime_updates=3009, parent_completed_epochs=49, absolute_completed_epochs=59,
        parent_checkpoint_sha256=PARENT_BEST_SHA, plan_sha256=sha(args.output / 'plan.json'),
        diagnostic_report_sha256=plan['diagnostic_report_sha256'], experiment_plan_sha256=plan['experiment_plan_sha256'],
        input_binding_sha256=sha(args.output / 'input-binding.json'), migration_sha256=sha(args.output / 'migration.json'),
        transaction_sha256=sha(args.output / 'transaction.pth'),
        comparison=dict(last_three_mean_ap_lower_minus_constant=difference, practical_margin=.001,
            interpretation='lower_lr_better' if difference >= .001 else 'constant_lr_better' if difference <= -.001 else 'similar'),
        selection=select_development_candidate(arm_receipts),
        completed_at=time.time(), limitations=plan['limitations']))
    write_status(args, 'completed', completed_child_epochs=10, child_epochs=10, lifetime_updates=3009,
                 absolute_completed_epochs=59, completed_arms=list(ARMS))


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k not in
            {'all_batch_orders', 'batch_orders', 'training_sample_ids', 'validation_sample_ids'}}, indent=2, allow_nan=False))
        return 0
    try:
        execute(args, plan)
    except Exception as exc:
        if getattr(args, '_output_created', False) and not (args.output / 'receipt.json').exists():
            from fwt_v2.artifacts import atomic_json
            atomic_json(args.output / 'failure.json', dict(status='failed', error_type=type(exc).__name__,
                message=str(exc), recorded_at=time.time()))
            write_status(args, 'failed', error_type=type(exc).__name__, message=str(exc))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
