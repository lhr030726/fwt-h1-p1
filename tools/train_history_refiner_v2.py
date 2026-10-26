#!/usr/bin/env python3
"""Three-arm event robustness experiment on frozen H1 + compact best49.

All arms receive real previous-input changes and identical initial parameters.
A retains pixel BCE, B mixes event-equal BCE, C also bounds the correction.
The historical v1 run and its weights remain immutable; these are fresh branches.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import continue_compact_refiner as previous

pilot = previous.pilot
require, sha, read_json, canonical_sha = previous.require, previous.sha, previous.read_json, previous.canonical_sha
tensor_cpu_tree = previous.tensor_cpu_tree
serialize_state, deserialize_state = previous.serialize_state, previous.deserialize_state
ARMS = ('pixel_bce', 'event_bce', 'event_bce_bounded')
SEEDS = (4, 17, 29)
EPOCHS = 50
EVENT_MIX = .5
CORRECTION_CAP = 2.
SCHEMA = 'fwt_history_event_training_state_v2'
TRANSACTION_SCHEMA = 'fwt_history_three_arm_transaction_v2'
PLAN_SCHEMA = 'fwt_history_event_plan_v2'
ADDITIONAL_PARENT_SOURCES = ('fwt_v2/history_refiner.py', 'fwt_v2/history_refiner_v2.py')
SCIENTIFIC_DEPENDENCIES = {'numpy': '1.22.3', 'scipy': '1.11.4',
                          'scikit-learn': '1.2.2', 'pyproj': '3.7.1'}


def source_files():
    return previous.source_files() | {name: sha(ROOT / name) for name in
        ('tools/train_history_refiner.py', 'fwt_v2/history_refiner.py',
         'tools/train_history_refiner_v2.py', 'fwt_v2/history_refiner_v2.py')}


def epoch_orders(seed, sample_ids, epochs=EPOCHS, batch_size=16):
    import torch
    require(len(sample_ids) == len(set(sample_ids)) and len(sample_ids) > 0, 'Sample IDs must be unique')
    require(type(epochs) is int and epochs > 0 and batch_size > 0, 'Invalid fixed training budget')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    orders = []
    for _ in range(epochs):
        ids = [sample_ids[i] for i in torch.randperm(len(sample_ids), generator=generator).tolist()]
        orders.append([ids[i:i + batch_size] for i in range(0, len(ids), batch_size)])
    return orders


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run', 'output', 'diagnostic-report', 'experiment-plan'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--preflight-run', type=Path,
                        help='Passed v2 preflight directory; required with --execute')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    require(args.epochs == EPOCHS and args.seeds == list(SEEDS), 'Budget is fixed at 50 epochs and seeds 4,17,29')
    require(not (args.preflight and args.resume_run), 'Preflight cannot resume training')
    require(not (args.preflight and args.preflight_run), 'Preflight mode cannot consume another preflight')
    require(not args.execute or args.preflight_run is not None, '--execute requires --preflight-run')
    return args


def make_plan(args):
    # Reuse the exhaustive original parent/diagnostic/artifact validator without
    # inheriting its continuation schedule, optimizer migration or output state.
    inherited_args = copy(args)
    inherited_args.resume_run, inherited_args.execute = None, False
    inherited = previous.make_plan(inherited_args, additional_source_files=ADDITIONAL_PARENT_SOURCES)
    plan = {key: deepcopy(inherited[key]) for key in (
        'parent_run', 'parent_commit', 'parent_plan_sha256', 'parent_receipt_sha256',
        'parent_best_sha256', 'parent_model_sha256', 'parent_binding_sha256',
        'parent_selected_validation', 'parent_source_files', 'input_files_sha256',
        'diagnostic_report', 'diagnostic_report_sha256', 'experiment_plan',
        'experiment_plan_sha256', 'input_paths', 'training_sample_ids', 'validation_sample_ids',
        'diffusion_config', 'data_config', 'model_config', 'manifest_sha256',
        'normalizer_sha256', 'bound_data_metadata_paths')}
    orders = {str(seed): epoch_orders(seed, plan['training_sample_ids']) for seed in SEEDS}
    recipe = dict(epochs=EPOCHS, batch_size=16, validation_batch_size=8, optimizer='AdamW',
        learning_rate=1e-4, weight_decay=.01, precision='float32', cpu_threads=2,
        sample_cache_gib=16, loss='A_pixel_BCE_B_C_mixed_event_BCE', event_mix=EVENT_MIX,
        correction_cap=CORRECTION_CAP, auxiliary_weight=0.,
        augmentation='none', scheduler='constant', gradient_clipping=False, ema=False,
        parent='frozen_H1_and_original_compact_best49_eval_no_grad',
        branch_initialization='same_seed_same_weights_three_arms_zero_output_fresh_AdamW',
        control='A_original_pixel_BCE_B_event_weighting_C_event_weighting_plus_bound',
        history='exact_previous_day_same_role_event_group_grid_input_only',
        checkpoint_selection='max_full_V_sigmoid_pooled_AP_earliest_tie_parent_fallback',
        metrics='epoch_sigmoid_pooled_AP_event_NLL_Brier_last5_and_selected_full_metrics_and_history_subsets',
        replacement_gate=dict(ap_noninferiority_tolerance=.001, proper_score_max_ratio=1.,
            positive_event_ap_max_drop=0., precision70_max_drop=0.),
        hypothesis_comparison='paired_last_five_epoch_B_minus_A_C_minus_B_and_C_minus_A',
        event_support='Tr_full_valid_pixel_totals_uniform_sample_batches_no_resampling',
        practical_ap_margin=.001)
    plan.update(schema=PLAN_SCHEMA, stage='history_event_robustness', seeds=list(SEEDS), arms=list(ARMS),
        recipe=recipe, source_files=source_files(), batch_orders=orders,
        batch_order_sha256={seed: canonical_sha(value) for seed, value in orders.items()},
        output=str(args.output), resume_run=str(args.resume_run) if args.resume_run else None,
        mode='preflight' if args.preflight else 'execute' if args.execute else 'dry_run',
        additional_parent_source_allowlist=list(ADDITIONAL_PARENT_SOURCES),
        migration_policy='fresh_only_same_array_paths_sizes_SHA_same_Python_3_10_Torch_numeric_settings_allow_stat_GPU_build_changes',
        limitations=['Three branch-initialization seeds on reused development V; not independent evaluation.',
            'Selected parent is fixed seed4 best49; parent-seed variability is not tested.',
            'Fresh residual branch training; original parent optimizer and progress remain unchanged.',
            'Thresholds are selected on V; no calibrated deployment or blind-test claim.'])
    plan['preflight_run'] = str(args.preflight_run) if args.preflight_run else None
    plan['preflight_artifacts'] = validate_preflight(args.preflight_run, plan) if args.preflight_run else None
    if args.resume_run:
        old = read_json(args.resume_run / 'plan.json')
        ignored = {'output', 'resume_run', 'mode'}
        require(canonical_sha({k: v for k, v in old.items() if k not in ignored}) == canonical_sha(
            {k: v for k, v in plan.items() if k not in ignored}), 'Resume source/data/recipe/budget differs')
    return plan


def validate_preflight(directory, plan):
    """A successful, source-bound v2 mechanism check must precede execution."""
    directory = Path(directory)
    report = read_json(directory / 'preflight.json')
    preflight_plan = read_json(directory / 'plan.json')
    binding = read_json(directory / 'input-binding.json')
    require(report.get('schema') == 'fwt_history_event_preflight_v2' and report.get('status') == 'passed'
            and report.get('zero_initialization_exact') is True and report.get('frozen_parent_unchanged') is True
            and report.get('training_checkpoints_created') is False
            and report.get('disposable_optimizer_steps_per_arm') == 1
            and set(report.get('arms', {})) == set(ARMS), 'Require successful three-arm v2 preflight')
    require(preflight_plan.get('schema') == PLAN_SCHEMA and preflight_plan.get('mode') == 'preflight',
            'Preflight plan is not a v2 preflight')
    require(report.get('plan_sha256') == sha(directory / 'plan.json')
            and report.get('input_binding_sha256') == sha(directory / 'input-binding.json'),
            'Preflight plan/input binding digest differs')
    excluded = {'output', 'resume_run', 'mode', 'preflight_run', 'preflight_artifacts'}
    require(canonical_sha({k: v for k, v in preflight_plan.items() if k not in excluded}) ==
            canonical_sha({k: v for k, v in plan.items() if k not in excluded}),
            'Preflight source/data/recipe/parent/budget differs')
    require(binding.get('source_files') == plan['source_files'] and binding.get('recipe') == plan['recipe'],
            'Preflight binding source/recipe differs')
    require(abs(report['parent_replay']['pooled_ap'] - plan['parent_selected_validation']['pooled_ap']) <= 1e-6,
            'Preflight parent replay differs')
    return {name: sha(directory / name) for name in ('preflight.json', 'plan.json', 'input-binding.json')}


class HistoryDataset:
    """Add audited input-only history to unchanged cached dataset samples."""
    def __init__(self, dataset, history):
        self.dataset, self.history = dataset, history
        require(len(dataset) == len(history['history_delta']) == len(history['history_valid']), 'History length differs')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        item = dict(self.dataset[index])
        item['history_delta'] = torch.from_numpy(self.history['history_delta'][index].copy())
        item['history_valid'] = torch.from_numpy(self.history['history_valid'][index].copy())
        return item


def initialize_pair(seed, device):
    import torch
    from fwt_v2.engine import seed_everything, model_state_digest
    from fwt_v2.history_refiner import HistoryRefiner
    from fwt_v2.history_refiner_v2 import BoundedHistoryRefiner
    seed_everything(seed)
    template = HistoryRefiner().to(device)
    heads = {arm: deepcopy(template) for arm in ARMS}
    heads['event_bce_bounded'] = BoundedHistoryRefiner(cap=CORRECTION_CAP).to(device)
    heads['event_bce_bounded'].load_state_dict(template.state_dict(), strict=True)
    optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=1e-4, weight_decay=.01) for arm in ARMS}
    require(len({model_state_digest(head) for head in heads.values()}) == 1, 'Paired initialization differs')
    require(all(not optimizer.state for optimizer in optimizers.values()), 'New branch must have fresh AdamW')
    return heads, optimizers


def corrected_logits(head, arm, outputs, parent_logits, batch):
    require(arm in ARMS, 'Unknown history arm')
    return head(outputs, parent_logits, batch['history_delta'], batch['history_valid'])


def arm_loss(arm, logits, batch, support):
    """One loss dispatch for preflight and formal training; A keeps frozen BCE."""
    require(arm in ARMS, 'Unknown loss arm')
    if arm == 'pixel_bce':
        from fwt_v2.literature_refiner import masked_refinement_loss
        pixel = masked_refinement_loss(logits, batch['target'], batch['valid_mask'], aux_weight=0.)['bce']
        return dict(total=pixel, pixel_bce=pixel)
    from fwt_v2.history_refiner_v2 import mixed_event_bce
    return mixed_event_bce(logits, batch['target'], batch['valid_mask'], batch['event_id'],
                          support, event_mix=EVENT_MIX)


def validate_fresh_parent_binding(old, array_hashes, sample_identity, h1_sha, runtime):
    """Permit documented fresh-run relocation, never data or numeric drift.

    Full array content hashes and sample/field/path/size remain exact. Only
    filesystem mtime/device/inode and GPU/Python 3.10 build may differ. This is
    not an adapter for resuming v1 or a v2 transaction on another machine.
    """
    for key, value in (('array_files_sha256', array_hashes), ('frozen_backbone_state_sha256', h1_sha)):
        require(canonical_sha(value) == canonical_sha(old[key]), 'Parent content differs: ' + key)
    retained = ('sample_id', 'field', 'path', 'size')
    stat_keys = set(retained) | {'mtime_ns', 'device', 'inode'}
    old_stats = old['array_stat_identity']
    require(all(set(row) == stat_keys for row in old_stats + sample_identity), 'Unexpected array stat fields')
    require([{key: row[key] for key in retained} for row in old_stats] ==
            [{key: row[key] for key in retained} for row in sample_identity],
            'Parent array sample/path/size differs')
    old_runtime = old['runtime']
    require(set(runtime) == set(old_runtime), 'Parent runtime field set differs')
    for key in runtime:
        if key not in ('python', 'gpu'):
            require(runtime[key] == old_runtime[key], 'Parent numerical runtime differs: ' + key)
    for label, value in (('parent', old_runtime['python']), ('current', runtime['python'])):
        require(isinstance(value, str) and re.match(r'^3\.10\.\d+(?:\s|$)', value) is not None,
                label + ' runtime must retain Python 3.10')
    stat_changes = [{key: current[key] for key in retained} | {
        'changed_fields': {key: dict(parent=prior[key], current=current[key])
                           for key in ('mtime_ns', 'device', 'inode') if prior[key] != current[key]}}
        for prior, current in zip(old_stats, sample_identity) if prior != current]
    runtime_changes = {key: dict(parent=old_runtime[key], current=runtime[key])
                       for key in ('python', 'gpu') if old_runtime[key] != runtime[key]}
    return dict(schema='fwt_history_fresh_runtime_difference_v1', fresh_branch_training=True,
        array_sha256_identical=True, array_sample_path_size_identical=True,
        frozen_backbone_identical=True, numerical_runtime_settings_identical=True,
        runtime_changes=runtime_changes, array_stat_changes=stat_changes,
        array_stat_changed_entries=len(stat_changes), cross_machine_resume_supported=False)


def validate_scientific_environment():
    """Keep preprocessing, metrics and spatial libraries on the archived build."""
    from importlib.metadata import version
    observed = {package: version(package) for package in SCIENTIFIC_DEPENDENCIES}
    require(observed == SCIENTIFIC_DEPENDENCIES, 'Scientific dependency versions differ from frozen experiment')
    return observed


def history_to_device(batch, device):
    """Extend the frozen engine's explicit device transfer for new inputs.

    Keep reporting metadata (including its CPU domain mask) under the original
    engine contract, while moving every tensor consumed by the history branch.
    """
    from fwt_v2.engine import to_device
    moved = to_device(batch, device)
    moved['history_delta'] = batch['history_delta'].to(device)
    moved['history_valid'] = batch['history_valid'].to(device)
    return moved


def make_state(head, optimizer, binding, history, baseline, device):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import capture_rng, optimizer_state_integrity
    n = len(history)
    state = dict(schema=SCHEMA, checkpoint_kind='new_branch_full_training_state', binding=deepcopy(binding),
        completed_epochs=n, updates=n * binding['updates_per_epoch'], frozen_parent_completed_epochs=49,
        frozen_parent_updates=2499, model_state_dict=tensor_cpu_tree(head.state_dict()),
        model_state_sha256=model_state_digest(head), optimizer_state_dict=tensor_cpu_tree(optimizer.state_dict()),
        rng_state=capture_rng(device), history=deepcopy(history), baseline_validation=deepcopy(baseline),
        validation=deepcopy(history[-1]['validation'] if n else baseline),
        origin='trained_new_history_branch' if n else 'zero_initialized_parent_equivalent')
    state['optimizer_integrity'] = optimizer_state_integrity(state['optimizer_state_dict'])
    check_state(state)
    return state


def check_state(state, binding=None):
    from fwt_v2.training_state import verify_optimizer_integrity
    current = state.get('binding', {})
    n = state.get('completed_epochs')
    require(state.get('schema') == SCHEMA and current.get('arm') in ARMS
        and current.get('seed') in SEEDS, 'History state schema/arm/seed differs')
    require(binding is None or current == binding, 'History state binding differs')
    require(type(n) is int and 0 <= n <= current['epochs'] and len(state['history']) == n
        and state['updates'] == n * current['updates_per_epoch'], 'History state progress differs')
    require([r['epoch'] for r in state['history']] == list(range(1, n + 1))
        and all(r['updates'] == r['epoch'] * current['updates_per_epoch'] for r in state['history']), 'History sequence differs')
    require(state['optimizer_state_dict']['param_groups'] == current['optimizer_param_groups']
        and all(g['lr'] == 1e-4 and g['weight_decay'] == .01 for g in current['optimizer_param_groups']), 'AdamW recipe differs')
    verify_optimizer_integrity(state)
    moments = state['optimizer_state_dict']['state']
    require((not moments if n == 0 else bool(moments))
        and all(float(item['step']) == state['updates'] for item in moments.values()),
        'Fresh branch AdamW steps differ from completed branch updates')
    require(state['validation'] == (state['history'][-1]['validation'] if n else state['baseline_validation'])
        and math.isfinite(state['validation']['pooled_ap']), 'State selected metrics differ')
    require(state['origin'] == ('trained_new_history_branch' if n else 'zero_initialized_parent_equivalent'), 'State origin differs')


def restore_state(state, head, optimizer, binding, device):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import restore_training_parameters, validate_rng, restore_rng
    check_state(state, binding)
    if hasattr(head, 'configuration'):
        require(head.configuration() == binding.get('branch_configuration'),
                'Restore branch architecture or correction bound differs')
    validate_rng(state['rng_state'], device)
    owned = deepcopy(state)  # AdamW load may otherwise alias input moment/step tensors.
    restore_training_parameters(dict(owned, selected={'state_dict': owned['model_state_dict']}), head, optimizer)
    require(model_state_digest(head) == state['model_state_sha256'], 'Restored history model differs')
    restore_rng(state['rng_state'], device)


def build_transaction(states, best, new_best, rng):
    import torch
    require(set(states) == set(best) == set(new_best) == set(ARMS), 'All three history arms are required')
    counts, seeds = {v['completed_epochs'] for v in states.values()}, {v['binding']['seed'] for v in states.values()}
    require(len(counts) == len(seeds) == 1, 'Paired transaction seed/progress differs')
    entries = {}
    for arm in ARMS:
        entries[arm] = {}
        for kind, data in (('last', serialize_state(states[arm])), ('best', best[arm]), ('new_best', new_best[arm])):
            if data is not None:
                entries[arm][kind + '_data'] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                entries[arm][kind + '_sha256'] = hashlib.sha256(data).hexdigest()
    transaction = dict(schema=TRANSACTION_SCHEMA, seed=seeds.pop(), completed_epochs=counts.pop(), arms=entries, rng_state=rng)
    decode_transaction(transaction)
    return transaction


def decode_transaction(transaction, bindings=None):
    require(transaction.get('schema') == TRANSACTION_SCHEMA and transaction.get('seed') in SEEDS
        and set(transaction.get('arms', {})) == set(ARMS), 'History transaction schema differs')
    n = transaction.get('completed_epochs')
    require(type(n) is int and 0 <= n <= EPOCHS, 'Invalid transaction epoch')
    decoded = {}
    for arm, entry in transaction['arms'].items():
        kinds = ('last', 'best', 'new_best') if n else ('last', 'best')
        require(set(entry) == {kind + suffix for kind in kinds for suffix in ('_data', '_sha256')}, 'Transaction payloads differ')
        states = {kind: deserialize_state(pilot.transaction_checkpoint_bytes(entry, kind)) for kind in kinds}
        last = states['last']
        for state in states.values():
            check_state(state, bindings[arm] if bindings else last['binding'])
            count = state['completed_epochs']
            require(state['binding']['arm'] == arm and state['binding']['seed'] == transaction['seed']
                and count <= n and state['history'] == last['history'][:count]
                and state['baseline_validation'] == last['baseline_validation'], 'Paired checkpoint history differs')
        aps = [last['baseline_validation']['pooled_ap']] + [r['validation']['pooled_ap'] for r in last['history']]
        require(last['completed_epochs'] == n
            and states['best']['completed_epochs'] == max(range(n + 1), key=lambda i: aps[i]), 'Overall best tie/selection differs')
        if n:
            require(states['new_best']['completed_epochs'] == 1 + max(range(n), key=lambda i: aps[i + 1]), 'New-only best differs')
        decoded[arm] = states
    return decoded


def publish_transaction(directory, transaction):
    from fwt_v2.artifacts import atomic_checkpoint
    decode_transaction(transaction)
    atomic_checkpoint(directory / 'transaction.pth', transaction)
    for arm, entry in transaction['arms'].items():
        (directory / arm).mkdir(parents=True, exist_ok=True)
        for kind in ('last', 'best', 'new_best'):
            if kind + '_data' in entry:
                pilot.atomic_bytes(directory / arm / (kind + '.pth'), pilot.transaction_checkpoint_bytes(entry, kind))


def publish_progress(directory, transaction, histories):
    """Materialize history mirrors even when resuming an already-finished seed."""
    from fwt_v2.artifacts import atomic_json
    publish_transaction(directory, transaction)
    for arm in ARMS:
        require(len(histories[arm]) == transaction['completed_epochs'], 'History mirror progress differs')
        atomic_json(directory / arm / 'history.json', histories[arm])


def write_status(args, phase, **details):
    from fwt_v2.artifacts import atomic_json
    value = dict(schema='fwt_history_event_training_status_v2', phase=phase, epochs_per_seed=EPOCHS,
        seeds=list(SEEDS), arms=list(ARMS), started_at=args._started_at, updated_at=time.time(),
        resources=pilot.old.resource_snapshot(), **details)
    atomic_json(args.output / 'status.json', value)
    with (args.output / 'resource-history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def validation_records(backbone, parent, heads, loader, device, include_parent=False,
                       require_parent_equal=False, collect_heads=True):
    import torch
    from fwt_v2.engine import records_from_batch
    backbone.eval(); parent.eval()
    for head in heads.values():
        head.eval()
    result = {arm: [] for arm in heads} if collect_heads else {}
    if include_parent:
        result['parent'] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = history_to_device(cpu_batch, device)
            outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
            anchor = parent(outputs, batch['physics'])
            if include_parent:
                result['parent'].extend(records_from_batch(batch, anchor.sigmoid(), anchor))
            for arm, head in heads.items():
                logits = corrected_logits(head, arm, outputs, anchor, batch)
                require(torch.equal(logits[~batch['history_valid']], anchor[~batch['history_valid']]),
                    'History-invalid pixels must remain exactly parent')
                if require_parent_equal:
                    require(torch.equal(logits, anchor), 'Full V zero-initialized branch differs from parent')
                if collect_heads:
                    result[arm].extend(records_from_batch(batch, logits.sigmoid(), logits))
    return result


def check_records(records, plan, identity):
    require([r['sample_id'] for r in records] == plan['validation_sample_ids']
        and pilot.records_identity(records) == identity, 'Original V sample/target/masks differ')


def detailed_metrics(records, history_sample_ids):
    """Expensive sorts only at baseline, fixed last five epochs and selection.

    Subsets are fixed from metadata pairing before outcomes are observed. An
    available pair with no common valid pixels still belongs to the available
    metadata subset; actual usable-pixel coverage is separately audited.
    """
    import numpy as np

    def measure(rows):
        n_valid = sum(int(np.asarray(row['valid_mask'], bool).sum()) for row in rows)
        n_positive = sum(int(((np.asarray(row['target']) > .5) & np.asarray(row['valid_mask'], bool)).sum()) for row in rows)
        if not rows or not n_valid or not n_positive:
            return dict(status='undefined_no_valid_positive', n_samples=len(rows),
                n_valid_pixel_days=n_valid, n_positive_pixel_days=n_positive)
        metrics = previous.extended_metrics(rows)
        metrics.update(pilot.selected_positive_event_ap(rows))
        metrics['operating_points'] = pilot.selected_operating_points(rows)
        metrics['n_samples'] = len(rows)
        return metrics

    result = measure(records)
    result['history_subsets'] = {
        name: measure([row for row in records if (row['sample_id'] in history_sample_ids) == flag])
        for name, flag in (('eligible_previous_input', True), ('no_eligible_previous_input', False))}
    result['history_subset_definition'] = 'fixed_exact_previous_day_metadata_pair_not_outcome_or_pixel_validity'
    return result


def development_comparison(seed_receipts, baseline):
    require(set(seed_receipts) == {str(seed) for seed in SEEDS}, 'All three paired seeds are required for comparison')
    require(all(set(arms) == set(ARMS) for arms in seed_receipts.values()), 'All three arms required for comparison')
    keys = ('pooled_ap', 'nll', 'brier', 'positive_event_ap')
    means = {arm: {key: sum(rows[arm]['last_five_means'][key] for rows in seed_receipts.values()) / len(SEEDS)
                  for key in keys} for arm in ARMS}
    for arm in ARMS:
        means[arm]['precision70'] = sum(rows[arm]['last_five_mean_precision_at_recall70']
                                       for rows in seed_receipts.values()) / len(SEEDS)
    contrasts = {}
    for candidate, reference in (('event_bce', 'pixel_bce'), ('event_bce_bounded', 'event_bce'),
                                 ('event_bce_bounded', 'pixel_bce')):
        left, right = means[candidate], means[reference]
        checks = dict(pooled_ap_noninferior=left['pooled_ap'] >= right['pooled_ap'] - .001,
            event_ap_nondecreasing=left['positive_event_ap'] >= right['positive_event_ap'],
            event_nll_nonincreasing=left['nll'] <= right['nll'],
            event_brier_nonincreasing=left['brier'] <= right['brier'],
            precision70_nondecreasing=left['precision70'] >= right['precision70'])
        event_differences = {}
        for seed, rows in seed_receipts.items():
            candidate_events = rows[candidate]['last_five_event_ap']
            reference_events = rows[reference]['last_five_event_ap']
            require(set(candidate_events) == set(reference_events), 'Per-event comparison domains differ')
            event_differences[seed] = {event: candidate_events[event] - reference_events[event]
                                       for event in candidate_events}
        events = set(next(iter(event_differences.values())))
        require(all(set(values) == events for values in event_differences.values()), 'Seed event sets differ')
        event_means = {event: sum(values[event] for values in event_differences.values()) / len(SEEDS)
                       for event in sorted(events)}
        contrasts[candidate + '_minus_' + reference] = dict(
            mean_differences={key: left[key] - right[key] for key in (*keys, 'precision70')},
            paired_seed_ap_differences={seed: rows[candidate]['last_five_means']['pooled_ap'] -
                rows[reference]['last_five_means']['pooled_ap'] for seed, rows in seed_receipts.items()},
            event_ap_difference_by_event=event_means,
            event_ap_improved=sum(value > 0 for value in event_means.values()),
            event_ap_degraded=sum(value < 0 for value in event_means.values()),
            event_ap_tied=sum(value == 0 for value in event_means.values()),
            development_gate_checks=checks, all_development_gates_passed=all(checks.values()))
    return dict(last_five_seed_means=means, contrasts=contrasts, ap_noninferiority_tolerance=.001,
        frozen_parent_reference={key: baseline[key] for key in keys},
        interpretation='Noninferiority gates are screening safeguards, not statistical significance or proof of improvement.',
        event_degradation_counts_are_diagnostic_not_acceptance_gates=True,
        independent_evaluation=False, automatic_parent_replacement=False)


def verify_zero_pair(backbone, parent, heads, cpu_batch, device):
    import torch
    batch = history_to_device(cpu_batch, device)
    with torch.no_grad():
        outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
        anchor = parent(outputs, batch['physics'])
        for arm, head in heads.items():
            logits = corrected_logits(head, arm, outputs, anchor, batch)
            require(torch.equal(logits, anchor) and bool(torch.isfinite(logits).all()), 'Zero branch is not exactly parent')


def data_array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_sha([str(value.dtype), list(value.shape)]).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def run(args, plan):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.artifacts import atomic_json, output_path
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import audited_manifest, load_det, model_state_digest, seed_everything
    from fwt_v2.gradient_diagnostics import gradients_finite, gradient_l2_norm
    from fwt_v2.history_refiner import load_history_arrays
    from fwt_v2.history_refiner_v2 import build_event_support, correction_diagnostics
    from fwt_v2.manifest import manifest_digest, samples_for_role
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    from fwt_v2.training_state import capture_rng, restore_rng, sample_file_identity

    output_path(args.output / 'plan.json')
    require(not args.output.exists() and args.parent_run not in args.output.parents, 'Use a new output outside original run')
    scientific_dependencies = validate_scientific_environment()
    pilot.old.assert_gpu_idle()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one CUDA GPU required')
    previous.check_disk(args.output)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    device = torch.device('cuda:0')
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_created, args._started_at = True, time.time()
    atomic_json(args.output / 'plan.json', plan)
    write_status(args, 'verify_data', step='hash_original_arrays_and_build_input_only_history')
    paths = {k: Path(v) for k, v in plan['input_paths'].items()}
    os.chdir(paths['manifest'].parent)
    manifest = audited_manifest(paths['manifest'])
    require(manifest_digest(manifest) == plan['manifest_sha256']
        and sha(paths['normalizer']) == plan['normalizer_sha256'], 'Original data metadata differs')
    train_rows, val_rows = (samples_for_role(manifest, role) for role in ('residual_train', 'validation'))
    require([r['sample_id'] for r in train_rows] == plan['training_sample_ids']
        and [r['sample_id'] for r in val_rows] == plan['validation_sample_ids'], 'Original Tr/V identities differ')
    samples = train_rows + val_rows
    sample_identity = sample_file_identity(samples)
    array_hashes = {str(Path(row[k]).resolve()): sha(row[k]) for row in samples for k in ('input_path', 'label_path')}
    pilot.old.verify_selected_array_index(read_json(plan['bound_data_metadata_paths']['arrays_index']),
        samples, array_hashes, paths['manifest'].parent)
    base_train = RCDADataset(manifest, 'residual_train', NormalizationStats.load(paths['normalizer']), DataConfig(**plan['data_config']))
    base_val = copy(base_train)
    base_val.role, base_val.samples = 'validation', val_rows
    histories_data = {'residual_train': load_history_arrays(base_train), 'validation': load_history_arrays(base_val)}
    history_val_ids = {row['sample_id'] for row, available in zip(val_rows, histories_data['validation']['has_history']) if available}
    history_audit = {}
    for role, history in histories_data.items():
        audit = history['audit']
        require(audit['pairing']['paired_samples'] == {'residual_train': 689, 'validation': 484}[role],
            'Audited exact previous-day pairing count differs: ' + role)
        require(audit['label_arrays_read'] == 0 and audit['totals']['common_input_valid_pixels'] > 0
            and audit['totals']['positive_change_pixels'] + audit['totals']['negative_change_pixels'] > 0,
            'History mechanism has no usable input validity/change: ' + role)
        history_audit[role] = dict(audit=history['audit'], previous_sample_ids=history['previous_sample_ids'],
            array_sha256={name: data_array_digest(history[name]) for name in ('history_delta', 'history_valid', 'has_history')})
    atomic_json(args.output / 'history-audit.json', history_audit)
    cache = SharedTensorCache(16 * 1024**3)
    training = HistoryDataset(CachedDataset(base_train, cache), histories_data['residual_train'])
    validation = HistoryDataset(CachedDataset(base_val, cache), histories_data['validation'])
    # Tr only, all original samples including unavailable history and negatives.
    support = build_event_support(training)
    require(support['n_samples'] == len(training), 'Training support omitted samples')
    atomic_json(args.output / 'training-event-support.json', support)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0))
    seed_everything(0)
    backbone, bundle = load_det(paths['backbone'], device)
    require(bundle.get('selection_frozen') and bundle['family'] == 'det'
        and all(canonical_sha(bundle[k]) == canonical_sha(plan[k]) for k in
            ('data_config', 'model_config', 'manifest_sha256', 'normalizer_sha256')), 'Frozen H1 metadata differs')
    backbone.requires_grad_(False).eval()
    h1_sha = model_state_digest(backbone)
    old_common = read_json(args.parent_run / 'input-binding.json')
    runtime = dict(python=sys.version, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        cpu_threads=2, interop_threads=2, matmul_tf32=False, cudnn_tf32=True)
    migration = validate_fresh_parent_binding(old_common, array_hashes, sample_identity, h1_sha, runtime)
    atomic_json(args.output / 'fresh-runtime-differences.json', migration)
    parent_state = torch.load(args.parent_run / 'seed-004/compact_bce/best.pth', map_location='cpu', weights_only=True)
    require(parent_state['completed_epochs'] == 49 and parent_state['updates'] == 2499
        and parent_state['model_state_sha256'] == plan['parent_model_sha256'], 'Frozen parent checkpoint differs')
    config = DiffusionConfig(**dict(plan['diffusion_config'], output_initialization='zero', prediction_type='clean_probability'))
    parent, _, _ = pilot.initialize_head(4, 'compact_bce', paths['backbone'], device, config, h1_sha)
    parent.load_state_dict(parent_state['model_state_dict'], strict=True)
    require(model_state_digest(parent) == plan['parent_model_sha256'], 'Frozen parent model digest differs')
    parent.requires_grad_(False).eval()
    del parent_state
    bound_stats = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in plan['input_files_sha256']}

    def verify_unchanged(full=False):
        require(source_files() == plan['source_files'], 'Source changed during experiment')
        require(all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == value for path, value in bound_stats.items()),
            'Bound historical artifact file identity changed')
        if full:
            require(all(sha(path) == digest for path, digest in plan['input_files_sha256'].items()), 'Historical artifact SHA changed')
        require(sample_file_identity(samples) == sample_identity, 'Original Tr/V array identity changed')
        for model, digest in ((backbone, h1_sha), (parent, plan['parent_model_sha256'])):
            require(model_state_digest(model) == digest and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                'Frozen parent acquired gradients or changed')

    write_status(args, 'parent_replay', step='full_original_V_568')
    probe_heads, probe_optimizers = initialize_pair(4, device)
    records = validation_records(backbone, parent, probe_heads, val_loader, device, include_parent=True,
        require_parent_equal=True, collect_heads=False)['parent']
    domain_sha = pilot.records_identity(records)
    require(domain_sha == old_common['target_and_masks_sha256'], 'Original V masks/targets differ')
    check_records(records, plan, domain_sha)
    baseline = detailed_metrics(records, history_val_ids)
    previous.check_metric_replay(baseline, plan['parent_selected_validation'])
    require(abs(baseline['pooled_ap'] - previous.PARENT_AP) <= 1e-6, 'Original best49 AP replay failed')
    baseline_operating = baseline['operating_points']
    del records
    common = dict(source_files=plan['source_files'], recipe=plan['recipe'], runtime=runtime,
        scientific_dependencies=scientific_dependencies,
        input_files_sha256=plan['input_files_sha256'], array_files_sha256=array_hashes,
        array_stat_identity=sample_identity, target_and_masks_sha256=domain_sha,
        history_audit_sha256=sha(args.output / 'history-audit.json'),
        event_support_sha256=sha(args.output / 'training-event-support.json'),
        event_support_content_sha256=canonical_sha(support),
        history_content=history_audit, frozen_backbone_state_sha256=h1_sha,
        frozen_parent_model_sha256=plan['parent_model_sha256'], frozen_parent_checkpoint_sha256=plan['parent_best_sha256'])
    if args.execute:
        require(args.preflight_run is not None, 'Formal training requires passed v2 preflight')
        require(validate_preflight(args.preflight_run, plan) == plan['preflight_artifacts'],
                'Preflight artifacts changed after plan validation')
        require(read_json(args.preflight_run / 'input-binding.json') == common,
                'Current source/data/support/runtime binding differs from passed preflight')
    if args.resume_run:
        require(read_json(args.resume_run / 'input-binding.json') == common,
                'V2 resume requires identical data/stat/runtime/support/source binding; cross-machine resume is unsupported')
    atomic_json(args.output / 'input-binding.json', common)
    atomic_json(args.output / 'baseline.json', dict(validation=baseline, operating_points=baseline_operating,
        parent_checkpoint_sha256=plan['parent_best_sha256'], independent_evaluation=False))
    # Exercise available history on the supervised domain in the disposable
    # probe; all-missing batches remain legal in the complete training order.
    probe_indices = []
    for i in range(len(training)):
        history = histories_data['residual_train']
        if history['has_history'][i] and (history['history_delta'][i] != 0).any():
            sample = training[i]
            if bool((sample['history_valid'] & sample['valid_mask'].bool()).any()):
                probe_indices.append(i)
        if len(probe_indices) == 16:
            break
    require(len(probe_indices) == 16, 'Cannot form a sixteen-sample usable-history mechanism probe')
    first_loader = DataLoader(training, batch_sampler=[probe_indices], num_workers=0,
        generator=torch.Generator().manual_seed(0))
    first_batch = next(iter(first_loader))
    if args.preflight:
        heads, optimizers = probe_heads, probe_optimizers
        verify_zero_pair(backbone, parent, heads, first_batch, device)
        initial = {arm: model_state_digest(head) for arm, head in heads.items()}
        batch = history_to_device(first_batch, device)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
            anchor = parent(outputs, batch['physics'])
        observations = {}
        for arm in ARMS:
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            logits = corrected_logits(heads[arm], arm, outputs, anchor, batch)
            loss = arm_loss(arm, logits, batch, support)['total']
            require(bool(torch.isfinite(loss)), 'Nonfinite preflight loss')
            loss.backward()
            gradients = [p.grad for p in heads[arm].parameters() if p.grad is not None]
            require(gradients and gradients_finite(gradients), 'Nonfinite/missing preflight gradients')
            norm = gradient_l2_norm(gradients)
            require(math.isfinite(norm) and norm > 0, 'Preflight failed to demonstrate a learnable history path')
            optimizer.step()
            with torch.no_grad():
                after = corrected_logits(heads[arm], arm, outputs, anchor, batch)
            require(bool((after != anchor).any()) and torch.equal(after[~batch['history_valid']], anchor[~batch['history_valid']]),
                'Disposable update must change predictions while preserving unavailable pixels')
            observations[arm] = dict(loss=float(loss.detach()), gradient_l2=norm,
                initial_model_sha256=initial[arm], disposable_updated_model_sha256=model_state_digest(heads[arm]),
                parameters=sum(p.numel() for p in heads[arm].parameters()), configuration=heads[arm].configuration())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        verify_unchanged(full=True)
        atomic_json(args.output / 'preflight.json', dict(schema='fwt_history_event_preflight_v2',
            status='passed', parent_replay=baseline, plan_sha256=sha(args.output / 'plan.json'),
            input_binding_sha256=sha(args.output / 'input-binding.json'),
            zero_initialization_exact=True, frozen_parent_unchanged=True, disposable_optimizer_steps_per_arm=1,
            training_checkpoints_created=False, timing_scope='one_disposable_shared_batch_not_steady_state',
            batch_samples=len(first_batch['sample_id']), shared_batch_seconds=elapsed,
            probe_sample_ids=list(first_batch['sample_id']),
            valid_history_supervision_pixels=int((batch['history_valid'] & batch['valid_mask'].bool()).sum()),
            shared_samples_per_second=len(first_batch['sample_id']) / elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            arms=observations))
        write_status(args, 'preflight_passed', parent_validation_ap=baseline['pooled_ap'])
        return

    del probe_heads, probe_optimizers
    seed_receipts = {}
    id_index = {row['sample_id']: i for i, row in enumerate(train_rows)}
    for seed in SEEDS:
        directory = args.output / f'seed-{seed:03d}'
        directory.mkdir()
        heads, optimizers = initialize_pair(seed, device)
        verify_zero_pair(backbone, parent, heads, first_batch, device)
        bindings, histories, states, best, new_best = {}, {}, {}, {}, {}
        for arm in ARMS:
            (directory / arm).mkdir()
            binding = dict(common, seed=seed, arm=arm, epochs=EPOCHS, updates_per_epoch=51,
                batch_order_sha256=plan['batch_order_sha256'][str(seed)],
                branch_configuration=heads[arm].configuration(), branch_parameters=sum(p.numel() for p in heads[arm].parameters()),
                initial_branch_model_sha256=model_state_digest(heads[arm]),
                optimizer_param_groups=deepcopy(optimizers[arm].state_dict()['param_groups']),
                optimizer_origin='fresh_new_branch_only_no_parent_state')
            bindings[arm], histories[arm] = binding, []
            atomic_json(directory / arm / 'configuration.json', binding)
            state = make_state(heads[arm], optimizers[arm], binding, [], baseline, device)
            states[arm], best[arm], new_best[arm] = state, serialize_state(state), None
        transaction = build_transaction(states, best, new_best, capture_rng(device))
        resume_path = args.resume_run / f'seed-{seed:03d}/transaction.pth' if args.resume_run else None
        if resume_path and resume_path.exists():
            transaction = torch.load(resume_path, map_location='cpu', weights_only=True)
            decoded = decode_transaction(transaction, bindings)
            for arm in ARMS:
                restore_state(decoded[arm]['last'], heads[arm], optimizers[arm], bindings[arm], device)
                histories[arm] = deepcopy(decoded[arm]['last']['history'])
                for row in histories[arm]:
                    if 'diagnostics_sha256' in row:
                        name = f"diagnostics-epoch{row['epoch']:03d}.json"
                        data = (args.resume_run / f'seed-{seed:03d}' / arm / name).read_bytes()
                        require(hashlib.sha256(data).hexdigest() == row['diagnostics_sha256'],
                                'Resume diagnostic file differs from checkpoint-bound digest')
                        pilot.atomic_bytes(directory / arm / name, data)
                best[arm] = pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
                new_best[arm] = pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'new_best') if transaction['completed_epochs'] else None
            restore_rng(transaction['rng_state'], device)
        publish_progress(directory, transaction, histories)
        for epoch in range(transaction['completed_epochs'], EPOCHS):
            verify_unchanged()
            previous.check_disk(directory, transaction)
            for head in heads.values():
                head.train()
            order = plan['batch_orders'][str(seed)][epoch]
            batches = [[id_index[sid] for sid in batch] for batch in order]
            loader = DataLoader(training, batch_sampler=batches, num_workers=0, generator=torch.Generator().manual_seed(0))
            write_status(args, 'training', seed=seed, epoch=epoch + 1, completed_epochs=epoch, batch=0, batches=51)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses, norms, observed = {a: [] for a in ARMS}, {a: [] for a in ARMS}, []
            for batch_index, cpu_batch in enumerate(loader):
                require(list(cpu_batch['sample_id']) == order[batch_index], 'Actual paired order differs')
                batch = history_to_device(cpu_batch, device)
                with torch.no_grad():
                    outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
                    anchor = parent(outputs, batch['physics'])
                for arm in ARMS:
                    optimizer, head = optimizers[arm], heads[arm]
                    optimizer.zero_grad(set_to_none=True)
                    logits = corrected_logits(head, arm, outputs, anchor, batch)
                    loss = arm_loss(arm, logits, batch, support)['total']
                    require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Nonfinite/nonscalar training BCE')
                    loss.backward()
                    gradients = [p.grad for p in head.parameters() if p.grad is not None]
                    require(gradients and gradients_finite(gradients), 'Missing/nonfinite gradients')
                    norm = gradient_l2_norm(gradients)
                    require(math.isfinite(norm), 'Invalid gradient norm')  # Zero is valid for missing-history batches.
                    optimizer.step()
                    losses[arm].append(float(loss.detach())); norms[arm].append(norm)
                observed.append(list(cpu_batch['sample_id']))
                if (batch_index + 1) % 10 == 0:
                    write_status(args, 'training', seed=seed, epoch=epoch + 1, completed_epochs=epoch,
                        batch=batch_index + 1, batches=51, branch_updates=epoch * 51 + batch_index + 1)
            require(observed == order and len(observed) == 51, 'Incomplete paired training epoch')
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            write_status(args, 'validation', seed=seed, epoch=epoch + 1, completed_epochs=epoch)
            validation_started = time.perf_counter()
            records = validation_records(backbone, parent, heads, val_loader, device,
                                         include_parent=epoch >= EPOCHS - 5)
            states = {}
            for arm in ARMS:
                check_records(records[arm], plan, domain_sha)
                metrics = (detailed_metrics(records[arm], history_val_ids) if epoch >= EPOCHS - 5
                    else pilot.epoch_metrics(records[arm]))
                histories[arm].append(dict(epoch=epoch + 1, updates=(epoch + 1) * 51,
                    learning_rate=1e-4, training_loss=sum(losses[arm]) / 51, validation=metrics,
                    training_checks=dict(updates=51, loss_finite=True, gradients_finite=True,
                        gradient_l2_min=min(norms[arm]), gradient_l2_max=max(norms[arm]),
                        shared_batch_order_sha256=canonical_sha(observed), shared_training_seconds=train_seconds,
                        shared_samples_per_second=811 / train_seconds, gpu_peak_allocated_bytes=allocated,
                        gpu_peak_reserved_bytes=reserved)))
                if epoch >= EPOCHS - 5:
                    diagnostic_path = directory / arm / f'diagnostics-epoch{epoch + 1:03d}.json'
                    atomic_json(diagnostic_path, correction_diagnostics(records[arm], records['parent']))
                    histories[arm][-1]['diagnostics_sha256'] = sha(diagnostic_path)
                state = make_state(heads[arm], optimizers[arm], bindings[arm], histories[arm], baseline, device)
                states[arm] = state
                if metrics['pooled_ap'] > deserialize_state(best[arm])['validation']['pooled_ap']:
                    best[arm] = serialize_state(state)
                if new_best[arm] is None or metrics['pooled_ap'] > deserialize_state(new_best[arm])['validation']['pooled_ap']:
                    new_best[arm] = serialize_state(state)
            del records
            verify_unchanged()
            transaction = build_transaction(states, best, new_best, capture_rng(device))
            publish_progress(directory, transaction, histories)
            write_status(args, 'training', step='paired_epoch_committed', seed=seed, epoch=epoch + 1,
                completed_epochs=epoch + 1, branch_updates=(epoch + 1) * 51,
                validation_seconds=time.perf_counter() - validation_started,
                last_validation_ap={arm: histories[arm][-1]['validation']['pooled_ap'] for arm in ARMS})
            print(json.dumps(dict(event='paired_epoch_completed', seed=seed, epoch=epoch + 1,
                validation_ap={arm: histories[arm][-1]['validation']['pooled_ap'] for arm in ARMS})), flush=True)

        reports = {arm: {} for arm in ARMS}
        for kind, payloads in (('overall_best', best), ('new_best', new_best)):
            selected = {arm: deserialize_state(payloads[arm]) for arm in ARMS}
            for arm in ARMS:
                heads[arm].load_state_dict(selected[arm]['model_state_dict'], strict=True)
                require(model_state_digest(heads[arm]) == selected[arm]['model_state_sha256'], 'Selected state model differs')
            records = validation_records(backbone, parent, heads, val_loader, device, include_parent=True)
            for arm in ARMS:
                check_records(records[arm], plan, domain_sha)
                metrics = detailed_metrics(records[arm], history_val_ids)
                previous.check_metric_replay(metrics, selected[arm]['validation'])
                diagnostic_path = directory / arm / f'diagnostics-selected-{kind}.json'
                atomic_json(diagnostic_path, correction_diagnostics(records[arm], records['parent']))
                qualifies = (selected[arm]['completed_epochs'] > 0 and metrics['pooled_ap'] >= baseline['pooled_ap'] + .001
                    and metrics['nll'] <= baseline['nll'] * 1.05 and metrics['brier'] <= baseline['brier'] * 1.05
                    and metrics['positive_event_ap'] >= baseline['positive_event_ap'] - .01)
                reports[arm][kind] = dict(completed_epochs=selected[arm]['completed_epochs'],
                    origin=selected[arm]['origin'], validation=metrics,
                    operating_points=metrics['operating_points'],
                    checkpoint_sha256=hashlib.sha256(payloads[arm]).hexdigest(),
                    qualifies_basic_checkpoint_safeguards=qualifies,
                    safeguard_scope='single_checkpoint_AP_NLL_Brier_event_AP_only_not_full_paired_acceptance_gate')
            del records
        seed_receipts[str(seed)] = {}
        for arm in ARMS:
            atomic_json(directory / arm / 'validation_selected.json', dict(independent_evaluation=False,
                selected_on='V_only', selection_frozen=True, **reports[arm]))
            receipt = dict(status='completed', seed=seed, arm=arm, completed_epochs=EPOCHS, branch_updates=EPOCHS * 51,
                last_validation=histories[arm][-1]['validation'],
                last_five_mean_ap=sum(row['validation']['pooled_ap'] for row in histories[arm][-5:]) / 5,
                last_five_means={key: sum(row['validation'][key] for row in histories[arm][-5:]) / 5
                    for key in ('pooled_ap', 'pooled_raw_logit_ap', 'nll', 'brier', 'pooled_nll', 'pooled_brier', 'positive_event_ap')},
                last_five_mean_precision_at_recall70=sum(row['validation']['operating_points']['validation_recall_70']['precision']
                    for row in histories[arm][-5:]) / 5,
                last_five_event_ap={event: sum(row['validation']['positive_event_ap_by_event'][event]
                    for row in histories[arm][-5:]) / 5
                    for event in histories[arm][-1]['validation']['positive_event_ap_by_event']},
                **reports[arm], files={name: sha(directory / arm / name) for name in
                    ('best.pth', 'last.pth', 'new_best.pth', 'history.json', 'configuration.json', 'validation_selected.json',
                     *(f'diagnostics-epoch{epoch:03d}.json' for epoch in range(EPOCHS - 4, EPOCHS + 1)),
                     'diagnostics-selected-overall_best.json', 'diagnostics-selected-new_best.json')})
            atomic_json(directory / arm / 'receipt.json', receipt)
            seed_receipts[str(seed)][arm] = receipt
        atomic_json(directory / 'receipt.json', dict(status='completed', seed=seed,
            arms=seed_receipts[str(seed)], transaction_sha256=sha(directory / 'transaction.pth')))
        write_status(args, 'seed_completed', seed=seed, completed_epochs=EPOCHS, completed_seeds=list(seed_receipts))
    verify_unchanged(full=True)
    atomic_json(args.output / 'receipt.json', dict(schema='fwt_history_event_completed_v2', status='completed',
        seeds=seed_receipts, plan_sha256=sha(args.output / 'plan.json'), input_binding_sha256=sha(args.output / 'input-binding.json'),
        history_audit_sha256=sha(args.output / 'history-audit.json'), parent_checkpoint_sha256=plan['parent_best_sha256'],
        event_support_sha256=sha(args.output / 'training-event-support.json'),
        fresh_runtime_differences_sha256=sha(args.output / 'fresh-runtime-differences.json'),
        comparison=development_comparison(seed_receipts, baseline),
        selection='No automatic deployment; retain original parent until paired development results reviewed',
        preservation=dict(old_artifacts_sha256_equal=True, current_run_array_stat_identity_unchanged=True,
            parent_array_content_sha256_equal=True,
            parent_array_stat_identity_equal=not migration['array_stat_changed_entries'],
            parent_runtime_identical=not migration['runtime_changes'],
            frozen_H1_and_parent_model_sha256_equal=True, parent_optimizer_updates=0),
        completed_at=time.time(), limitations=plan['limitations']))
    write_status(args, 'completed', completed_seeds=list(SEEDS), completed_epochs_per_seed=EPOCHS)


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if not (args.execute or args.preflight):
        print(json.dumps({k: v for k, v in plan.items() if k not in
            {'batch_orders', 'training_sample_ids', 'validation_sample_ids'}}, indent=2, allow_nan=False))
        return 0
    try:
        run(args, plan)
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
