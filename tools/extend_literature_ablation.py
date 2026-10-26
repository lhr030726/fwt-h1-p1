#!/usr/bin/env python3
"""Explicit seed-4 four-arm budget extension, from completed 10 to total 50 epochs.

The frozen pilot implementation is never patched. Its original schema is first
validated as-is; a named migration binds the preserved checkpoint bytes, source,
inputs and history to a new experiment budget and a new output directory.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import copy, deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import train_literature_ablation as pilot

SCHEMA = 'fwt_literature_extension_state_v1'
TRANSACTION_SCHEMA = 'fwt_literature_extension_transaction_v1'
EPOCHS, PARENT_EPOCHS, SEED = 50, 10, 4
ARMS, EXPECTED_AP = pilot.ARMS, pilot.EXPECTED_AP
read_json, sha, canonical_sha = pilot.read_json, pilot.sha, pilot.canonical_sha
tensor_cpu_tree = pilot.tensor_cpu_tree
serialize_state, deserialize_state = pilot.serialize_state, pilot.deserialize_state
validation_records, records_identity = pilot.validation_records, pilot.records_identity
epoch_metrics, selected_operating_points = pilot.epoch_metrics, pilot.selected_operating_points
selected_positive_event_ap = pilot.selected_positive_event_ap
cuda_interval, cuda_elapsed_seconds = pilot.cuda_interval, pilot.cuda_elapsed_seconds
publish_transaction = pilot.publish_transaction

ALLOWED_BINDING_CHANGES = ('recipe', 'stage', 'source_files', 'input_files_sha256',
                           'batch_order_sha256', 'extension')


def source_files():
    return pilot.source_files() | {'tools/extend_literature_ablation.py': sha(__file__)}


def extended_order(train_ids, parent_order):
    import torch
    pilot.old.validate_order(parent_order, train_ids)
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    orders = []
    for _ in range(EPOCHS):
        ids = [train_ids[i] for i in torch.randperm(len(train_ids), generator=generator).tolist()]
        orders.append([ids[i:i + 16] for i in range(0, len(ids), 16)])
    if orders[:PARENT_EPOCHS] != parent_order:
        raise ValueError('The regenerated first ten orders differ from the frozen pilot')
    for epoch in orders:
        if ([len(b) for b in epoch] != [16] * 50 + [11]
                or sorted(x for b in epoch for x in b) != sorted(train_ids)):
            raise ValueError('Invalid extension full-epoch order')
    return orders


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return args


def make_plan(args):
    parent_path = args.pilot_run / 'plan.json'
    parent, receipt = read_json(parent_path), read_json(args.pilot_run / 'receipt.json')
    if (parent.get('schema') != 'fwt_literature_ablation_plan_v1' or parent.get('stage') != 'pilot'
            or parent.get('seeds') != [SEED] or parent.get('arms') != list(ARMS)
            or canonical_sha(parent.get('recipe')) != canonical_sha(pilot.BASE_RECIPE)
            or receipt.get('schema') != 'fwt_literature_ablation_completed_v1'
            or receipt.get('status') != 'completed' or receipt.get('stage') != 'pilot'
            or receipt.get('plan_sha256') != sha(parent_path)
            or set(receipt.get('seeds', {})) != {'4'} or set(receipt['seeds']['4']) != set(ARMS)):
        raise ValueError('Require the original completed seed-4 four-arm ten-epoch pilot')
    if parent['source_files'] != pilot.source_files():
        raise ValueError('Frozen pilot source differs; only the new extension runner may be added')
    source = source_files()
    if set(source) - set(parent['source_files']) != {'tools/extend_literature_ablation.py'}:
        raise ValueError('Unexpected extension source additions')
    files = dict(parent['input_files_sha256'])
    parent_root = args.pilot_run.parent.parent
    files.update({str(parent_root / name): digest for name, digest in parent['source_files'].items()})
    for name in ('plan.json', 'receipt.json', 'input-binding.json', 'seed-004/transaction.pth'):
        files[str(args.pilot_run / name)] = sha(args.pilot_run / name)
    if receipt['input_binding_sha256'] != files[str(args.pilot_run / 'input-binding.json')]:
        raise ValueError('Pilot input binding digest differs')
    original_bindings, original_histories = {}, {}
    for arm in ARMS:
        directory = args.pilot_run / 'seed-004' / arm
        item = read_json(directory / 'receipt.json')
        if (item != receipt['seeds']['4'][arm] or item.get('status') != 'completed'
                or item.get('seed') != SEED or item.get('arm') != arm
                or item.get('completed_epochs') != PARENT_EPOCHS or item.get('total_updates') != 510):
            raise ValueError('Pilot arm is not completed: ' + arm)
        files[str(directory / 'receipt.json')] = sha(directory / 'receipt.json')
        for name, digest in item['files'].items():
            if Path(name).name != name:
                raise ValueError('Unexpected parent checkpoint path')
            files[str(directory / name)] = digest
        original_bindings[arm] = canonical_sha(read_json(directory / 'configuration.json'))
        original_histories[arm] = canonical_sha(read_json(directory / 'history.json'))
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Original source/input/artifact changed: ' + path)
    reference = read_json(parent['reference_plan'])
    paths = reference['input_paths']
    for key in ('manifest', 'normalizer', 'backbone'):
        setattr(args, key, Path(paths[key]).resolve())
        if str(getattr(args, key)) not in parent['input_files_sha256']:
            raise ValueError('Reference data path not bound by pilot')
    raw_manifest = read_json(args.manifest)
    if {r['role'] for r in raw_manifest['samples']} != {'backbone_train', 'residual_train', 'validation'}:
        raise ValueError('Held-out roles forbidden in extension')
    for role, key, count in (('residual_train', 'training_sample_ids', 811), ('validation', 'validation_sample_ids', 568)):
        ids = [r['sample_id'] for r in raw_manifest['samples'] if r['role'] == role]
        if ids != parent[key] or len(ids) != count or len(set(ids)) != count:
            raise ValueError('Original Tr/V identities differ')
    order = extended_order(parent['training_sample_ids'], parent['batch_orders']['4'])
    recipe = dict(parent['recipe'], epochs=EPOCHS)
    plan = dict(schema='fwt_literature_extension_plan_v1', stage='extension', candidate=None,
        seeds=[SEED], arms=list(ARMS), recipe=recipe, parent_epochs=PARENT_EPOCHS,
        total_epochs=EPOCHS, additional_epochs=EPOCHS - PARENT_EPOCHS, total_updates_per_arm=EPOCHS * 51,
        pilot_run=str(args.pilot_run), parent_plan_sha256=sha(parent_path),
        parent_receipt_sha256=sha(args.pilot_run / 'receipt.json'),
        parent_transaction_sha256=sha(args.pilot_run / 'seed-004/transaction.pth'),
        parent_source_files=parent['source_files'], source_files=source, input_files_sha256=files,
        source_migration=dict(added=['tools/extend_literature_ablation.py'], modified=[],
            unchanged=sorted(parent['source_files'])),
        original_binding_sha256=original_bindings, original_history_sha256=original_histories,
        allowed_binding_migration_fields=list(ALLOWED_BINDING_CHANGES),
        batch_orders={'4': order}, batch_order_sha256={'4': canonical_sha(order)},
        parent_batch_order_sha256=parent['batch_order_sha256'], input_paths={k: str(getattr(args, k))
            for k in ('manifest', 'normalizer', 'backbone')},
        output=str(args.output), resume_run=str(args.resume_run) if args.resume_run else None,
        dry_run=not args.execute, expected_h1_validation_ap=EXPECTED_AP,
        limits=['A new 50-epoch budget study after inspecting the completed 10-epoch pilot.',
                'Four existing seed-4 heads continue from last model+AdamW+shared RNG; no fresh seed or best restart.',
                'Original ten-epoch artifacts and source stay untouched; best/history are inherited.',
                'Tr/V development only. No final or calibration arrays. Not independent confirmation.',
                'All hyperparameters except the total epoch budget are unchanged.'])
    for key in ('training_sample_ids', 'validation_sample_ids', 'diffusion_config', 'data_config',
                'model_config', 'manifest_sha256', 'normalizer_sha256', 'bound_data_metadata_paths'):
        plan[key] = parent[key]
    if args.resume_run:
        previous = read_json(args.resume_run / 'plan.json')
        ignore = {'output', 'resume_run', 'dry_run'}
        if canonical_sha({k: v for k, v in plan.items() if k not in ignore}) != canonical_sha(
                {k: v for k, v in previous.items() if k not in ignore}):
            raise ValueError('Extension resume plan/source/data/budget differs')
    return plan


def extension_binding(original, plan):
    if canonical_sha(original) != plan['original_binding_sha256'][original['arm']]:
        raise ValueError('Original arm binding differs from preserved configuration')
    if (original.get('stage') != 'pilot' or original.get('candidate') is not None
            or original.get('recipe') != pilot.BASE_RECIPE
            or plan['recipe'] != dict(original['recipe'], epochs=EPOCHS)
            or original.get('source_files') != plan['parent_source_files']
            or set(plan['source_files']) - set(original['source_files']) != {'tools/extend_literature_ablation.py'}
            or any(plan['source_files'].get(k) != v for k, v in original['source_files'].items())
            or any(plan['input_files_sha256'].get(k) != v for k, v in original['input_files_sha256'].items())
            or original['batch_order_sha256'] != plan['parent_batch_order_sha256']['4']):
        raise ValueError('Unlisted recipe/source/input/order migration')
    result = deepcopy(original)
    result.update(recipe=plan['recipe'], stage='extension', source_files=plan['source_files'],
        input_files_sha256=plan['input_files_sha256'], batch_order_sha256=plan['batch_order_sha256']['4'],
        extension=dict(parent_epochs=10, total_epochs=50, parent_plan_sha256=plan['parent_plan_sha256'],
            parent_transaction_sha256=plan['parent_transaction_sha256'], parent_binding_sha256=canonical_sha(original),
            allowed_changes=list(ALLOWED_BINDING_CHANGES)))
    changed = {key for key in original.keys() | result.keys() if original.get(key) != result.get(key)}
    if not changed <= set(ALLOWED_BINDING_CHANGES):
        raise ValueError('Unlisted original binding migration')
    return result


def migrate_state(original, binding, original_checkpoint_sha):
    if original.get('schema') != pilot.SCHEMA:
        raise ValueError('Migration requires the unchanged original checkpoint schema')
    result = deepcopy(original)
    result.update(schema=SCHEMA, binding=binding,
        inherited_checkpoint=dict(schema=original['schema'], sha256=original_checkpoint_sha,
            model_state_sha256=original['model_state_sha256'], completed_epochs=original['completed_epochs']))
    result.pop('selected_checkpoint_sha256', None)
    return result


def restore_state(state, model, optimizer, binding, device, parent_history):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import restore_training_parameters, validate_rng, restore_rng
    completed = state.get('completed_epochs')
    if state.get('schema') != SCHEMA or state.get('binding') != binding:
        raise ValueError('Extension checkpoint schema/binding differs')
    if (type(completed) is not int or not 1 <= completed <= EPOCHS or state.get('updates') != completed * 51
            or state.get('epoch') != completed - 1 or len(state.get('history', [])) != completed
            or [r['epoch'] for r in state['history']] != list(range(completed))
            or state['history'][:PARENT_EPOCHS] != parent_history[:min(completed, PARENT_EPOCHS)]):
        raise ValueError('Extension checkpoint progress/original history differs')
    validate_rng(state['rng_state'], device)
    restore_training_parameters(dict(state, selected=dict(state_dict=state['model_state_dict'])), model, optimizer)
    if model_state_digest(model) != state['model_state_sha256']:
        raise ValueError('Extension model digest differs')
    restore_rng(state['rng_state'], device)


def build_transaction(states, best_bytes, rng):
    # Only reuse the schema-agnostic byte packing; validation is explicitly new.
    if (set(states) != set(ARMS) or set(best_bytes) != set(ARMS)
            or any(s.get('schema') != SCHEMA or not PARENT_EPOCHS <= s.get('completed_epochs', -1) <= EPOCHS
                   for s in states.values())):
        raise ValueError('Extension transaction requires all four completed states within its budget')
    transaction = pilot.build_transaction(SEED, states, best_bytes, rng)
    transaction['schema'] = TRANSACTION_SCHEMA
    return transaction


def decode_transaction(transaction):
    if (transaction.get('schema') != TRANSACTION_SCHEMA or transaction.get('seed') != SEED
            or set(transaction.get('arms', {})) != set(ARMS)
            or type(transaction.get('completed_epochs')) is not int
            or not PARENT_EPOCHS <= transaction['completed_epochs'] <= EPOCHS):
        raise ValueError('Extension transaction schema/arms/progress differs')
    decoded = {}
    for arm, entry in transaction['arms'].items():
        last, best = (deserialize_state(pilot.transaction_checkpoint_bytes(entry, kind)) for kind in ('last', 'best'))
        if (last.get('schema') != SCHEMA or best.get('schema') != SCHEMA
                or last['binding'] != best['binding'] or last['binding'].get('arm') != arm
                or last['binding'].get('seed') != SEED or last['completed_epochs'] != transaction['completed_epochs']
                or len(last['history']) != last['completed_epochs'] or last['updates'] != last['completed_epochs'] * 51
                or last['selected_checkpoint_sha256'] != entry['best_sha256']
                or best['completed_epochs'] != best['epoch'] + 1 or best['updates'] != best['completed_epochs'] * 51
                or last['selected_epoch'] != best['epoch'] or best['history'] != last['history'][:best['completed_epochs']]
                or best['validation_ap'] != last['history'][best['epoch']]['validation']['pooled_ap']
                or best['epoch'] != max(range(len(last['history'])), key=lambda i: last['history'][i]['validation']['pooled_ap'])):
            raise ValueError('Extension transaction best/last selection or progress differs')
        decoded[arm] = last, best
    return decoded


def migrate_parent(transaction, heads, optimizers, original_bindings, plan, device):
    """Strictly accept old states before the explicit new-schema migration."""
    from fwt_v2.training_state import restore_rng, validate_rng
    decoded = pilot.decode_transaction(transaction, SEED, ARMS)
    if transaction['completed_epochs'] != PARENT_EPOCHS:
        raise ValueError('The parent transaction must end at exactly ten epochs')
    validate_rng(transaction['rng_state'], device)
    states, best_bytes, bindings, histories = {}, {}, {}, {}
    for arm, (last, best) in decoded.items():
        original = original_bindings[arm]
        pilot.restore_state(best, heads[arm], optimizers[arm], original, device)
        pilot.restore_state(last, heads[arm], optimizers[arm], original, device)
        if canonical_sha(last['history']) != plan['original_history_sha256'][arm]:
            raise ValueError('Parent last history differs from preserved history file')
        binding = extension_binding(original, plan)
        entry = transaction['arms'][arm]
        states[arm] = migrate_state(last, binding, entry['last_sha256'])
        best_bytes[arm] = serialize_state(migrate_state(best, binding, entry['best_sha256']))
        bindings[arm], histories[arm] = binding, last['history']
    restore_rng(transaction['rng_state'], device)
    return build_transaction(states, best_bytes, transaction['rng_state']), bindings, histories


def restore_transaction(transaction, heads, optimizers, bindings, parent_histories, device):
    from fwt_v2.training_state import restore_rng, validate_rng
    decoded = decode_transaction(transaction)
    validate_rng(transaction['rng_state'], device)
    histories, best_bytes, best_aps, selected_epochs = {}, {}, {}, {}
    for arm, (last, best) in decoded.items():
        restore_state(best, heads[arm], optimizers[arm], bindings[arm], device, parent_histories[arm])
        restore_state(last, heads[arm], optimizers[arm], bindings[arm], device, parent_histories[arm])
        histories[arm], best_aps[arm], selected_epochs[arm] = last['history'], best['validation_ap'], best['epoch']
        best_bytes[arm] = pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
    restore_rng(transaction['rng_state'], device)
    return transaction['completed_epochs'], histories, best_bytes, best_aps, selected_epochs


def write_status(args, phase, **details):
    from fwt_v2.artifacts import atomic_json
    value = dict(schema='fwt_literature_extension_status_v1', phase=phase, stage='extension', total_epochs=EPOCHS,
        parent_epochs=PARENT_EPOCHS, updated_at=time.time(), started_at=args._run_started_at,
        resources=pilot.old.resource_snapshot(), **details)
    atomic_json(args.output / 'status.json', value)
    with (args.output / 'resource-history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def execute(args, plan):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.artifacts import atomic_json, atomic_checkpoint, output_path
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import audited_manifest, load_det, model_state_digest, seed_everything, to_device
    from fwt_v2.gradient_diagnostics import gradients_finite, gradient_l2_norm
    from fwt_v2.literature_refiner import masked_refinement_loss
    from fwt_v2.manifest import manifest_digest, samples_for_role
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    from fwt_v2.training_state import capture_rng, restore_rng, sample_file_identity, optimizer_state_integrity

    output_path(args.output / "plan.json")
    if args.output.exists():
        raise FileExistsError("Output must be a new directory, also for resume")
    pilot.old.assert_gpu_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_created, args._run_started_at = True, time.time()
    atomic_json(args.output / "plan.json", plan)
    write_status(args, "verify_data", step="original_Tr_V_array_hashes")
    device = torch.device("cuda:0")
    os.chdir(args.manifest.parent)
    manifest = audited_manifest(args.manifest)
    if manifest_digest(manifest) != plan["manifest_sha256"] or sha(args.normalizer) != plan["normalizer_sha256"]:
        raise ValueError("Manifest/normalizer content identity changed")
    train_rows = samples_for_role(manifest, "residual_train")
    val_rows = samples_for_role(manifest, "validation")
    for rows, key in ((train_rows, "training_sample_ids"), (val_rows, "validation_sample_ids")):
        if [r["sample_id"] for r in rows] != plan[key]:
            raise ValueError("Tr/V original identities changed")
    samples = train_rows + val_rows
    sample_identity = sample_file_identity(samples)
    array_hashes = {str(Path(row[k]).resolve()): sha(row[k]) for row in samples for k in ("input_path", "label_path")}
    pilot.old.verify_selected_array_index(read_json(plan["bound_data_metadata_paths"]["arrays_index"]),
                                   samples, array_hashes, args.manifest.parent)
    write_status(args, "verify_data", step="validate_hourly_wind_once")
    dataset = RCDADataset(manifest, "residual_train", NormalizationStats.load(args.normalizer), DataConfig(**plan["data_config"]))
    val_dataset = copy(dataset)
    val_dataset.role, val_dataset.samples = "validation", val_rows
    cache = SharedTensorCache(16 * 1024**3)
    training, validation = CachedDataset(dataset, cache), CachedDataset(val_dataset, cache)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
    id_index = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    seed_everything(0)
    backbone, bundle = load_det(args.backbone, device)
    if (not bundle.get("selection_frozen") or bundle["family"] != "det"
            or any(canonical_sha(bundle[k]) != canonical_sha(plan[k]) for k in
                   ("data_config", "model_config", "manifest_sha256", "normalizer_sha256"))):
        raise ValueError("Fixed H1 metadata differs from reference")
    backbone.requires_grad_(False).eval()
    h1_sha = model_state_digest(backbone)
    write_status(args, "h1_baseline", step="full_original_V")
    baseline = validation_records(backbone, None, val_loader, device)
    baseline_metrics, domain_sha = epoch_metrics(baseline), records_identity(baseline)
    if ([r["sample_id"] for r in baseline] != plan["validation_sample_ids"]
            or abs(baseline_metrics["pooled_ap"] - EXPECTED_AP) > 1e-6):
        raise RuntimeError("Original H1 full-V AP reproduction failed")
    del baseline
    atomic_json(args.output / "baseline.json", dict(validation=baseline_metrics,
        backbone_state_sha256=h1_sha, target_and_masks_sha256=domain_sha))
    common = dict(recipe=plan["recipe"], stage=plan["stage"], candidate=plan["candidate"],
        source_files=plan["source_files"], input_files_sha256=plan["input_files_sha256"],
        array_files_sha256=array_hashes, array_stat_identity=sample_identity,
        training_sample_ids=plan["training_sample_ids"], validation_sample_ids=plan["validation_sample_ids"],
        target_and_masks_sha256=domain_sha, frozen_backbone_state_sha256=h1_sha,
        runtime=dict(python=sys.version, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                     cpu_threads=2, interop_threads=2, matmul_tf32=False, cudnn_tf32=True))
    atomic_json(args.output / "input-binding.json", common)

    def verify_unchanged():
        if source_files() != plan["source_files"]:
            raise RuntimeError("Extension source changed during execution")
        if any(sha(p) != h for p, h in plan["input_files_sha256"].items()):
            raise RuntimeError("Reference inputs changed during execution")
        if sample_file_identity(samples) != sample_identity:
            raise RuntimeError("Tr/V array identity changed")
        if model_state_digest(backbone) != h1_sha or any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
            raise RuntimeError("Frozen H1 changed or acquired gradients")

    original_common = read_json(args.pilot_run / 'input-binding.json')
    unchanged_common = set(common) - {'recipe', 'stage', 'source_files', 'input_files_sha256'}
    if any(canonical_sha(common[k]) != canonical_sha(original_common[k]) for k in unchanged_common):
        raise ValueError('Original runtime, arrays, masks or frozen H1 binding differs')
    config = DiffusionConfig(**dict(plan['diffusion_config'], output_initialization='zero', prediction_type='clean_probability'))
    seed = SEED
    seed_dir = args.output / 'seed-004'
    seed_dir.mkdir()
    heads, optimizers, original_bindings, initial_hashes = {}, {}, {}, {}
    for arm in ARMS:
        (seed_dir / arm).mkdir()
        original = read_json(args.pilot_run / 'seed-004' / arm / 'configuration.json')
        if any(canonical_sha(original.get(k)) != canonical_sha(v) for k, v in original_common.items()):
            raise ValueError('Original per-arm binding differs from original common binding')
        head, initial_sha, _ = pilot.initialize_head(seed, arm, args.backbone, device, config, h1_sha)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
        expected = dict(seed=seed, arm=arm, head_configuration=head.configuration(),
            head_parameters=sum(p.numel() for p in head.parameters()),
            trainable_head_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
            auxiliary_weight=.05 if arm.endswith('tversky') else 0.,
            initial_head_state_sha256=initial_sha, optimizer_defaults=optimizer.defaults,
            batch_order_sha256=plan['parent_batch_order_sha256']['4'])
        if any(canonical_sha(original.get(k)) != canonical_sha(v) for k, v in expected.items()):
            raise ValueError('Head/optimizer/initialization changed from original pilot')
        heads[arm], optimizers[arm], original_bindings[arm] = head, optimizer, original
        initial_hashes[arm] = initial_sha
    parent_transaction = torch.load(args.pilot_run / 'seed-004/transaction.pth', map_location='cpu', weights_only=True)
    for arm, entry in parent_transaction['arms'].items():
        for kind in ('best', 'last'):
            if entry[kind + '_sha256'] != sha(args.pilot_run / 'seed-004' / arm / (kind + '.pth')):
                raise ValueError('Original transaction and best/last mirror differ')
    transaction, bindings, parent_histories = migrate_parent(parent_transaction, heads, optimizers,
        original_bindings, plan, device)
    atomic_json(args.output / 'migration.json', dict(schema='fwt_literature_extension_migration_v1',
        parent_transaction_sha256=plan['parent_transaction_sha256'], parent_epochs=10, total_epochs=50,
        original_binding_sha256=plan['original_binding_sha256'],
        original_history_sha256=plan['original_history_sha256'],
        allowed_binding_changes=list(ALLOWED_BINDING_CHANGES),
        preserved={arm: {kind: dict(checkpoint_sha256=parent_transaction['arms'][arm][kind + '_sha256'],
            model_state_sha256=deserialize_state(pilot.transaction_checkpoint_bytes(parent_transaction['arms'][arm], kind))['model_state_sha256'])
            for kind in ('best', 'last')} for arm in ARMS}))
    for arm, binding in bindings.items():
        if any(canonical_sha(binding.get(k)) != canonical_sha(v) for k, v in common.items()):
            raise ValueError('Migrated arm binding differs from the explicit extension common binding')
        atomic_json(seed_dir / arm / 'configuration.json', binding)
    if args.resume_run:
        previous = args.resume_run / 'seed-004/transaction.pth'
        if not previous.is_file():
            raise ValueError('Extension resume requires an authoritative completed-epoch transaction')
        transaction = torch.load(previous, map_location='cpu', weights_only=True)
    start_epoch, histories, best_bytes, best_aps, selected_epochs = restore_transaction(
        transaction, heads, optimizers, bindings, parent_histories, device)
    publish_transaction(seed_dir, transaction)
    for arm in ARMS:
        atomic_json(seed_dir / arm / 'history.json', histories[arm])
    write_status(args, 'training', seed=SEED, epoch=start_epoch, completed_epochs=start_epoch,
        step='original_full_states_migrated' if not args.resume_run else 'extension_full_states_restored',
        updates_per_arm=start_epoch * 51)
    for epoch in range(start_epoch, EPOCHS):
        verify_unchanged()
        write_status(args, "training", seed=seed, epoch=epoch + 1, completed_epochs=epoch, batch=0, batches=51)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for head in heads.values():
            head.train()
        batches = [[id_index[sid] for sid in batch] for batch in plan["batch_orders"][str(seed)][epoch]]
        loader = DataLoader(training, batch_sampler=batches, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        losses, norms = {a: [] for a in heads}, {a: [] for a in heads}
        gpu_intervals = defaultdict(list)
        observed = []
        for index, cpu_batch in enumerate(loader):
            if list(cpu_batch["sample_id"]) != plan["batch_orders"][str(seed)][epoch][index]:
                raise RuntimeError("Actual batch differs from the frozen shared order")
            batch = to_device(cpu_batch, device)
            with torch.no_grad(), cuda_interval(gpu_intervals, "shared_H1"):
                outputs = backbone(batch["x_norm"], batch["physics"], return_decoder_features=True)
            for arm, head in heads.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                with cuda_interval(gpu_intervals, arm + "/forward_and_loss"):
                    logits = head(outputs, batch["physics"])
                    parts = masked_refinement_loss(logits, batch["target"], batch["valid_mask"],
                        aux_weight=bindings[arm]["auxiliary_weight"], alpha_fp=.3, beta_fn=.7)
                loss = parts["total"]
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite or nonscalar loss")
                with cuda_interval(gpu_intervals, arm + "/backward"):
                    loss.backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                if not gradients or not gradients_finite(gradients):
                    raise RuntimeError("Missing/nonfinite head gradients")
                norm = gradient_l2_norm(gradients)
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Invalid aggregate gradient")
                with cuda_interval(gpu_intervals, arm + "/optimizer"):
                    optimizer.step()
                losses[arm].append({k: float(v.detach()) for k, v in parts.items()})
                norms[arm].append(norm)
            observed.append(list(cpu_batch["sample_id"]))
            if (index + 1) % 10 == 0:
                write_status(args, "training", seed=seed, epoch=epoch + 1, batch=index + 1,
                             batches=51, completed_epochs=epoch, updates_per_arm=epoch * 51 + index + 1)
        if observed != plan["batch_orders"][str(seed)][epoch]:
            raise RuntimeError("Incomplete shared training epoch")
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        gpu_seconds = cuda_elapsed_seconds(gpu_intervals)
        train_peak_allocated = torch.cuda.max_memory_allocated()
        train_peak_reserved = torch.cuda.max_memory_reserved()
        states = {}
        for arm, head in heads.items():
            write_status(args, "validation", seed=seed, epoch=epoch + 1, arm=arm, completed_epochs=epoch)
            records = validation_records(backbone, head, val_loader, device, arm.startswith("compact_"))
            if ([r["sample_id"] for r in records] != plan["validation_sample_ids"]
                    or records_identity(records) != domain_sha):
                raise RuntimeError("Full-V identities/targets/masks changed")
            metrics = epoch_metrics(records)
            del records
            ap = metrics["pooled_ap"]
            if not math.isfinite(ap):
                raise RuntimeError("Nonfinite validation AP")
            checks = dict(updates=51, loss_finite=True, gradients_finite=True,
                gradient_l2_min=min(norms[arm]), gradient_l2_max=max(norms[arm]),
                shared_batch_order_sha256=canonical_sha(observed),
                shared_training_seconds=train_seconds, shared_H1_cuda_stream_seconds=gpu_seconds["shared_H1"],
                head_cuda_stream_seconds={phase: gpu_seconds[arm + "/" + phase]
                    for phase in ("forward_and_loss", "backward", "optimizer")},
                timing_scope="CUDA event stream elapsed, includes launch gaps; not GPU kernel busy time",
                gpu_peak_allocated_bytes=train_peak_allocated, gpu_peak_reserved_bytes=train_peak_reserved)
            history = histories[arm]
            history.append(dict(epoch=epoch, training_loss=sum(v["total"] for v in losses[arm]) / 51,
                loss_components={k: sum(v[k] for v in losses[arm]) / 51 for k in ("bce", "tversky")},
                validation=metrics, training_checks=checks))
            state = dict(schema=SCHEMA, checkpoint_kind="last_full_training_state", binding=bindings[arm],
                completed_epochs=epoch + 1, updates=(epoch + 1) * 51, epoch=epoch,
                model_state_dict=tensor_cpu_tree(head.state_dict()), model_state_sha256=model_state_digest(head),
                optimizer_state_dict=tensor_cpu_tree(optimizers[arm].state_dict()),
                rng_state=capture_rng(device), history=deepcopy(history), validation_ap=ap)
            state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
            if ap > best_aps[arm]:
                best_aps[arm], selected_epochs[arm] = ap, epoch
                best = dict(state, checkpoint_kind="selected_best_full_training_state")
                best_bytes[arm] = serialize_state(best)
            state.update(best_validation_ap=best_aps[arm], selected_epoch=selected_epochs[arm])
            states[arm] = state
        verify_unchanged()
        transaction = build_transaction(states, best_bytes, capture_rng(device))
        publish_transaction(seed_dir, transaction)
        for arm in heads:
            atomic_json(seed_dir / arm / "history.json", histories[arm])
        write_status(args, "training", seed=seed, epoch=epoch + 1, completed_epochs=epoch + 1,
                     step="paired_epoch_committed", best_ap=best_aps, updates_per_arm=(epoch + 1) * 51,
                     shared_epoch_seconds=time.perf_counter() - started)
        print(json.dumps(dict(event="paired_epoch_completed", seed=seed, epoch=epoch + 1,
                              best_ap=best_aps, seconds=time.perf_counter() - started)), flush=True)
    seed_receipts = {}
    for arm, head in heads.items():
        if len(histories[arm]) != EPOCHS:
            raise RuntimeError("Arm did not finish fifty epochs")
        best = deserialize_state(best_bytes[arm])
        head.load_state_dict(best["model_state_dict"], strict=True)
        if model_state_digest(head) != best["model_state_sha256"]:
            raise RuntimeError("Selected model digest mismatch")
        records = validation_records(backbone, head, val_loader, device, arm.startswith("compact_"))
        if records_identity(records) != domain_sha:
            raise RuntimeError("Selected V target/mask identity changed")
        metrics = epoch_metrics(records)
        metrics.update(selected_positive_event_ap(records))
        if abs(metrics["pooled_ap"] - best["validation_ap"]) > 1e-7:
            raise RuntimeError("Selected checkpoint AP did not reproduce")
        operating = selected_operating_points(records)
        del records
        arm_dir = seed_dir / arm
        atomic_json(arm_dir / "history.json", histories[arm])
        atomic_json(arm_dir / "validation_selected.json", dict(schema="fwt_literature_extension_selected_v1",
            seed=seed, arm=arm, completed_epochs=EPOCHS, epoch=best["epoch"], selection_frozen=True,
            independent_evaluation=False, checkpoint_sha256=sha(arm_dir / "best.pth"),
            model_state_sha256=best["model_state_sha256"], validation=metrics, operating_points=operating))
        receipt = dict(schema="fwt_literature_extension_arm_completed_v1", status="completed", seed=seed, arm=arm,
            completed_epochs=EPOCHS, total_updates=EPOCHS * 51, best_epoch=best["epoch"],
            best_validation_ap=metrics["pooled_ap"], last_validation_ap=histories[arm][-1]["validation"]["pooled_ap"],
            binding_sha256=canonical_sha(bindings[arm]), initial_head_state_sha256=initial_hashes[arm],
            head_parameters=bindings[arm]["head_parameters"],
            files={name: sha(arm_dir / name) for name in ("best.pth", "last.pth",
                "configuration.json", "history.json", "validation_selected.json")})
        atomic_json(arm_dir / "receipt.json", receipt)
        seed_receipts[arm] = receipt
    verify_unchanged()
    atomic_json(args.output / 'receipt.json', dict(schema='fwt_literature_extension_completed_v1', status='completed',
        stage='extension', candidate=None, seeds={'4': seed_receipts}, completed_epochs=EPOCHS,
        parent_epochs=PARENT_EPOCHS, additional_epochs=EPOCHS-PARENT_EPOCHS,
        parent_plan_sha256=plan['parent_plan_sha256'], parent_transaction_sha256=plan['parent_transaction_sha256'],
        initial_head_state_sha256={'4': initial_hashes}, batch_order_sha256=plan['batch_order_sha256'],
        baseline_ap=baseline_metrics['pooled_ap'], input_binding_sha256=sha(args.output / 'input-binding.json'),
        plan_sha256=sha(args.output / 'plan.json'), transaction_sha256=sha(seed_dir / 'transaction.pth'),
        migration_sha256=sha(args.output / 'migration.json'), completed_at=time.time(), limitations=plan['limits']))
    write_status(args, 'completed', completed_seeds=[SEED], completed_arms=list(ARMS), completed_epochs=EPOCHS,
        updates_total=len(ARMS)*EPOCHS*51, additional_updates_total=len(ARMS)*(EPOCHS-PARENT_EPOCHS)*51)


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k not in {
            'batch_orders', 'training_sample_ids', 'validation_sample_ids'}}, indent=2, allow_nan=False))
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
