"""Opt-in two-update P1 zero-output mechanism witness on two real Tr samples.

Default is a plan with no torch/data/GPU access. Execution uses disposable
random frozen H1 weights, original source checks and FP32; no checkpoint IO,
validation/test arrays, formal training, or model-quality conclusion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_real_batches import require_no_jobs, write_report
from benchmark_sample_cache import fingerprint, spread_indices
from verify_real_batch import checked_output, configure_gpu_mode, compare, rng_snapshot

SCHEMA = 'fwt_zero_output_real_witness_v1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--blocked-pid', type=int, action='append', default=[])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.device_index < 0 or any(pid < 1 for pid in args.blocked_pid):
        parser.error('device index must be nonnegative and blocked PIDs positive')
    return args


def make_plan(args):
    return dict(schema=SCHEMA, status='plan_only', created_at=datetime.now(timezone.utc).isoformat(),
        role='residual_train', samples=2, batch_size=1, updates_requested=2, seed=0,
        precision='FP32', cpu_threads=2, output_initialization='zero', K=8, steps=4,
        prediction_type='clean_probability', training_timestep_sampling='antithetic',
        checkpoint_reads=0, checkpoint_writes=0, validation_arrays_read=0, heldout_arrays_read=0,
        purpose='Initialization/gradient/frozen-state mechanism only; no quality or speed claim',
        probability_absolute_tolerance=1e-7,
        limitation='Random frozen H1 and two Tr samples do not establish trained-checkpoint behavior or convergence',
        initial_predictions=[], updates=[])


def verify_gradients(named_parameters, step, torch):
    gradients = {name: p.grad for name, p in named_parameters}
    if not gradients or any(g is None or not bool(torch.isfinite(g).all()) for g in gradients.values()):
        raise RuntimeError('Missing/nonfinite denoiser gradient')
    output = [g for name, g in gradients.items() if name.startswith('denoiser.output.')]
    upstream = [g for name, g in gradients.items() if not name.startswith('denoiser.output.')]
    output_nonzero = sum(int(torch.count_nonzero(g)) for g in output)
    upstream_nonzero = sum(int(torch.count_nonzero(g)) for g in upstream)
    if output_nonzero == 0 or (step == 1 and upstream_nonzero != 0) or (step == 2 and upstream_nonzero == 0):
        raise RuntimeError('Unexpected zero-initialization gradient-opening behavior')
    norm = math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                        for g in gradients.values()))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError('Total gradient norm must be finite and positive')
    return dict(gradients_finite=True, output_nonzero_gradient_elements=output_nonzero,
                upstream_nonzero_gradient_elements=upstream_nonzero, total_gradient_l2=norm)


def execute(args, report, save):
    require_no_jobs(args)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data._utils.collate import default_collate
    import train as portable_train
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig
    from fwt_v2.engine import build_det, model_state_digest, raw_probability, seed_everything, to_device
    from fwt_v2.models import FWTConfig
    from fwt_v2.portable import bound_file, file_sha256, read_json

    root = args.data_root.resolve()
    device = torch.device(f'cuda:{args.device_index}')
    torch.set_num_threads(2)
    controls = configure_gpu_mode(SimpleNamespace(deterministic_gpu=False), torch)
    check_args = portable_train.parser().parse_args(['--phase', 'det', '--data-root', str(root),
        '--run-dir', str(ROOT / 'profile_output' / ('unused-' + uuid.uuid4().hex)),
        '--p1-zero-init', '--device', str(device)])
    checked = portable_train.preflight(check_args)
    config = read_json(checked['config_path'])
    bundle = read_json(root / 'bundle.json')
    paths = [root / 'bundle.json', root / bundle['original_full_manifest']['path'],
             root / bundle['original_normalizer']['path']]
    paths += [bound_file(root, ref) for ref in bundle['files'].values()]
    weather = read_json(bound_file(root, bundle['files']['weather']))
    paths += [bound_file(root, ref) for ref in weather['files']]
    paths += list((ROOT / 'fwt_v2').glob('*.py')) + [ROOT / 'train.py', Path(__file__)]
    paths += [ROOT / 'scripts' / name for name in ('benchmark_real_batches.py',
              'benchmark_sample_cache.py', 'benchmark_batches.py', 'verify_real_batch.py')]
    watched = {str(path.resolve()): file_sha256(path) for path in paths}
    report['provenance'] = dict(binding=checked['binding'], source_and_metadata_sha256=dict(watched),
        witness_sha256=file_sha256(__file__), role_counts=checked['role_counts'])
    report['environment'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                 cpu_threads=torch.get_num_threads(), gpu_controls=controls)
    indexed = {(row['sample_id'], row['field']): row for row in
               read_json(bound_file(root, bundle['files']['arrays_index']))['files']}
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        data_config = DataConfig(**config['data'])
        dataset = RCDADataset(checked['manifest'], 'residual_train',
                              NormalizationStats.load(checked['normalizer']), data_config)
        report['full_weather_source_verification_passed'] = True
        indices = spread_indices(len(dataset), 2)
        if len(indices) != 2:
            raise RuntimeError('Two distinct Tr samples are required')
        cpu_batches, selected = [], []
        for index in indices:
            row, hashes = dataset.samples[index], {}
            for field in ('input_path', 'label_path'):
                bound = indexed[(row['sample_id'], field)]
                digest = file_sha256(row[field])
                if bound['role'] != 'residual_train' or digest != bound['sha256']:
                    raise RuntimeError('Selected Tr array differs from its published bytes')
                hashes[field] = digest
                watched[str(Path(row[field]).resolve())] = digest
            batch = default_collate([dataset[index]])
            if tuple(batch['target'].shape) != (1, 1, 256, 256):
                raise RuntimeError('Expected full 256x256 batch-1 real data')
            cpu_batches.append(batch)
            selected.append(dict(index=index, sample_id=row['sample_id'], arrays_sha256=hashes,
                                 cpu_batch_sha256=fingerprint(batch, torch.Tensor)))
        report['selected_samples'] = selected
        save()
        require_no_jobs(args)
        if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
            raise RuntimeError('Configured CUDA device unavailable')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.85, device)
        report['environment']['gpu_name'] = torch.cuda.get_device_name(device)
        seed_everything(0)
        model_config = FWTConfig(**config['model'])
        backbone = build_det(model_config, 'fwt', data_config).to(device).requires_grad_(False).eval()
        frozen_before = model_state_digest(backbone)
        settings = dict(config['diffusion'], backbone_feature_channels=model_config.decoder_widths,
                        relational_channels=model_config.prop_channels)
        constructor_rng = torch.get_rng_state().clone()
        random_head = ConditionalResidualDiffusion(DiffusionConfig(**settings))
        random_rng_after = torch.get_rng_state().clone()
        torch.set_rng_state(constructor_rng)
        head = ConditionalResidualDiffusion(DiffusionConfig(**settings, output_initialization='zero'))
        if not torch.equal(random_rng_after, torch.get_rng_state()):
            raise RuntimeError('Zero initialization changed constructor RNG draws')
        for name, value in head.state_dict().items():
            if name in ('denoiser.output.weight', 'denoiser.output.bias'):
                if bool(value.any()):
                    raise RuntimeError('Output parameters were not zero')
            elif not torch.equal(value, random_head.state_dict()[name]):
                raise RuntimeError('Zero initialization changed another parameter/buffer: ' + name)
        del random_head
        head = head.to(device).eval()
        report['constructor'] = dict(rng_preserved=True, other_state_exactly_equal=True,
                                     output_weight_and_bias_zero=True)
        initial_head = model_state_digest(head)
        for cpu_batch in cpu_batches:
            require_no_jobs(args)
            batch = to_device(cpu_batch, device)
            rng_before = rng_snapshot(torch, np, device)
            correction_maxima = []
            hook = head.denoiser.register_forward_hook(lambda module, inputs, output:
                                                       correction_maxima.append(float(output.abs().max())))
            try:
                with torch.no_grad():
                    outputs = backbone(batch['x_norm'], batch['physics'])
                    residuals = head.sample(outputs, batch['physics'], K=8, steps=4,
                                            generator=torch.Generator().manual_seed(0))
                    clean = head.predict_clean_residual(torch.zeros_like(outputs['logits_det']),
                        torch.tensor([199], device=device), torch.zeros_like(outputs['logits_det']),
                        outputs['logits_det'])
                    prediction_logits = []
                    backbone_hook = backbone.register_forward_hook(lambda module, inputs, result:
                        prediction_logits.append(result['logits_det'].detach()))
                    try:
                        probability, calibration_logit = raw_probability(backbone, head, batch,
                            dict(seed=0, K=8, steps=4, method='ddim'), return_logits=True)
                    finally:
                        backbone_hook.remove()
                    if len(prediction_logits) != 1:
                        raise RuntimeError('Expected one captured backbone forward in raw_probability')
                    h1_probability = prediction_logits[0].sigmoid()
            finally:
                hook.remove()
            if (not correction_maxima or any(value != 0 for value in correction_maxima)
                    or not bool(torch.isfinite(residuals).all()) or bool(residuals.any()) or bool(clean.any())
                    or not bool(torch.isfinite(probability).all() and torch.isfinite(calibration_logit).all())):
                raise RuntimeError('Initial zero-output prediction mechanism failed')
            difference = float((probability - h1_probability).abs().max())
            if not math.isfinite(difference) or difference > report['probability_absolute_tolerance']:
                raise RuntimeError('Initial P1 probability differs from H1 beyond rounding tolerance')
            compare(rng_before, rng_snapshot(torch, np, device), torch, path='initial_prediction_rng')
            report['initial_predictions'].append(dict(sample_id=batch['sample_id'][0],
                correction_max_abs=max(correction_maxima), clean_residual_max_abs=float(clean.abs().max()),
                final_residual_max_abs=float(residuals.abs().max()), probability_max_abs_difference=difference,
                probability_exactly_equal=bool(torch.equal(probability, h1_probability)),
                probability_reference='sigmoid of the same backbone forward used by raw_probability',
                separate_backbone_forward_probability_max_abs_difference=float(
                    (outputs['logits_det'].sigmoid() - h1_probability).abs().max()),
                calibration_logit_max_abs_difference=float((calibration_logit - prediction_logits[0]).abs().max()),
                global_rng_unchanged=True))
        if model_state_digest(head) != initial_head:
            raise RuntimeError('Initial prediction changed head state')
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        for step, cpu_batch in enumerate(cpu_batches, 1):
            require_no_jobs(args)
            batch = to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            upstream = {name: p for name, p in head.named_parameters() if not name.startswith('denoiser.output.')}
            before_upstream = {name: p.detach().clone() for name, p in upstream.items()} if step == 1 else {}
            with torch.no_grad():
                outputs = backbone(batch['x_norm'], batch['physics'])
            loss = head.training_loss(batch['target'], outputs, batch['physics'], mask=batch['valid_mask'])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss_difference = None
            if step == 1:
                bce = F.binary_cross_entropy_with_logits(outputs['logits_det'], batch['target'], reduction='none')
                bce = (bce * batch['valid_mask']).sum() / batch['valid_mask'].sum()
                loss_difference = float((loss - bce).abs())
                if loss_difference != 0:
                    raise RuntimeError('Initial masked BCE differs from frozen H1-logit BCE')
            loss.backward()
            gradients = verify_gradients(list(head.named_parameters()), step, torch)
            optimizer.step()
            decay = None
            if step == 1:
                changed = 0
                for name, p in upstream.items():
                    expected = before_upstream[name] * (1 - .001 * .01)
                    if not torch.equal(p, expected):
                        raise RuntimeError('First-step upstream update differs from AdamW decay: ' + name)
                    changed += int(not torch.equal(p, before_upstream[name]))
                decay = dict(matches_weight_decay_only=True, changed_parameter_tensors=changed)
            if (model_state_digest(backbone) != frozen_before
                    or any(p.grad is not None or p.requires_grad for p in backbone.parameters())):
                raise RuntimeError('Frozen backbone state or gradients changed')
            report['updates'].append(dict(update=step, sample_id=batch['sample_id'][0], loss=float(loss.detach()),
                initial_h1_bce_max_abs_difference=loss_difference, upstream_first_step_decay=decay,
                frozen_backbone_unchanged=True, **gradients))
            save()
        for path, expected in watched.items():
            if file_sha256(path) != expected:
                raise RuntimeError('Source/data changed during witness: ' + path)
        require_no_jobs(args)
        report.update(status='passed', final_source_data_recheck_passed=True,
            frozen_backbone=dict(before_sha256=frozen_before, after_sha256=model_state_digest(backbone),
                                 unchanged=True, gradients_absent=True),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    finally:
        os.chdir(previous_cwd)


def main(argv=None):
    args = parse_args(argv)
    report = make_plan(args)
    if not args.execute:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if sys.platform != 'linux':
        raise RuntimeError('Execution requires the configured Linux GPU environment')
    require_no_jobs(args)
    output = checked_output(args.output or ROOT / 'profile_output' / ('zero-init-' + uuid.uuid4().hex + '.json'),
                            args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    report['status'] = 'running'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    save = lambda: write_report(output, report)
    try:
        execute(args, report, save)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(dict(status=report['status'], output=str(output))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
