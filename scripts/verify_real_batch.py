"""Opt-in, disposable real-data H1/P1 GPU and cache-equivalence witness.

The default prints a plan without importing torch, opening data, or querying a
GPU. Execution preserves the original portable metadata and complete POWER
source checks, reads only declared Tb/Tr development samples, and compares two
batch-1 updates plus one original prediction with and without the CPU sample
cache. No checkpoint is read or written. This is a kernel/readiness witness,
not an accuracy or speed experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fwt_real_batch_cache_witness_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_batches import active_gpu_processes, memory_snapshot
from benchmark_sample_cache import fingerprint, spread_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--updates', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--blocked-pid', type=int, action='append', default=[])
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--max-memory-fraction', type=float, default=.85)
    parser.add_argument('--gpu-atol', type=float, default=0.)
    parser.add_argument('--gpu-rtol', type=float, default=0.)
    parser.add_argument('--deterministic-gpu', action='store_true',
                        help='Explicit diagnostic control; does not change the training entrypoint')
    parser.add_argument('--synthetic-repeat-diagnostic', action='store_true',
                        help='Two uncached H1 arms on synthetic inputs; no data access or witness pass')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 32 or args.device_index < 0 or not 1 <= args.cache_mib <= 1024:
        parser.error('use 1..32 CPU threads, a nonnegative device, and 1..1024 cache MiB')
    if not 0 <= args.seed < 2**32 or any(pid < 1 for pid in args.blocked_pid):
        parser.error('use a uint32 seed and positive blocked process IDs')
    if not math.isfinite(args.max_memory_fraction) or not 0 < args.max_memory_fraction <= .95:
        parser.error('max-memory-fraction must be in (0, 0.95]')
    if any(not math.isfinite(value) or value < 0 for value in (args.gpu_atol, args.gpu_rtol)):
        parser.error('GPU tolerances must be finite and nonnegative')
    if args.synthetic_repeat_diagnostic and (args.gpu_atol or args.gpu_rtol):
        parser.error('Repeat diagnostics report exact differences; do not specify a passing tolerance')
    return args


def configure_gpu_mode(args, torch):
    """Configure explicit deterministic controls before the first CUDA context."""
    initialized = torch.cuda.is_initialized()
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if args.deterministic_gpu:
        if initialized:
            raise RuntimeError('Deterministic GPU controls require a fresh process before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return dict(requested_deterministic_gpu=args.deterministic_gpu,
        cuda_initialized_before_settings=initialized,
        previous_cublas_workspace_config=previous,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def blocked_process_state(pid):
    """Read Linux process liveness without signalling it or treating Z as live."""
    if not Path('/proc/self/stat').is_file():
        raise RuntimeError('Cannot verify blocked processes: Linux procfs is unavailable')
    try:
        value = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return 'missing'
    except OSError as exc:
        raise RuntimeError(f'Cannot read blocked PID {pid} state') from exc
    try:
        recorded_pid = int(value.split(' ', 1)[0])
        state = value[value.rindex(')') + 2:].split()[0]
        if recorded_pid != pid or state not in {'R', 'S', 'D', 'Z', 'T', 't', 'X', 'x', 'K', 'W', 'P', 'I'}:
            raise ValueError('invalid proc stat')
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f'Cannot parse blocked PID {pid} state') from exc
    return state


def require_no_jobs(args):
    # A queued job may not yet own a CUDA context. Explicit process blockers
    # supplement the compute-process check and fail conservatively on PID reuse.
    for pid in args.blocked_pid:
        state = blocked_process_state(pid)
        if state not in ('missing', 'Z', 'X', 'x'):
            raise RuntimeError(f'Blocked process {pid} is live (state={state}); do not run this witness yet')
    # The shared query already excludes self; keep the rule explicit here so a
    # still-owned CUDA context after H1 cannot block the cached arm or P1. No
    # other process is allowed, including an unrelated process on another GPU.
    active = [row for row in active_gpu_processes() if row['pid'] != os.getpid()]
    if active:
        raise RuntimeError('Other GPU compute processes are active: ' + json.dumps(active))


def checked_output(path, data_root):
    path, data_root = path.resolve(), data_root.resolve()
    if path.suffix != '.json' or path.exists():
        raise ValueError('A new .json report path is required')
    forbidden = [data_root, ROOT / 'runs', ROOT / 'data', ROOT / 'fwt_v2',
                 ROOT / 'scripts', ROOT / 'configs', ROOT / 'tests']
    if any(path == directory or directory in path.parents for directory in forbidden):
        raise ValueError('Reports must stay outside data, training outputs, and source directories')
    if any(parent.name.lower() in ('runs', 'checkpoints') for parent in path.parents):
        raise ValueError('Reports must stay outside training/checkpoint directories')
    return path


def snapshot(value, torch):
    """Private in-memory snapshots only: never serialize model weights."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item, torch) for item in value)
    return value


def compare(expected, actual, torch, *, atol=0., rtol=0., path='root'):
    """Every tensor/scalar participates; zero tolerance means exact equality."""
    count, exact, max_abs = 0, True, 0.
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError('Tensor metadata mismatch at ' + path)
        a, b = expected.detach().cpu(), actual.detach().cpu()
        exact = bool(torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                 b.contiguous().reshape(-1).view(torch.uint8)))
        if a.is_floating_point():
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise AssertionError('Nonfinite witness tensor at ' + path)
            max_abs = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
            if not exact and ((atol == 0 and rtol == 0) or
                              not bool(torch.allclose(a, b, atol=atol, rtol=rtol))):
                raise AssertionError(f'GPU equivalence mismatch at {path}; max_abs={max_abs}')
        elif not exact:
            raise AssertionError('Exact tensor mismatch at ' + path)
        return dict(tensors=1, bitwise_equal=exact, max_absolute_difference=max_abs)
    if type(expected) is not type(actual):
        raise AssertionError('Container/scalar type mismatch at ' + path)
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError('Dictionary key mismatch at ' + path)
        items = [(key, expected[key], actual[key]) for key in expected]
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError('Sequence length mismatch at ' + path)
        items = [(i, a, b) for i, (a, b) in enumerate(zip(expected, actual))]
    else:
        if expected != actual:
            raise AssertionError('Exact scalar mismatch at ' + path)
        return dict(tensors=0, bitwise_equal=True, max_absolute_difference=0.)
    for key, a, b in items:
        result = compare(a, b, torch, atol=atol, rtol=rtol, path=f'{path}.{key}')
        count += result['tensors']
        exact &= result['bitwise_equal']
        max_abs = max(max_abs, result['max_absolute_difference'])
    return dict(tensors=count, bitwise_equal=exact, max_absolute_difference=max_abs)


def rng_snapshot(torch, np, device):
    numpy = np.random.get_state()
    return dict(python=random.getstate(), numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device))


def diagnose_differences(expected, actual, torch, path='root'):
    """Describe every floating difference, never convert a difference to a pass.

    Exact integer/RNG/identity mismatches and nonfinite tensors invalidate the
    repeat control. Floating differences retain their own per-tensor magnitude.
    """
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError('Diagnostic tensor metadata mismatch at ' + path)
        try:
            compare(expected, actual, torch, path=path)
            return dict(tensors=1, differences=[])
        except AssertionError:
            if not expected.is_floating_point():
                raise
        a, b = expected.detach().cpu().double(), actual.detach().cpu().double()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise AssertionError('Nonfinite diagnostic tensor at ' + path)
        delta = a - b
        norm = float(torch.linalg.vector_norm(a))
        difference_norm = float(torch.linalg.vector_norm(delta))
        return dict(tensors=1, differences=[dict(path=path, shape=list(a.shape), dtype=str(expected.dtype),
            max_absolute_difference=float(delta.abs().max()) if a.numel() else 0.,
            differing_values=int(torch.count_nonzero(delta)), reference_l2=norm,
            difference_l2=difference_norm,
            relative_l2=difference_norm / norm if norm else (0. if not difference_norm else None))])
    if type(expected) is not type(actual):
        raise AssertionError('Diagnostic container/scalar type mismatch at ' + path)
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError('Diagnostic dictionary key mismatch at ' + path)
        pairs = [(key, expected[key], actual[key]) for key in expected]
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError('Diagnostic sequence length mismatch at ' + path)
        pairs = [(index, a, b) for index, (a, b) in enumerate(zip(expected, actual))]
    else:
        if expected != actual:
            raise AssertionError('Diagnostic scalar/identity mismatch at ' + path)
        return dict(tensors=0, differences=[])
    result = dict(tensors=0, differences=[])
    for key, a, b in pairs:
        item = diagnose_differences(a, b, torch, path=f'{path}.{key}')
        result['tensors'] += item['tensors']
        result['differences'].extend(item['differences'])
    return result


def run_arm(args, config, data_config, dataset, indices, phase, device, expected=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from fwt_v2.engine import (seed_everything, build_det, to_device, deterministic_loss,
                               gradient_l2_norm, model_state_digest, raw_probability)
    from fwt_v2.models import FWTConfig
    from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig

    require_no_jobs(args)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_config = FWTConfig(**config['model'])
    backbone = build_det(model_config, 'fwt', data_config).to(device)
    head = None
    if phase == 'diff':
        backbone.requires_grad_(False).eval()
        settings = dict(config['diffusion'], backbone_feature_channels=model_config.decoder_widths,
                        relational_channels=model_config.prop_channels)
        head = ConditionalResidualDiffusion(DiffusionConfig(**settings)).to(device)
    model = backbone if head is None else head
    model.train()
    module_types = {name: type(module).__name__ for name, module in model.named_modules()}
    parameter_modules = {name: module_types[name.rsplit('.', 1)[0]]
                         for name, _ in model.named_parameters() if '.' in name}
    initial_digest = model_state_digest(model)
    frozen_digest = model_state_digest(backbone) if head is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0)
    records, private_states = [], dict(updates=[])
    for index, cpu_batch in enumerate(loader):
        require_no_jobs(args)
        before_rng = rng_snapshot(torch, np, device)
        cpu_digest = fingerprint(cpu_batch, torch.Tensor)
        batch = to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = []
        hook = None
        if head is not None:
            hook = head.denoiser.register_forward_hook(
                lambda module, inputs, output: predictions.append(snapshot(output, torch)))
        try:
            if head is None:
                outputs = backbone(batch['x_norm'], batch['physics'])
                loss = deterministic_loss(outputs['logits_det'], batch['target'], batch['valid_mask'], 'focal')
            else:
                with torch.no_grad():
                    outputs = backbone(batch['x_norm'], batch['physics'])
                loss = head.training_loss(batch['target'], outputs, batch['physics'], mask=batch['valid_mask'])
        finally:
            if hook is not None:
                hook.remove()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite real-data loss')
        loss.backward()
        gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise RuntimeError('Missing/nonfinite real-data gradients')
        norm = gradient_l2_norm(list(gradients.values()))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('Real-data gradient norm must be finite and positive')
        optimizer.step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        state = snapshot(dict(backbone_outputs=outputs, denoiser_outputs=predictions,
            loss=loss, gradients=gradients, parameters_and_buffers=model.state_dict(),
            optimizer=optimizer.state_dict(), rng_before=before_rng,
            rng_after=rng_snapshot(torch, np, device), sample_ids=list(batch['sample_id']),
            diffusion=None if head is None else head.last_training_diagnostics), torch)
        # RNG, sample order and collated input bytes are exact even when an
        # explicitly requested tolerance is used for floating GPU results.
        equality = None
        if expected is not None:
            if cpu_digest != expected['updates'][index]['input_sha256']:
                raise AssertionError('Collated input bytes or sample order changed')
            for name in ('rng_before', 'rng_after', 'sample_ids', 'diffusion'):
                compare(expected['updates'][index]['state'][name], state[name], torch, path=name)
            equality = compare(expected['updates'][index]['state'], state, torch,
                               atol=args.gpu_atol, rtol=args.gpu_rtol)
        else:
            private_states['updates'].append(dict(input_sha256=cpu_digest, state=state))
        records.append(dict(update=index + 1, sample_ids=list(batch['sample_id']),
            input_sha256=cpu_digest, loss=float(loss.detach()), gradient_l2=norm,
            gradients_finite=True, gradient_tensors=len(gradients),
            step_seconds=seconds, equivalence=equality))
        del state, outputs, loss, gradients, batch, cpu_batch, predictions
    # Exercise the actual prediction path on one of the already selected Tb/Tr
    # samples. No validation/test dataset is constructed or array is opened.
    require_no_jobs(args)
    prediction_model_digest = model_state_digest(model)
    backbone.eval()
    if head is not None:
        head.eval()
    sampling = dict(seed=args.seed, K=8, steps=4, method='ddim')
    cpu_batch = default_collate([dataset[indices[0]]])
    prediction_input_digest = fingerprint(cpu_batch, torch.Tensor)
    batch = to_device(cpu_batch, device)
    rng_before_prediction = rng_snapshot(torch, np, device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        probability, logits = raw_probability(backbone, head, batch, sampling, return_logits=True)
    torch.cuda.synchronize(device)
    prediction_seconds = time.perf_counter() - started
    if (not bool(torch.isfinite(probability).all() and torch.isfinite(logits).all())
            or not bool(((probability >= 0) & (probability <= 1)).all())):
        raise RuntimeError('Nonfinite or out-of-range prediction from real input')
    prediction_state = snapshot(dict(probability=probability, logits=logits,
        rng_before=rng_before_prediction, rng_after=rng_snapshot(torch, np, device),
        sample_ids=list(batch['sample_id'])), torch)
    # raw_probability uses deterministic eval plus a per-sample CPU generator;
    # it must not advance the global training RNGs or mutate learned state.
    compare(prediction_state['rng_before'], prediction_state['rng_after'], torch, path='prediction_rng')
    if model_state_digest(model) != prediction_model_digest:
        raise RuntimeError('Prediction mutated disposable model parameters or buffers')
    prediction_equality = None
    if expected is not None:
        reference = expected['prediction']
        if reference['input_sha256'] != prediction_input_digest:
            raise AssertionError('Prediction input bytes changed between cache arms')
        for name in ('rng_before', 'rng_after', 'sample_ids'):
            compare(reference['state'][name], prediction_state[name], torch, path='prediction.' + name)
        prediction_equality = compare(reference['state'], prediction_state, torch,
                                      atol=args.gpu_atol, rtol=args.gpu_rtol, path='prediction')
    else:
        private_states['prediction'] = dict(input_sha256=prediction_input_digest, state=prediction_state)
    prediction = dict(sample_ids=list(batch['sample_id']), input_sha256=prediction_input_digest,
        finite=True, probability_min=float(probability.min()), probability_max=float(probability.max()),
        probability_sha256=fingerprint(prediction_state['probability'], torch.Tensor),
        logits_sha256=fingerprint(prediction_state['logits'], torch.Tensor),
        sampling=sampling if head is not None else None,
        seed_strategy='sha256_base_seed_sample_id_v1' if head is not None else None,
        global_rng_unchanged=True, model_state_unchanged=True,
        seconds=prediction_seconds, equivalence=prediction_equality)
    del probability, logits, prediction_state, batch, cpu_batch
    changed = initial_digest != model_state_digest(model)
    if not changed:
        raise RuntimeError('AdamW failed to change the disposable model')
    frozen = None
    if head is not None:
        frozen = dict(unchanged=frozen_digest == model_state_digest(backbone),
                      gradients_absent=all(p.grad is None for p in backbone.parameters()),
                      requires_grad_false=all(not p.requires_grad for p in backbone.parameters()))
        if not all(frozen.values()):
            raise RuntimeError('P1 random backbone was not frozen')
    result = dict(updates=records, prediction=prediction, adamw_changed_model=changed,
                  parameter_modules=parameter_modules,
                  initial_model_sha256=initial_digest, final_model_sha256=model_state_digest(model),
                  frozen_backbone=frozen, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device), cpu_memory=memory_snapshot(),
                  gpu_memory_scope='initialization, disposable updates, and one original raw_probability prediction')
    del optimizer, loader, model, backbone, head
    gc.collect()
    torch.cuda.empty_cache()
    return result, private_states


def execute_synthetic_diagnostic(args, report, save):
    """Reuse the complete H1 update path, with synthetic inputs and no cache."""
    if sys.platform != 'linux':
        raise RuntimeError('Run the GPU diagnostic in the configured Linux environment')
    require_no_jobs(args)
    import torch
    from benchmark_batches import synthetic_batch
    from fwt_v2.data import DataConfig
    from fwt_v2.portable import read_json, file_sha256

    controls = configure_gpu_mode(args, torch)  # Before any CUDA availability/device call.
    torch.set_num_threads(args.cpu_threads)
    config_path = ROOT / 'configs/h1.json'
    config = read_json(config_path)
    report['provenance'] = dict(config_sha256=file_sha256(config_path), config=config,
        script_sha256=file_sha256(__file__),
        sources={p.name: file_sha256(p) for p in sorted((ROOT / 'fwt_v2').glob('*.py'))},
        helper_sha256={name: file_sha256(ROOT / 'scripts' / name) for name in
                       ('benchmark_batches.py', 'benchmark_sample_cache.py')})
    report['environment'] = dict(torch=torch.__version__, python=sys.version,
        cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(), gpu_controls=controls)
    save()
    if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
        raise RuntimeError('Configured CUDA device unavailable')
    device = torch.device(f'cuda:{args.device_index}')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device)
    report['environment']['gpu_name'] = torch.cuda.get_device_name(device)
    dataset = []
    for index in range(args.updates):
        batch = synthetic_batch(torch, 1, 256, 256,
            18 + config['model']['extra_weather_channels'], args.seed + index, 'cpu')
        sample = {name: value[0] for name, value in batch.items() if name != 'physics'}
        sample['physics'] = {name: value[0] for name, value in batch['physics'].items()}
        sample['sample_id'] = f'SYNTHETIC-REPEAT-{index}'
        dataset.append(sample)
    del batch, sample
    phase = dict(phase='det', role='synthetic_no_role', arms={}, status='running')
    report['phases'].append(phase)
    indices = list(range(args.updates))
    phase['arms']['uncached_a'], reference = run_arm(
        args, config, DataConfig(**config['data']), dataset, indices, 'det', device)
    save()
    # No expected-state argument: collect the whole repeat before measuring all
    # differences, rather than stopping at the first nondeterministic gradient.
    phase['arms']['uncached_b'], repeated = run_arm(
        args, config, DataConfig(**config['data']), dataset, indices, 'det', device)
    if phase['arms']['uncached_a']['initial_model_sha256'] != phase['arms']['uncached_b']['initial_model_sha256']:
        raise AssertionError('Synthetic repeat initial models differ')
    comparison = diagnose_differences(reference, repeated, torch)
    owners = phase['arms']['uncached_a']['parameter_modules']
    for difference in comparison['differences']:
        if '.gradients.' in difference['path']:
            parameter = difference['path'].split('.gradients.', 1)[1]
            difference['parameter_module_type'] = owners.get(parameter)
    comparison['different_tensors'] = len(comparison['differences'])
    comparison['repeat_bitwise_equal'] = not comparison['differences']
    comparison['interpretation'] = ('No tensor difference observed in this bounded repeat.'
        if comparison['repeat_bitwise_equal'] else
        'Differences occurred without any cache; this is native execution repeatability evidence, not a cache equivalence pass.')
    comparison['attribution_limit'] = ('A differing parameter identifies an affected module, not the originating kernel. '
        'upsamples.*.up.1 is Conv2d; the preceding F.interpolate uses bilinear align_corners=True.')
    phase.update(status='diagnostic_completed', comparison=comparison)
    report.update(status='diagnostic_completed', witness_passed=False,
                  cpu_memory_after=memory_snapshot())


def execute(args, report, save):
    if sys.platform != 'linux':
        raise RuntimeError('Execute this witness in the configured Linux GPU environment')
    require_no_jobs(args)  # Before torch import or expensive data verification.
    import torch
    controls = configure_gpu_mode(args, torch)
    import train as portable_train
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.portable import read_json, file_sha256
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset

    torch.set_num_threads(args.cpu_threads)
    root = args.data_root.resolve()
    # The original formal metadata preflight includes manifest/normalizer/index
    # binding and all relocated immutable weather-source hashes. No execute call.
    check_args = portable_train.parser().parse_args(['--phase', 'det', '--data-root', str(root),
        '--run-dir', str(ROOT / 'profile_output' / ('unused-' + uuid.uuid4().hex)),
        '--epochs', '30', '--device', f'cuda:{args.device_index}', '--cpu-threads', str(args.cpu_threads)])
    started = time.perf_counter()
    checked = portable_train.preflight(check_args)
    config = read_json(checked['config_path'])
    report['provenance'] = dict(binding=checked['binding'],
        sources={p.name: file_sha256(p) for p in sorted((ROOT / 'fwt_v2').glob('*.py'))},
        script_sha256=file_sha256(__file__), entry_sha256=file_sha256(ROOT / 'train.py'),
        helper_sha256={name: file_sha256(ROOT / 'scripts' / name) for name in
                       ('benchmark_batches.py', 'benchmark_sample_cache.py')},
        config=config, role_counts=checked['role_counts'])
    report['metadata_preflight_seconds'] = time.perf_counter() - started
    report['environment'] = dict(torch=torch.__version__, python=sys.version,
        cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(), gpu_controls=controls)
    save()
    normalizer = NormalizationStats.load(checked['normalizer'])
    data_config = DataConfig(**config['data'])
    indexed = {(row['sample_id'], row['field']): row for row in
               read_json(root / read_json(root / 'bundle.json')['files']['arrays_index']['path'])['files']}
    device = torch.device(f'cuda:{args.device_index}')
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)  # Original relative paths and weather/source verification.
        for phase, role in [('det', 'backbone_train'), ('diff', 'residual_train')]:
            require_no_jobs(args)
            started = time.perf_counter()
            dataset = RCDADataset(checked['manifest'], role, normalizer, data_config)
            result = dict(phase=phase, role=role,
                full_dataset_and_weather_verification_seconds=time.perf_counter() - started,
                selected_samples=[], arms={})
            report['phases'].append(result)
            save()
            indices = spread_indices(len(dataset), args.updates)
            cache = SharedTensorCache(args.cache_mib * 1024**2)
            cached = CachedDataset(dataset, cache)
            for index in indices:
                row, hashes = dataset.samples[index], {}
                for field in ('input_path', 'label_path'):
                    bound = indexed[(row['sample_id'], field)]
                    actual = file_sha256(row[field])
                    if bound['role'] != role or actual != bound['sha256']:
                        raise ValueError('Selected development array differs from its bound bytes')
                    hashes[field] = actual
                plain, cold, hot = dataset[index], cached[index], cached[index]
                digest = fingerprint(plain, torch.Tensor)
                if any(fingerprint(sample, torch.Tensor) != digest for sample in (cold, hot)):
                    raise AssertionError('Original/cold-cache/hot-cache sample bytes differ')
                result['selected_samples'].append(dict(index=index, sample_id=row['sample_id'],
                    arrays_sha256=hashes, sample_sha256=digest, cache_bitwise_equal=True))
                del plain, cold, hot
            if cache.stats()['entries'] != len(indices) or cache.stats()['oversize_skips']:
                raise RuntimeError('Cache budget cannot retain the selected witness samples')
            require_no_jobs(args)  # Verification may take minutes; recheck before CUDA.
            if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
                raise RuntimeError('Configured CUDA device unavailable')
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device)
            report['environment'].update(gpu_name=torch.cuda.get_device_name(device),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
            result['arms']['uncached'], states = run_arm(args, config, data_config, dataset, indices, phase, device)
            save()
            result['arms']['cached'], _ = run_arm(args, config, data_config, cached, indices, phase, device, states)
            if result['arms']['uncached']['initial_model_sha256'] != result['arms']['cached']['initial_model_sha256']:
                raise AssertionError('Identical seeds did not produce identical initial models')
            result['cache'] = cache.stats()
            result['status'] = 'passed'
            del states, cached, dataset
            cache.clear()
            del cache
            gc.collect()
            save()
        # Detect source/metadata changes during the long verification interval.
        if portable_train.source_digest() != checked['binding']['source_sha256']:
            raise RuntimeError('Source changed during witness')
        if file_sha256(ROOT / 'train.py') != report['provenance']['entry_sha256']:
            raise RuntimeError('Portable entrypoint changed during witness')
        if file_sha256(root / 'bundle.json') != checked['binding']['bundle_sha256']:
            raise RuntimeError('Data bundle changed during witness')
        report.update(status='passed', cpu_memory_after=memory_snapshot())
    finally:
        os.chdir(previous_cwd)


def main(argv=None):
    args = parse_args(argv)
    report = dict(schema=SCHEMA, status='plan_only', created_at=datetime.now(timezone.utc).isoformat(),
        purpose='real-data disposable kernel and cache-equivalence witness; not scientific results',
        data_root=str(args.data_root.resolve()), batch_size=1, updates_per_arm=args.updates,
        phases=[], seed=args.seed, cache_mib=args.cache_mib, blocked_pids=args.blocked_pid,
        checkpoint_reads=0, checkpoint_writes=0, training_outputs_created=0,
        cpu_sample_comparison='bitwise structural SHA256, including every tensor and metadata scalar',
        gpu_tolerance=dict(atol=args.gpu_atol, rtol=args.gpu_rtol),
        deterministic_gpu_requested=args.deterministic_gpu,
        precision='FP32; explicit deterministic controls' if args.deterministic_gpu else 'FP32; inherited PyTorch defaults',
        protocol='first/last declared Tb and Tr samples; identical fixed order, RNG, and AdamW lr=0.001',
        prediction_protocol='first selected Tb/Tr sample; original raw_probability; P1 K=8, steps=4, DDIM and sample-ID seed',
        limitations=['Random temporary weights do not measure training quality or checkpoint compatibility.',
            'Batch-1 cache equivalence does not imply equivalence at a different batch size.',
            'At most two selected development samples per phase do not establish dataset-wide equivalence.',
            'Prediction checks reuse selected Tb/Tr samples; no validation/test arrays, selection, resume, or checkpoints.',
            'Step timings include diagnostic hooks; do not interpret as throughput benchmark.',
            'GPU allocator peaks exclude driver/context memory; CPU RSS peak is process cumulative.'])
    if args.synthetic_repeat_diagnostic:
        report.update(schema='fwt_synthetic_repeat_diagnostic_v1',
            purpose='native GPU repeatability diagnostic; no cache equivalence or scientific result',
            data_root=None, data_reads=0, cache_enabled=False, cache_mib=0,
            protocol='two fresh H1 models, same seed, same synthetic batch-1 inputs and update sequence',
            witness_passed=False,
            gpu_tolerance=None,
            cpu_sample_comparison='same synthetic input bytes; no RCDADataset or source verification required',
            prediction_protocol='one synthetic sample through original H1 raw_probability after each arm',
            limitations=['Synthetic-only H1 repeat; no real data, P1, cache, validation, or training-quality evidence.',
                'A parameter-gradient difference does not identify the originating kernel.',
                'Deterministic GPU controls affect this disposable diagnostic only; unsupported operators fail explicitly.',
                'diagnostic_completed means observations collected, never a passing cache-equivalence witness.',
                'GPU allocator peaks exclude driver/context memory.'])
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    require_no_jobs(args)
    default = ROOT / 'profile_output' / ('real-batch-witness-' + uuid.uuid4().hex[:12] + '.json')
    output = checked_output(args.output or default, args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.update(status='running', cpu_memory_before=memory_snapshot())
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    def save():
        temporary = output.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(output)
        print(json.dumps(dict(status=report['status'], output=str(output),
                             phases=[dict(phase=p['phase'], status=p.get('status', 'running'))
                                     for p in report['phases']])), flush=True)

    try:
        action = execute_synthetic_diagnostic if args.synthetic_repeat_diagnostic else execute
        action(args, report, save)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
