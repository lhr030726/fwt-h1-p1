"""Bounded synthetic throughput probe of the configured H1/P1 models.

Default is a CPU-only plan. --execute requires an otherwise idle GPU. Every
phase/batch pair runs in a fresh subprocess and discards its randomly initialized
weights. No dataset, checkpoint, training output or held-out sample is opened.
This measures compute capacity, not learning quality or end-to-end data loading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fwt_synthetic_batch_probe_v1'


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def batch_list(value):
    try:
        result = [int(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use comma-separated positive integers') from exc
    if not result or any(n < 1 for n in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('batch sizes must be positive and distinct')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/h1.json')
    parser.add_argument('--phases', nargs='+', choices=['det', 'diff'], default=['det', 'diff'])
    parser.add_argument('--batches', type=batch_list, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--height', type=positive_int, default=256)
    parser.add_argument('--width', type=positive_int, default=256)
    parser.add_argument('--warmup', type=positive_int, default=2)
    parser.add_argument('--iterations', type=positive_int, default=5)
    parser.add_argument('--validation-iterations', type=positive_int, default=1)
    parser.add_argument('--cpu-threads', type=positive_int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--max-memory-fraction', type=float, default=.85)
    parser.add_argument('--trial-timeout', type=positive_int, default=180)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'profile_output')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--_trial', choices=['det', 'diff'], help=argparse.SUPPRESS)
    parser.add_argument('--_batch', type=positive_int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.height, args.width) < 8:
        parser.error('four pyramid levels require height/width >= 8')
    if not math.isfinite(args.max_memory_fraction) or not 0 < args.max_memory_fraction <= .95:
        parser.error('max-memory-fraction must be in (0, 0.95]')
    if args.device_index < 0:
        parser.error('device-index must be nonnegative')
    if args._trial and (not args.execute or args._batch is None):
        parser.error('internal trials require --execute and --_batch')
    return args


def active_gpu_processes(run=subprocess.run):
    """Fail closed if process accounting is unavailable; do not compete with jobs."""
    result = run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                  '--format=csv,noheader,nounits'], capture_output=True, text=True,
                 check=False, timeout=15)
    if result.returncode:
        raise RuntimeError('Cannot verify GPU idleness: ' + result.stderr.strip())
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, separator, name = line.partition(',')
        if not separator or not pid.strip().isdigit():
            raise RuntimeError('Unrecognized nvidia-smi process accounting: ' + line)
        if int(pid) != os.getpid():
            rows.append({'pid': int(pid), 'name': name.strip()})
    return rows


def require_idle_gpu():
    active = active_gpu_processes()
    if active:
        raise RuntimeError('GPU compute processes are active; retry after they end: '
                           + json.dumps(active, ensure_ascii=False))


def memory_snapshot():
    """Linux process RSS/HWM and host/cgroup-aware available memory, in bytes."""
    result = {}
    for path, wanted in [('/proc/self/status', {'VmRSS': 'rss_bytes', 'VmHWM': 'peak_rss_bytes'}),
                         ('/proc/meminfo', {'MemTotal': 'host_total_bytes',
                                            'MemAvailable': 'host_available_bytes'})]:
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(':')
                if key in wanted:
                    result[wanted[key]] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
    for path, key in [('/sys/fs/cgroup/memory.current', 'cgroup_current_bytes'),
                      ('/sys/fs/cgroup/memory.max', 'cgroup_limit_bytes'),
                      ('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'cgroup_current_bytes'),
                      ('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'cgroup_limit_bytes')]:
        if key in result:
            continue
        try:
            value = Path(path).read_text().strip()
            result[key] = None if value == 'max' else int(value)
        except (OSError, ValueError):
            pass
    return result


def summarize_timings(seconds, batch_size):
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError('positive finite synchronized timing samples required')
    total = sum(seconds)
    return dict(iterations=len(seconds), seconds=seconds,
                mean_seconds=statistics.mean(seconds), median_seconds=statistics.median(seconds),
                min_seconds=min(seconds), max_seconds=max(seconds),
                samples_per_second=batch_size * len(seconds) / total)


def synthetic_batch(torch, size, height, width, channels, seed, device):
    """Independent synthetic examples; values obey model interfaces, not geography."""
    generator = torch.Generator(device='cpu').manual_seed(seed + 104729)
    shape = (size, 1, height, width)
    x = torch.randn((size, channels, height, width), generator=generator)
    valid = torch.rand(shape, generator=generator) > .15
    valid[:, :, 0, 0] = True
    physics = dict(distance=torch.rand(shape, generator=generator) * 2 - 1,
                   wind_xy=torch.zeros(size, 2, height, width),
                   slope=torch.rand(shape, generator=generator),
                   upslope_xy=torch.zeros(size, 2, height, width), domain_mask=valid)
    # Upslope unit direction and wind are not used to select real measurements.
    physics['upslope_xy'][:, 0] = 1
    target = ((torch.rand(shape, generator=generator) < .05) & valid).float()
    return dict(x_norm=x.to(device), target=target.to(device), valid_mask=valid.to(device),
                physics={name: value.to(device) for name, value in physics.items()},
                sample_id=[f'SYNTHETIC-PROBE-{index}' for index in range(size)])


def trial(args):
    # Check before importing torch/creating a CUDA context, including direct trials.
    require_idle_gpu()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import random
    import torch
    from fwt_v2.models import FWTConfig, FWTDet
    from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig
    from fwt_v2.engine import deterministic_loss, gradient_l2_norm, model_state_digest, raw_probability

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(f'cuda:{args.device_index}')
    if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
        raise RuntimeError('Requested CUDA device is unavailable')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device)
    # Independent child processes have identical initial seeds; input generation
    # uses a separate generator. Batch sizes change SGD/RNG consumption semantics.
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_config = FWTConfig(**config['model'])
    backbone = FWTDet(model_config).to(device)
    head = None
    frozen_before = None
    if args._trial == 'diff':
        backbone.requires_grad_(False).eval()
        settings = dict(config['diffusion'], backbone_feature_channels=model_config.decoder_widths,
                        relational_channels=model_config.prop_channels)
        head = ConditionalResidualDiffusion(DiffusionConfig(**settings)).to(device).train()
        frozen_before = model_state_digest(backbone)
    model = backbone if head is None else head
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    batch = synthetic_batch(torch, args._batch, args.height, args.width,
                            18 + model_config.extra_weather_channels, args.seed, device)
    free, total = torch.cuda.mem_get_info(device)
    metadata = dict(schema=SCHEMA, phase=args._trial, batch_size=args._batch,
                    synthetic_only=True, checkpoint_reads=0, checkpoint_writes=0,
                    device=str(device), gpu_name=torch.cuda.get_device_name(device),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    input_shape=list(batch['x_norm'].shape), cpu_threads=torch.get_num_threads(),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    backbone_parameters=sum(p.numel() for p in backbone.parameters()),
                    gpu_total_bytes=total, gpu_free_bytes_before_steps=free,
                    memory_fraction_limit=args.max_memory_fraction,
                    precision='fp32, inherited PyTorch default TF32 controls',
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    timing_scope='GPU-resident tensors; forward, loss, backward, original finite/gradient checks, AdamW step',
                    optimizer='AdamW lr=0.001; default optimizer settings; disposable random model',
                    cpu_memory_before=memory_snapshot(),
                    diffusion_config=None if head is None else settings)

    def update():
        optimizer.zero_grad(set_to_none=True)
        if head is None:
            logits = backbone(batch['x_norm'], batch['physics'])['logits_det']
            loss = deterministic_loss(logits, batch['target'], batch['valid_mask'], 'focal')
        else:
            with torch.no_grad():
                outputs = backbone(batch['x_norm'], batch['physics'])
            loss = head.training_loss(batch['target'], outputs, batch['physics'], mask=batch['valid_mask'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite synthetic loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Missing or nonfinite synthetic gradients')
        norm = gradient_l2_norm(gradients)
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('Invalid synthetic gradient norm')
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norm=norm)

    def measured(action, count):
        samples, values = [], []
        for _ in range(count):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            value = action()
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
            values.append(value)
        return summarize_timings(samples, args._batch), values

    for _ in range(args.warmup):
        update()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    metadata['training'], metadata['step_checks'] = measured(update, args.iterations)
    metadata['training'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                allocated_bytes_after=torch.cuda.memory_allocated(device))
    metadata['disposable_optimizer_updates'] = args.warmup + args.iterations
    backbone.eval()
    if head is not None:
        head.eval()
    sampling = dict(seed=args.seed, K=8, steps=4, method='ddim')

    @torch.no_grad()
    def validate():
        probability = raw_probability(backbone, head, batch, sampling)
        if not bool(torch.isfinite(probability).all()):
            raise RuntimeError('Nonfinite synthetic validation predictions')
        return dict(probability_min=float(probability.min()), probability_max=float(probability.max()))

    validate()  # Separate validation kernel warmup.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    metadata['validation'], metadata['validation_checks'] = measured(validate, args.validation_iterations)
    metadata['validation'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                  sampling=sampling if head is not None else None,
                                  scope='original raw_probability; P1 samples one item at a time with K=8')
    if head is not None:
        metadata['frozen_backbone_unchanged'] = frozen_before == model_state_digest(backbone)
        metadata['frozen_backbone_gradients_absent'] = all(p.grad is None for p in backbone.parameters())
        if not metadata['frozen_backbone_unchanged'] or not metadata['frozen_backbone_gradients_absent']:
            raise RuntimeError('Probe changed its frozen random backbone')
    metadata['cpu_memory_after'] = memory_snapshot()
    metadata['status'] = 'passed'
    return metadata


def child_command(args, phase, batch):
    return [sys.executable, str(Path(__file__).resolve()), '--execute', '--_trial', phase,
            '--_batch', str(batch), '--config', str(args.config.resolve()),
            '--height', str(args.height), '--width', str(args.width),
            '--warmup', str(args.warmup), '--iterations', str(args.iterations),
            '--validation-iterations', str(args.validation_iterations),
            '--cpu-threads', str(args.cpu_threads), '--seed', str(args.seed),
            '--device-index', str(args.device_index),
            '--max-memory-fraction', str(args.max_memory_fraction)]


def main(argv=None):
    args = parse_args(argv)
    if args._trial:
        try:
            result = trial(args)
        except Exception as exc:
            is_oom = isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()
            result = dict(schema=SCHEMA, phase=args._trial, batch_size=args._batch,
                          status='oom' if is_oom else 'failed', error_type=type(exc).__name__, error=str(exc))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] == 'passed' else 1
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode('utf-8-sig'))
    plan = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
                purpose='synthetic capacity/throughput only; not training evidence',
                config_sha256=hashlib.sha256(config_bytes).hexdigest(), config=config,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(b''.join(p.name.encode() + p.read_bytes()
                    for p in sorted((ROOT / 'fwt_v2').glob('*.py')))).hexdigest(),
                phases=args.phases, batches=args.batches, height=args.height, width=args.width,
                warmup=args.warmup, measured_iterations=args.iterations,
                seed=args.seed, rng_scope='fresh subprocess and identical seed per trial; no equivalence claim across batch sizes',
                limitations=['No data loading, host-to-device transfers, weather verification or epoch metrics.',
                             'Random initialized weights and synthetic interface-valid tensors; not scientific results.',
                             'Changing batch size changes update count, pixel weighting and RNG consumption.',
                             'P1 validation is sequential per sample with K=8 and four DDIM steps.',
                             'Allocator peaks exclude CUDA context/driver and other non-PyTorch GPU allocations.'])
    if not args.execute:
        plan['status'] = 'plan_only'
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    require_idle_gpu()
    output_dir = args.output_dir.resolve()
    for forbidden in [ROOT / 'data', ROOT / 'runs']:
        if output_dir == forbidden or forbidden in output_dir.parents:
            raise ValueError('Probe reports must stay outside dataset and training run directories')
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = 'synthetic-batch-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = output_dir / (filename + '-' + uuid.uuid4().hex[:8] + '.json')
    plan['results'] = []
    plan['status'] = 'running'

    def save():
        temporary = output.with_suffix('.json.partial')
        temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(output)

    save()
    for phase in args.phases:
        for batch in args.batches:
            try:
                require_idle_gpu()
                completed = subprocess.run(child_command(args, phase, batch), capture_output=True,
                                           text=True, check=False, timeout=args.trial_timeout)
                result = json.loads(completed.stdout.strip())
                if completed.returncode and result.get('status') == 'passed':
                    raise RuntimeError('Child status and exit code disagree')
                if completed.stderr.strip():
                    result['stderr_tail'] = completed.stderr[-4000:]
            except subprocess.TimeoutExpired:
                result = dict(phase=phase, batch_size=batch, status='timeout',
                              timeout_seconds=args.trial_timeout)
            except Exception as exc:
                result = dict(phase=phase, batch_size=batch, status='failed',
                              error_type=type(exc).__name__, error=str(exc))
            plan['results'].append(result)
            save()
            print(json.dumps(dict(phase=phase, batch_size=batch, status=result['status'],
                                  training=result.get('training'), output=str(output)), ensure_ascii=False), flush=True)
            if result['status'] != 'passed':
                # Do not continue larger batches after OOM, timeout or an unknown error.
                break
    plan['status'] = 'completed' if all(r['status'] == 'passed' for r in plan['results']) else 'incomplete'
    save()
    print(json.dumps(dict(status=plan['status'], output=str(output))), flush=True)
    return 0 if plan['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
