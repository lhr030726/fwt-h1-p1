"""Opt-in, bounded real Tb/Tr warm-cache throughput probe; never formal training.

The default only prints a plan. --execute requires an idle Linux GPU host and
the complete original data/source verification. Random disposable models, no
checkpoint IO, no validation/test arrays, and no change to training source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_batches import active_gpu_processes, memory_snapshot
from benchmark_sample_cache import fingerprint, spread_indices
from verify_real_batch import blocked_process_state, checked_output, configure_gpu_mode

SCHEMA = 'fwt_real_batch_throughput_v1'
POLL_SECONDS = 2.
ROLES = dict(det='backbone_train', diff='residual_train')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phases', nargs='+', choices=ROLES, default=['det', 'diff'])
    parser.add_argument('--batches', nargs='+', type=int, choices=(1, 16, 32), default=[1, 16, 32])
    parser.add_argument('--samples', type=int, choices=(32, 64, 96), default=96)
    parser.add_argument('--passes', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--blocked-pid', type=int, action='append', default=[])
    parser.add_argument('--cache-mib', type=int, default=1024)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.phases)) != len(args.phases) or len(set(args.batches)) != len(args.batches):
        parser.error('phases and batches must be distinct')
    if not 1 <= args.cpu_threads <= 32 or args.device_index < 0 or not 1 <= args.cache_mib <= 2048:
        parser.error('use 1..32 CPU threads, a nonnegative device, and 1..2048 cache MiB')
    if not 0 <= args.seed < 2**32 or any(pid < 1 for pid in args.blocked_pid):
        parser.error('use a uint32 seed and positive blocked process IDs')
    return args


def make_plan(args):
    return dict(schema=SCHEMA, status='plan_only', created_at=datetime.now(timezone.utc).isoformat(),
        purpose='real-data warm-cache training throughput only; random temporary weights',
        data_root=str(args.data_root.resolve()), requested_phases=args.phases,
        seed=args.seed, samples_per_phase=args.samples, cache_mib=args.cache_mib,
        num_workers=0, pin_memory=False, warmup_passes=1, timed_passes=args.passes,
        configurations=[dict(batch_size=b, warmup_updates=args.samples // b,
            timed_updates=args.samples * args.passes // b,
            timed_sample_exposures=args.samples * args.passes) for b in args.batches],
        precision='FP32, inherited GPU controls, no AMP or deterministic override',
        gpu_allocator_memory_fraction=.85, checkpoint_reads=0, checkpoint_writes=0,
        sample_order='uniformly spread original role indices; identical fixed order every pass/configuration',
        timing_boundary='CUDA synchronize before timer; timer starts before DataLoader iteration; '
            'ends after final CUDA synchronize. Includes warm CPU cache clones, default collate, '
            'H2D, original forward/loss/backward, finite checks, gradient norm, AdamW, '
            'loss/norm lists, and P1 deepcopy/step metadata. Excludes validation, checkpoint/report IO, '
            'first-batch report IO, model initialization, source checks, cache fill and summary reduction.',
        resource_observer=dict(interval_seconds=POLL_SECONDS,
            method='background read-only procfs/cgroup and nvidia-smi; checked at batch boundaries',
            overhead='Observation overhead is included, not subtracted. Poll elapsed seconds are '
                'reported by completion time; overlapping polls may straddle timing boundaries. '
                'Sampled RSS/cgroup maxima can miss transient peaks; VmHWM is process cumulative.'),
        limitations=['Batch size changes update count, weight trajectory, dropout and diffusion RNG draws.',
            'No equivalence, convergence, quality, confidence interval or full-epoch speed claim.',
            'Only selected Tb/Tr arrays; complete metadata/weather checks may also reference V metadata.',
            'Random P1 backbone is frozen but does not represent a trained H1 checkpoint.',
            'Configuration order, GPU clocks and allocator high-water marks may affect timings.',
            'GPU allocated/reserved peaks exclude CUDA context/driver and other-process memory.',
            'Polling cannot guarantee exclusivity between observations; contention/error aborts at next check.'],
        phases=[])


def training_command(argv):
    """Recognize known training/probe entrypoints, including the formal queue."""
    if any(Path(arg).name in ('fwt_cache_formal_pipeline.sh', 'launch_fwt_cache_formal.sh',
                              'fwt_pipeline.sh', 'fwt_wait_and_probe_gpu.sh') for arg in argv):
        return True
    if not argv or not Path(argv[0]).name.lower().startswith(('python', 'pypy')):
        return False
    names = {Path(arg).name for arg in argv[1:]}
    if names & {'train.py', 'benchmark_batches.py', 'benchmark_real_batches.py',
                'benchmark_sample_cache.py', 'verify_real_batch.py'}:
        return True
    return 'train' in argv and any(arg == 'fwt_v2' or arg.startswith('fwt_v2.') for arg in argv)


def active_training_processes(proc_root=Path('/proc')):
    if not (proc_root / 'self' / 'stat').is_file():
        raise RuntimeError('Linux procfs is required to verify training-process idleness')
    active = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if training_command(argv):
                value = directory.joinpath('stat').read_text()
                state = value[value.rindex(')') + 2:].split()[0]
                if state not in ('Z', 'X', 'x'):
                    active.append(dict(pid=int(directory.name), state=state, command=argv))
        except FileNotFoundError:
            continue  # Process disappeared during a read-only observation.
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError('Cannot verify process state: ' + directory.name) from exc
    return active


def require_no_jobs(args):
    for pid in args.blocked_pid:
        state = blocked_process_state(pid)
        if state not in ('missing', 'Z', 'X', 'x'):
            raise RuntimeError(f'Blocked process {pid} is live (state={state})')
    active = active_training_processes()
    if active:
        raise RuntimeError('Other training/probe processes are active: ' + json.dumps(active))
    active = [row for row in active_gpu_processes() if row['pid'] != os.getpid()]
    if active:
        raise RuntimeError('Other GPU compute processes are active: ' + json.dumps(active))


class Observer:
    """No CUDA calls or signalling; errors propagate at the next main-thread check."""
    def __init__(self, args):
        self.args, self.error = args, None
        self.stop = threading.Event()
        self.records = []
        self.thread = threading.Thread(target=self._run, name='resource-observer', daemon=True)

    def poll(self):
        started = time.perf_counter()
        try:
            require_no_jobs(self.args)
            memory = memory_snapshot()
            self.records.append(dict(at=time.perf_counter(),
                poll_seconds=time.perf_counter() - started, memory=memory))
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def _run(self):
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(POLL_SECONDS)

    def check(self):
        if self.error is not None:
            raise RuntimeError('Resource/process observation failed; aborting: ' + str(self.error)) from self.error

    def summary(self, start=0., end=float('inf')):
        records = [r for r in list(self.records) if start <= r['at'] <= end]
        keys = ('rss_bytes', 'peak_rss_bytes', 'cgroup_current_bytes')
        return dict(polls=len(records), poll_elapsed_seconds=sum(r['poll_seconds'] for r in records),
            sampled_maximum={key: max((r['memory'][key] for r in records
                if isinstance(r['memory'].get(key), int)), default=None) for key in keys})

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=16)
            if self.thread.is_alive():
                raise RuntimeError('Resource observer did not finish within its query timeout')


def verify_warm_cache(before, after, samples, exposures):
    deltas = {key: after[key] - before[key] for key in
              ('hits', 'misses', 'evictions', 'oversize_skips', 'disabled_bypasses')}
    if (before['entries'] != samples or after['entries'] != samples
            or deltas['hits'] != exposures or any(deltas[key] for key in deltas if key != 'hits')):
        raise RuntimeError('Not a fully retained warm-cache pass: ' + json.dumps(deltas))
    return deltas


def require_positive_gradient_norms(norms):
    if not norms or any(not math.isfinite(n) or n <= 0 for n in norms):
        raise RuntimeError('Every update must have a finite, nonzero gradient norm')


def verify_initializations(trials, phase):
    optimized = {trial['initial_model_sha256'] for trial in trials}
    if len(optimized) != 1 or not next(iter(optimized)):
        raise RuntimeError('Batch configurations did not start from the same optimized model state')
    frozen = {trial['frozen_backbone']['state_sha256_before'] for trial in trials} if phase == 'diff' else {None}
    if phase == 'diff' and (len(frozen) != 1 or not next(iter(frozen))):
        raise RuntimeError('P1 batch configurations did not start from the same frozen backbone state')
    return dict(all_batch_initializations_equal=True, optimized_state_sha256=next(iter(optimized)),
                frozen_backbone_state_sha256=next(iter(frozen)))


def write_report(output, report):
    """Replace only our exclusively claimed report; never reuse a fixed .partial."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                prefix='.' + output.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_pass(torch, backbone, head, optimizer, loader, device, observer, updates):
    """Preserve engine.train step semantics; omit epoch validation/persistence only."""
    from fwt_v2.engine import to_device, deterministic_loss, gradient_l2_norm
    model = backbone if head is None else head
    model.train()
    if head is not None:
        backbone.eval()
    losses, gradient_norms, diffusion_training_steps = [], [], []
    observed_sample_ids = []
    observer.check()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        observer.check()
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if head is None:
            logits = backbone(batch['x_norm'], batch['physics'])['logits_det']
            loss = deterministic_loss(logits, batch['target'], batch['valid_mask'], 'focal')
        else:
            with torch.no_grad():
                outputs = backbone(batch['x_norm'], batch['physics'])
            loss = head.training_loss(batch['target'], outputs, batch['physics'], mask=batch['valid_mask'])
            diffusion_training_steps.append(dict(update_index=updates + 1, epoch_batch_index=len(losses),
                sample_ids=list(batch['sample_id']), **deepcopy(head.last_training_diagnostics)))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        gradient_norms.append(gradient_l2_norm(gradients))
        optimizer.step()
        losses.append(float(loss.detach()))
        updates += 1
        observed_sample_ids.extend(batch['sample_id'])
    torch.cuda.synchronize(device)
    ended = time.perf_counter()
    observer.check()
    require_positive_gradient_norms(gradient_norms)  # Probe acceptance, outside the measured engine steps.
    return dict(seconds=ended - started, updates=len(losses), cumulative_updates=updates,
        sample_exposures=len(observed_sample_ids), observed_sample_ids=observed_sample_ids,
        loss_finite=True, loss_mean=sum(losses) / len(losses), loss_min=min(losses), loss_max=max(losses),
        gradient_finite=True, gradient_nonzero_updates=sum(n > 0 for n in gradient_norms),
        gradient_l2_min=min(gradient_norms), gradient_l2_max=max(gradient_norms),
        diffusion_step_records=len(diffusion_training_steps),
        diffusion_denoiser_evaluations=sum(s['denoiser_evaluations'] for s in diffusion_training_steps),
        diffusion_noise_draws=sum(s['noise_draws'] for s in diffusion_training_steps),
        observer=observer.summary(started, ended)), updates


def run_trial(args, config, data_config, cached, indices, phase, device, observer, result, save):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.engine import seed_everything, build_det, model_state_digest
    from fwt_v2.models import FWTConfig
    from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig

    require_no_jobs(args)
    observer.check()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result['memory_before'] = memory_snapshot()
    seed_everything(args.seed)
    model_config = FWTConfig(**config['model'])
    backbone = build_det(model_config, 'fwt', data_config).to(device)
    head = None
    if phase == 'diff':
        backbone.requires_grad_(False).eval()
        settings = dict(config['diffusion'], backbone_feature_channels=model_config.decoder_widths,
                        relational_channels=model_config.prop_channels)
        head = ConditionalResidualDiffusion(DiffusionConfig(**settings)).to(device)
    model = backbone if head is None else head
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    frozen_before = model_state_digest(backbone) if head is not None else None
    result.update(initial_model_sha256=model_state_digest(model),
        optimizer_defaults=deepcopy(optimizer.defaults),
        model_modes=dict(optimized='train', frozen_backbone='eval' if head is not None else None),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    loader = DataLoader(cached, batch_size=result['batch_size'], sampler=indices,
                        num_workers=0, pin_memory=False)
    torch.cuda.synchronize(device)
    result['initialization_and_state_hash_seconds'] = time.perf_counter() - started
    save()
    expected_ids = [cached.samples[i]['sample_id'] for i in indices]
    updates = 0
    for pass_index in range(args.passes + 1):
        before = cached.cache.stats()
        stats, updates = run_pass(torch, backbone, head, optimizer, loader, device, observer, updates)
        if stats.pop('observed_sample_ids') != expected_ids:
            raise RuntimeError('DataLoader changed the fixed selected sample order')
        stats['cache_delta'] = verify_warm_cache(before, cached.cache.stats(), len(indices), len(indices))
        if stats['updates'] != len(indices) // result['batch_size']:
            raise RuntimeError('Unexpected optimizer update count')
        stats['pass'] = pass_index
        if pass_index == 0:
            result['warmup'] = stats
            # Timed GPU peaks include resident model, optimizer and cached allocator
            # from warmup, but exclude a peak seen only during initialization/warmup.
            result['warmup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['warmup_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            result['timed_passes'].append(stats)
        save()
    measured = result['timed_passes']
    result.update(timed_seconds=sum(s['seconds'] for s in measured),
        timed_sample_exposures=sum(s['sample_exposures'] for s in measured),
        timed_updates=sum(s['updates'] for s in measured),
        timed_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        timed_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        memory_after=memory_snapshot(), observer=observer.summary(started, time.perf_counter()))
    result['samples_per_second'] = result['timed_sample_exposures'] / result['timed_seconds']
    if head is not None:
        frozen_after = model_state_digest(backbone)
        result['frozen_backbone'] = dict(state_sha256_before=frozen_before, state_sha256_after=frozen_after,
            unchanged=frozen_before == frozen_after,
            gradients_absent=all(p.grad is None for p in backbone.parameters()),
            parameters_require_grad=any(p.requires_grad for p in backbone.parameters()),
            eval_mode=not backbone.training)
        frozen = result['frozen_backbone']
        if not frozen['unchanged'] or not frozen['gradients_absent'] or frozen['parameters_require_grad'] or not frozen['eval_mode']:
            raise RuntimeError('P1 disposable backbone was not frozen')
    result['status'] = 'completed'
    del loader, optimizer, model, head, backbone
    gc.collect()
    torch.cuda.empty_cache()
    observer.check()
    save()


def execute(args, report, save):
    if sys.platform != 'linux':
        raise RuntimeError('Execution requires the configured Linux GPU environment')
    require_no_jobs(args)
    observer = Observer(args)
    observer.thread.start()
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        import torch
        import train as portable_train
        from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
        from fwt_v2.portable import read_json, file_sha256
        from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
        torch.set_num_threads(args.cpu_threads)
        root = args.data_root.resolve()
        controls = configure_gpu_mode(SimpleNamespace(deterministic_gpu=False), torch)
        report['imports_and_runtime_setup_seconds'] = time.perf_counter() - started
        report['environment'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            cpu_threads=torch.get_num_threads(), gpu_controls=controls)
        check_args = portable_train.parser().parse_args(['--phase', 'det', '--data-root', str(root),
            '--run-dir', str(ROOT / 'profile_output' / ('unused-' + uuid.uuid4().hex)),
            '--epochs', '30', '--device', f'cuda:{args.device_index}', '--cpu-threads', str(args.cpu_threads)])
        preflight_started = time.perf_counter()
        checked = portable_train.preflight(check_args)  # Original metadata and full source hashes.
        report['metadata_preflight_seconds'] = time.perf_counter() - preflight_started
        config = read_json(checked['config_path'])
        bundle = read_json(root / 'bundle.json')
        bound_paths = {name: root / item['path'] for name, item in bundle['files'].items()}
        bound_paths.update(original_manifest=root / bundle['original_full_manifest']['path'],
                           original_normalizer=root / bundle['original_normalizer']['path'], bundle=root / 'bundle.json')
        source_paths = list((ROOT / 'fwt_v2').glob('*.py')) + [ROOT / 'train.py', Path(__file__)]
        source_paths += [ROOT / 'scripts' / name for name in
                        ('benchmark_batches.py', 'benchmark_sample_cache.py', 'verify_real_batch.py')]
        watched = {str(p.resolve()): file_sha256(p) for p in source_paths + list(bound_paths.values())}
        report['provenance'] = dict(binding=checked['binding'], config=config,
            role_counts=checked['role_counts'], source_and_metadata_sha256=dict(watched))
        indexed = {(row['sample_id'], row['field']): row for row in read_json(bound_paths['arrays_index'])['files']}
        normalizer, data_config = NormalizationStats.load(checked['normalizer']), DataConfig(**config['data'])
        device = torch.device(f'cuda:{args.device_index}')
        os.chdir(root)  # Keep original dataset path resolution/weather semantic verification.
        save()
        for phase in args.phases:
            observer.check()
            phase_result = dict(phase=phase, role=ROLES[phase], status='preparing', selected_samples=[], trials=[])
            report['phases'].append(phase_result)
            phase_started = time.perf_counter()
            dataset = RCDADataset(checked['manifest'], ROLES[phase], normalizer, data_config)
            phase_result['full_dataset_and_weather_verification_seconds'] = time.perf_counter() - phase_started
            observer.check()
            if len(dataset) < args.samples:
                raise RuntimeError('Role has fewer samples than requested; do not silently reduce exposure')
            indices = spread_indices(len(dataset), args.samples)
            hashing_started = time.perf_counter()
            for index in indices:
                row, hashes = dataset.samples[index], {}
                for field in ('input_path', 'label_path'):
                    bound = indexed[(row['sample_id'], field)]
                    digest = file_sha256(row[field])
                    if bound['role'] != ROLES[phase] or digest != bound['sha256']:
                        raise RuntimeError('Selected array differs from its bound development bytes')
                    hashes[field] = digest
                    watched[str(Path(row[field]).resolve())] = digest
                phase_result['selected_samples'].append(dict(index=index, sample_id=row['sample_id'], arrays_sha256=hashes))
            phase_result['selected_array_hash_seconds'] = time.perf_counter() - hashing_started
            cache = SharedTensorCache(args.cache_mib * 1024**2)
            cached = CachedDataset(dataset, cache)
            fill_seconds = hash_seconds = 0.
            for sample in phase_result['selected_samples']:
                observer.check()
                tick = time.perf_counter()
                value = cached[sample['index']]
                fill_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                sample['cpu_sample_sha256'] = fingerprint(value, torch.Tensor)
                hash_seconds += time.perf_counter() - tick
                del value
            phase_result.update(cold_cache_getitem_seconds=fill_seconds, cpu_sample_hash_seconds=hash_seconds,
                                cache_after_fill=cache.stats())
            if cache.stats()['entries'] != args.samples or cache.stats()['misses'] != args.samples or cache.stats()['evictions']:
                raise RuntimeError('Cache cannot retain every selected real sample')
            require_no_jobs(args)  # Dataset initialization can take minutes; recheck before CUDA.
            observer.check()
            if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
                raise RuntimeError('Configured CUDA device is unavailable')
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(.85, device)
            report['environment']['gpu_name'] = torch.cuda.get_device_name(device)
            phase_result['status'] = 'running'
            save()
            for batch_size in args.batches:
                result = dict(batch_size=batch_size, status='running', timed_passes=[])
                phase_result['trials'].append(result)
                run_trial(args, config, data_config, cached, indices, phase, device, observer, result, save)
            phase_result['initialization_comparison'] = verify_initializations(phase_result['trials'], phase)
            phase_result.update(status='completed', cache_after_trials=cache.stats())
            del cached, dataset
            cache.clear()
            del cache
            gc.collect()
            save()
        recheck_started = time.perf_counter()
        for path, expected in watched.items():
            observer.check()
            if file_sha256(path) != expected:
                raise RuntimeError('Source, metadata or selected array changed during probe: ' + path)
        report['final_source_data_recheck_seconds'] = time.perf_counter() - recheck_started
        require_no_jobs(args)
        observer.check()
        report['status'] = 'completed'
    except BaseException:
        report['memory_at_failure'] = memory_snapshot()
        if 'torch' in locals() and torch.cuda.is_initialized():
            report['gpu_at_failure'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        raise
    finally:
        os.chdir(previous_cwd)
        observer.close()
        report['resource_observer_summary'] = observer.summary()
        report['execution_wall_seconds'] = time.perf_counter() - started
        report['memory_after'] = memory_snapshot()
        observer.check()


def main(argv=None):
    args = parse_args(argv)
    report = make_plan(args)
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != 'linux':
        raise RuntimeError('Execution requires the configured Linux GPU environment')
    require_no_jobs(args)
    output = checked_output(args.output or ROOT / 'profile_output' / ('real-batches-' + uuid.uuid4().hex[:12] + '.json'), args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.update(status='running', memory_before=memory_snapshot())
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    def save():
        write_report(output, report)

    try:
        execute(args, report, save)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        for phase in report['phases']:
            if phase['status'] in ('preparing', 'running'):
                phase['status'] = 'failed'
            for trial in phase['trials']:
                if trial['status'] == 'running':
                    trial['status'] = 'failed'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(dict(status=report['status'], output=str(output))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
