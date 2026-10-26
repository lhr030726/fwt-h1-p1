"""Opt-in, idle-GPU-only diagnostic benchmark; no datasets, weights or training."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def idle_gate(proc_root=Path('/proc')):
    if sys.platform != 'linux':
        raise RuntimeError('GPU execution requires Linux /proc and nvidia-smi checks')
    conflicts = []
    names = {'train.py', 'benchmark_real_batches.py', 'verify_zero_initialization.py',
             'check_zero_init_real.py',
             'launch_extended_formal.py', 'launch_zero_formal.py'}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            raw = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Never expose command lines, which may contain unrelated credentials.
        if any(Path(arg.decode(errors='replace')).name in names for arg in raw if arg):
            conflicts.append(int(directory.name))
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    for row in csv.reader(output.splitlines()):
        if row and (len(row) != 1 or not row[0].strip().isdigit()):
            raise RuntimeError('Unexpected nvidia-smi PID query output; refuse GPU work')
        if row and int(row[0]) != os.getpid():
            conflicts.append(int(row[0]))
    if conflicts:
        raise RuntimeError(f'Idle gate blocked: other training/compute PIDs {sorted(set(conflicts))}')
    return {'observed_at_unix': time.time(), 'other_training_or_compute_pids': []}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--profile-dir', type=Path)
    args = parser.parse_args()
    plan = {'status': 'plan_only', 'scope': 'fixed synthetic gradients in actual H1/P1 parameter shapes',
            'excluded': ['data loading', 'forward/backward', 'optimizer', 'validation', 'full epoch'],
            'candidate': 'per-tensor reductions with batched CUDA-to-CPU scalar transfers',
            'required': 'no training process, no other GPU compute process, explicit --execute'}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None or args.output.exists():
        raise ValueError('--output must name a new file')
    if not 1 <= args.iterations <= 100 or not 3 <= args.repeats <= 15:
        raise ValueError('Require 1..100 iterations and 3..15 repeats')
    if args.profile_dir is not None and args.profile_dir.exists():
        raise ValueError('--profile-dir must name a new directory')
    gates = [idle_gate()]  # Must precede every torch/CUDA import and initialization.
    import torch
    from fwt_v2.gradient_diagnostics import gradient_l2_norm, gradients_finite
    from fwt_v2.models import FWTConfig, FWTDet
    from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    config_path = ROOT / 'configs' / 'h1.json'
    config = json.loads(config_path.read_text())
    model_config = FWTConfig(**config['model'])
    report = dict(plan, status='running', torch_version=torch.__version__,
                  device_name=torch.cuda.get_device_name(device),
                  source_sha256=hashlib.sha256(''.join(p.name + digest(p)
                      for p in sorted((ROOT / 'fwt_v2').glob('*.py'))).encode()).hexdigest(),
                  config_sha256=digest(config_path), engine_sha256=digest(ROOT / 'fwt_v2/engine.py'),
                  candidate_sha256=digest(ROOT / 'fwt_v2/gradient_diagnostics.py'),
                  baseline='exact pre-candidate all(isfinite(...)) plus float64 vector_norm/math.hypot',
                  iterations=args.iterations, repeats=args.repeats, phases={}, idle_checks=gates)
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True)

    def legacy(gradients):
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        return math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                            for g in gradients))

    def candidate(gradients):
        if not gradients or not gradients_finite(gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        return gradient_l2_norm(gradients)

    for phase in ('det', 'diff'):
        gates.append(idle_gate())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = FWTDet(model_config) if phase == 'det' else ConditionalResidualDiffusion(
                DiffusionConfig(**dict(config['diffusion'], output_initialization='zero',
                    backbone_feature_channels=model_config.decoder_widths,
                    relational_channels=model_config.prop_channels)))
        parameter_shapes = [{'name': name, 'shape': list(p.shape), 'numel': p.numel()}
                            for name, p in model.named_parameters() if p.requires_grad]
        del model
        # Identical fixed values; no model gradients, checkpoint state or RNG are consumed.
        gradients = [((torch.arange(p['numel'], dtype=torch.float32) % 257 - 128) / 128)
                     .reshape(p['shape']).to(device) for p in parameter_shapes]
        before = [g.clone() for g in gradients]
        rng_before = torch.cuda.get_rng_state(device).clone()
        expected = legacy(gradients)
        observed = candidate(gradients)
        if expected != observed or not math.isfinite(observed):
            raise RuntimeError('Exact norm equivalence failed')
        for _ in range(3):
            legacy(gradients)
            candidate(gradients)
        measured = {'legacy': [], 'candidate': []}
        for repeat in range(args.repeats):
            # Alternate order so one implementation is not always measured after the other.
            order = [('legacy', legacy), ('candidate', candidate)]
            if repeat % 2:
                order.reverse()
            for name, operation in order:
                gates.append(idle_gate())
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for _ in range(args.iterations):
                    result = operation(gradients)
                torch.cuda.synchronize(device)
                measured[name].append((time.perf_counter() - started) / args.iterations)
                gates.append(idle_gate())
                if result != expected:
                    raise RuntimeError('Repeated exact norm mismatch')
        profiles = {}
        if args.profile_dir is not None:
            for name, operation in [('legacy', legacy), ('candidate', candidate)]:
                gates.append(idle_gate())
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    with torch.profiler.record_function(f'{phase}_{name}_gradient_diagnostics'):
                        operation(gradients)
                    torch.cuda.synchronize(device)
                trace = args.profile_dir / f'{phase}-{name}.json'
                profile.export_chrome_trace(str(trace))
                profiles[name] = dict(path=str(trace), sha256=digest(trace))
        unchanged = all(torch.equal(g, old) for g, old in zip(gradients, before))
        rng_unchanged = torch.equal(rng_before, torch.cuda.get_rng_state(device))
        if not unchanged or not rng_unchanged:
            raise RuntimeError('Diagnostics changed gradients or CUDA RNG')
        medians = {name: statistics.median(values) for name, values in measured.items()}
        report['phases'][phase] = dict(parameter_shapes=parameter_shapes,
            parameter_tensors=len(gradients), parameters=sum(p['numel'] for p in parameter_shapes),
            exact_norm=observed, exact_equivalence=True, input_unchanged=unchanged,
            cuda_rng_unchanged=rng_unchanged, seconds_per_diagnostics=measured,
            median_seconds=medians, diagnostics_only_ratio=medians['legacy'] / medians['candidate'],
            explicit_scalar_transfer_calls=dict(legacy=2 * len(gradients), candidate=2),
            profiles=profiles)
        del gradients, before
        torch.cuda.empty_cache()
    gates.append(idle_gate())
    report['status'] = 'completed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'report': str(args.output),
                      'report_sha256': digest(args.output)}, indent=2))


if __name__ == '__main__':
    main()
