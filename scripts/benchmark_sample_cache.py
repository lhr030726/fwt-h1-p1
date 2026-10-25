"""CPU-only, read-only Tb sample-cache probe with full original weather verification.

Measures __getitem__ on 96 spread samples, first ordered then shuffled. Hashing,
source verification, and dataset construction are outside the reported fetch time.
No model, GPU context, checkpoint, held-out arrays, or training run is created.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def spread_indices(length, count):
    if length < 1 or count < 1:
        raise ValueError('positive dataset length and sample count required')
    count = min(length, count)
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


def fingerprint(sample, tensor_type=()):
    """Length-framed structural hash includes every scalar and tensor dtype/shape/byte."""
    digest = hashlib.sha256()

    def frame(payload):
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)

    def visit(value):
        if isinstance(value, tensor_type):
            frame(json.dumps(['tensor', str(value.dtype), list(value.shape)]).encode())
            frame(value.detach().contiguous().numpy().tobytes(order='C'))
        elif type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('sample dictionary keys must be strings')
            frame(b'dict')
            frame(str(len(value)).encode())
            for key in sorted(value):
                frame(key.encode('utf-8'))
                visit(value[key])
        elif type(value) in (tuple, list):
            frame(type(value).__name__.encode())
            frame(str(len(value)).encode())
            for item in value:
                visit(item)
        elif value is None or type(value) in (str, int, bool, float):
            frame(json.dumps([type(value).__name__, value], ensure_ascii=False, allow_nan=False).encode('utf-8'))
        else:
            raise TypeError('unsupported sample value: ' + type(value).__name__)

    visit(sample)
    return digest.hexdigest()


def process_state(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text()
        fields = value[value.rfind(')') + 2:].split()
        return dict(pid=pid, exists=True, state=fields[0], start_ticks=int(fields[19]),
                    user_ticks=int(fields[11]), system_ticks=int(fields[12]))
    except (OSError, ValueError, IndexError):
        return dict(pid=pid, exists=False)


def memory_snapshot():
    result = {}
    for path, keys in [('/proc/self/status', {'VmRSS': 'rss_bytes', 'VmHWM': 'peak_rss_bytes'}),
                       ('/proc/meminfo', {'MemAvailable': 'host_available_bytes'})]:
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(':')
                if key in keys:
                    result[keys[key]] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
    for base, suffixes in [('/sys/fs/cgroup/', [('memory.current', 'current'), ('memory.max', 'limit')]),
                           ('/sys/fs/cgroup/memory/', [('memory.usage_in_bytes', 'current'), ('memory.limit_in_bytes', 'limit')])]:
        for suffix, name in suffixes:
            key = 'cgroup_' + name + '_bytes'
            try:
                if key not in result:
                    raw = Path(base + suffix).read_text().strip()
                    result[key] = None if raw == 'max' else int(raw)
            except (OSError, ValueError):
                pass
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--concurrent-training-pid', type=int)
    args = parser.parse_args(argv)
    root, output = args.data_root.resolve(), args.output.resolve()
    if not 1 <= args.samples <= 256 or output.exists() or root == output or root in output.parents:
        parser.error('use 1..256 samples and a new output path outside the data root')
    if 'torch' in sys.modules:
        raise RuntimeError('Run in a fresh process so CUDA/thread guards precede imports')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2 import geography
    from fwt_v2.portable import read_json, bound_file, file_sha256
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was unexpectedly initialized')
    report = dict(schema='fwt_actual_sample_cache_probe_v1', status='running',
                  created_at=datetime.now(timezone.utc).isoformat(), role='backbone_train',
                  scope='CPU __getitem__ only; not end-to-end training speed or scientific metrics',
                  cpu_threads=1, cuda_visible_devices='-1', model_loaded=False,
                  seed=args.seed, memory_before=memory_snapshot(), results=[])
    if args.concurrent_training_pid:
        report['concurrent_training_before'] = process_state(args.concurrent_training_pid)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix + '.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(output)

    save()
    previous_cwd = Path.cwd()
    old_capacity = geography._image_basis.cache_info().maxsize
    try:
        started = time.perf_counter()
        bundle = read_json(root / 'bundle.json')
        if (bundle.get('data_scope') != 'development_only' or bundle.get('heldout_arrays_included') is not False):
            raise ValueError('development-only bound bundle required')
        files = {name: bound_file(root, bundle['files'][name])
                 for name in ('manifest', 'normalizer', 'config', 'weather', 'arrays_index')}
        report['provenance'] = dict(bundle=file_sha256(root / 'bundle.json'),
            files={name: file_sha256(path) for name, path in files.items()},
            sources={path.name: file_sha256(path) for path in sorted((ROOT / 'fwt_v2').glob('*.py'))},
            probe=file_sha256(__file__), torch=torch.__version__, python=sys.version)
        report['metadata_verification_seconds'] = time.perf_counter() - started
        os.chdir(root)  # Preserve bound relative weather paths and all original checks.
        started = time.perf_counter()
        dataset = RCDADataset(files['manifest'], 'backbone_train', NormalizationStats.load(files['normalizer']),
                              DataConfig(**read_json(files['config'])['data']))
        report['dataset_startup_and_weather_verification_seconds'] = time.perf_counter() - started
        indices = spread_indices(len(dataset), args.samples)
        sequences = dict(ordered=indices, shuffled=list(indices))
        random.Random(args.seed).shuffle(sequences['shuffled'])
        report['indices'] = indices
        report['sample_ids'] = [dataset.samples[index]['sample_id'] for index in indices]
        # Verify exact selected input/label bytes against the bound published index.
        started = time.perf_counter()
        indexed = {(row['sample_id'], row['field']): row for row in read_json(files['arrays_index'])['files']}
        selected_hashes = {}
        for index in indices:
            row = dataset.samples[index]
            for field in ('input_path', 'label_path'):
                bound = indexed[(row['sample_id'], field)]
                actual = file_sha256(row[field])
                if bound['role'] != 'backbone_train' or actual != bound['sha256']:
                    raise ValueError('selected source array differs from its bound bytes')
                selected_hashes[row['sample_id'] + ':' + field] = actual
        report['selected_array_sha256'] = selected_hashes
        report['selected_array_verification_seconds'] = time.perf_counter() - started
        reference, reference_sequences = {}, {}
        modes = [('original_geometry32', 32, 0), ('geometry1024', 1024, 0),
                 ('geometry1024_ram2gib', 1024, 2 * 1024**3)]
        for mode, capacity, budget in modes:
            geography.configure_image_basis_cache(capacity)
            cache = SharedTensorCache(budget) if budget else None
            selected = CachedDataset(dataset, cache) if cache is not None else dataset
            gc.collect()
            for order, sequence in sequences.items():
                fetch_seconds = hash_seconds = 0.
                combined = hashlib.sha256()
                geo_before = geography._image_basis.cache_info()
                for index in sequence:
                    started = time.perf_counter()
                    sample = selected[index]
                    fetch_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    checksum = fingerprint(sample, torch.Tensor)
                    if checksum != reference.setdefault(index, checksum):
                        raise AssertionError(f'Bitwise sample mismatch: mode={mode}, index={index}')
                    combined.update(index.to_bytes(8, 'big') + bytes.fromhex(checksum))
                    hash_seconds += time.perf_counter() - started
                    del sample
                sequence_hash = combined.hexdigest()
                if sequence_hash != reference_sequences.setdefault(order, sequence_hash):
                    raise AssertionError('sequence hash mismatch')
                after = geography._image_basis.cache_info()
                result = dict(mode=mode, order=order, samples=len(sequence), getitem_seconds=fetch_seconds,
                    samples_per_second=len(sequence) / fetch_seconds, hash_seconds=hash_seconds,
                    outputs_sha256=sequence_hash, bitwise_equal=True,
                    geography_cache=after._asdict(), geography_hits=after.hits-geo_before.hits,
                    geography_misses=after.misses-geo_before.misses,
                    sample_cache=None if cache is None else cache.stats(), memory=memory_snapshot())
                report['results'].append(result)
                save()
                print(json.dumps(result), flush=True)
            if cache is not None:
                cache.clear()
            del selected, cache
        report.update(status='passed', sample_output_sha256=reference,
                      cuda_initialized=torch.cuda.is_initialized(), memory_after=memory_snapshot(),
                      memory_note='RSS peak is cumulative across modes; tensor budget excludes temporaries/Python overhead',
                      timing_note='fixed mode order, shared OS page cache and concurrent CPU activity can affect timings')
        if report['cuda_initialized']:
            raise RuntimeError('CUDA context guard failed')
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        geography.configure_image_basis_cache(old_capacity)
        os.chdir(previous_cwd)
        if args.concurrent_training_pid:
            report['concurrent_training_after'] = process_state(args.concurrent_training_pid)
        save()
    print(json.dumps(dict(status=report['status'], output=str(output))), flush=True)


if __name__ == '__main__':
    main()
