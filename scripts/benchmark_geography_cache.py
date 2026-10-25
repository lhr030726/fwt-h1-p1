"""CPU-only cache benchmark; reads manifest geometry, never arrays or labels.

Compare the original capacity 32 with the candidate on identical ordered and
shuffled sample sequences. The wind inputs are synthetic fixed FP32 arrays;
the georeferences are real. Equality hashes cover every FP64 basis and every
FP32 projected wind output. Timing excludes hashing and input construction.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2 import geography


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except ImportError:
        return None


def benchmark(rows, sizes, repeats, seed, pixel_size):
    permutations = {'ordered': list(rows), 'shuffled': list(rows)}
    random.Random(seed).shuffle(permutations['shuffled'])
    results, reference = [], {}
    old_size = geography._image_basis.cache_info().maxsize
    inputs = {}
    try:
        for capacity in sizes:
            for order, sequence in permutations.items():
                geography.configure_image_basis_cache(capacity)
                retained = OrderedDict()
                for repeat in range(repeats):
                    before = geography._image_basis.cache_info()
                    digest = hashlib.sha256()
                    operator_seconds = 0.0
                    started = time.perf_counter()
                    peak_cached_bytes = sum(retained.values())
                    for row in sequence:
                        geo = row['georeference']
                        shape = tuple(geo['shape'])
                        if shape not in inputs:
                            inputs[shape] = (np.ones(shape, np.float32),
                                             np.full(shape, .37, np.float32))
                        speed, angle = inputs[shape]
                        affine, crs = geography.validate_georeference(geo, shape, pixel_size)
                        key = (affine, crs, shape)
                        tick = time.perf_counter()
                        basis = geography._image_basis(*key)
                        wind = geography.geographic_wind(speed, angle, geo, pixel_size)
                        operator_seconds += time.perf_counter() - tick
                        if basis.flags.writeable:
                            raise AssertionError('Cached geometry must remain immutable')
                        digest.update(row['sample_id'].encode('utf-8'))
                        for array in (basis, *wind):
                            digest.update(array.tobytes())
                        retained[key] = basis.nbytes
                        retained.move_to_end(key)
                        if len(retained) > capacity:
                            retained.popitem(last=False)
                        peak_cached_bytes = max(peak_cached_bytes, sum(retained.values()))
                    after = geography._image_basis.cache_info()
                    checksum = digest.hexdigest()
                    if order not in reference:
                        reference[order] = checksum
                    if checksum != reference[order]:
                        raise AssertionError(f'Bitwise output mismatch: {capacity}/{order}/{repeat}')
                    result = dict(capacity=capacity, order=order, repeat=repeat,
                                  cache_state='cold' if repeat == 0 else 'warm',
                                  samples=len(sequence), operator_seconds=operator_seconds,
                                  elapsed_seconds=time.perf_counter() - started,
                                  outputs_sha256=checksum, bitwise_equal=True,
                                  hits=after.hits - before.hits, misses=after.misses - before.misses,
                                  cache_info=after._asdict(), cached_basis_bytes=sum(retained.values()),
                                  peak_cached_basis_bytes=peak_cached_bytes,
                                  process_peak_rss_bytes=peak_rss_bytes())
                    results.append(result)
                    print(json.dumps(result), flush=True)
    finally:
        geography.configure_image_basis_cache(old_size)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roles', nargs='+', default=['backbone_train', 'residual_train', 'validation'],
                        choices=['backbone_train', 'residual_train', 'validation'])
    parser.add_argument('--max-samples', type=int, default=0, help='0 means all selected role rows')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--candidate-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pixel-size', type=float, default=50.0)
    args = parser.parse_args()
    if args.repeats < 2 or args.max_samples < 0 or args.output.exists():
        parser.error('Use repeats>=2, max-samples>=0 and a new output path')
    payload = args.manifest.read_bytes()
    document = json.loads(payload.decode('utf-8-sig'))
    rows = [row for row in document['samples'] if row['role'] in args.roles]
    if args.max_samples:
        # Spread a bounded test over the manifest, avoiding only the first event.
        indices = np.linspace(0, len(rows) - 1, min(len(rows), args.max_samples)).astype(int)
        rows = [rows[index] for index in indices]
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        parser.error('Require nonempty, unique sample identities')
    identities = {json.dumps(row['georeference'], sort_keys=True) for row in rows}
    results = benchmark(rows, [32, args.candidate_size], args.repeats, args.seed, args.pixel_size)
    import pyproj
    report = dict(schema='fwt_geography_cache_benchmark_v1', status='passed',
                  manifest_sha256=hashlib.sha256(payload).hexdigest(),
                  geography_source_sha256=hashlib.sha256(Path(geography.__file__).read_bytes()).hexdigest(),
                  python=sys.version, numpy=np.__version__, pyproj=pyproj.__version__,
                  selected_roles=args.roles, samples=len(rows), geometries=len(identities),
                  sample_ids=[row['sample_id'] for row in rows], seed=args.seed,
                  scope='actual georeferences; synthetic wind; no model/data-array/GPU benchmark',
                  memory_note='basis bytes exclude Python/cache/temporary overhead; RSS is cumulative process peak',
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
