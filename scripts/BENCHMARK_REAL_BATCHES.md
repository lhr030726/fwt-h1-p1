# Real-data batch throughput probe

Preparation only until all formal training and queue processes have ended. This
script never loads/saves checkpoints or starts formal training. Its default is a
stdlib-only plan; it does not open the data or query/initialize a GPU:

```sh
python scripts/benchmark_real_batches.py --data-root /path/to/development-bundle
```

After verifying that the server is idle, explicitly opt in with a **new** report:

```sh
python scripts/benchmark_real_batches.py --data-root /path/to/development-bundle \
  --output profile_output/real-batches-first.json --execute
```

Defaults: H1 and P1; batches 1, 16, 32; 96 uniformly spread real samples from Tb
and Tr respectively; 1 warmup pass plus **3 timed passes** per configuration.
Timed exposure is always 288 samples, giving 288/18/9 updates. Each configuration
gets newly seeded random models and AdamW; P1's random backbone is frozen and
its parameters/buffers are hashed before/after. Each phase verifies equal initial
model hashes across its batch configurations (including P1's frozen backbone).
No V/test arrays are loaded. The original full bundle/source preflight runs once
per probe; original RCDADataset/POWER verification runs once per phase, without
monkeypatching. Metadata checks include the full
development manifest. Selected input/label bytes are bound to the array index.

For a smaller explicit smoke only, use `--phases det --batches 1 16 32 --samples 32
--passes 1`. The full comparison remains the default. `--blocked-pid PID` can
add a known job blocker (missing/zombie allowed, all other states rejected).
Known formal queue scripts, training/probe Python processes and **all other GPU
compute processes** are rejected. The observer checks every 2 seconds; errors or
new jobs abort at the next batch check. It does not wait, retry, kill processes,
skip a failed size or recover automatically from OOM. This polling gate is not an
atomic GPU reservation.

Each whole-pass timer starts before DataLoader iteration and ends after CUDA
synchronization. It includes CPU cached-sample cloning, default collate, original
H2D, model forward/loss/backward, finite checks, original gradient norm and AdamW,
loss/norm bookkeeping, P1 diagnostics `deepcopy`, and lightweight sample-order /
observer checks. H1 stays in training mode (including modality dropout); P1 head
is training and backbone is eval/no_grad. FP32 and inherited GPU backend settings
are recorded without changes. The allocator is capped at 85% of GPU memory.

Startup/imports, metadata validation, full dataset/weather validation, selected
array hashes, cold-cache fetch, CPU sample fingerprints, random model setup and
warmup are reported separately. Validation, checkpoint/report writes, model
hashing, first-batch report IO and summary reduction are outside pass timing.
Cache budget defaults to 1 GiB of retained tensors; process RSS is larger. All
96 entries must remain resident and every hot pass must have exactly 96 hits,
zero misses and zero evictions; otherwise the report fails rather than claiming
warm-cache throughput. Source/metadata and selected array hashes are rechecked
after execution.

Per-pass seconds are retained; aggregate samples/s uses total samples / total
timed seconds. GPU allocated/reserved peaks include resident model/optimizer but
exclude context/driver memory. Timed peaks reset after warmup. Background
procfs/cgroup and nvidia-smi observations add overhead **included in throughput**;
poll count and elapsed time are reported, not subtracted. RSS/cgroup maxima are
sampled and can miss peaks; VmHWM is cumulative across the process. CPU allocator
state, trial order and GPU clocks are not controlled experiments.

Different batch sizes change update counts, weights and RNG consumption. These
finite, nonzero-gradient checks and throughput measurements provide no quality,
convergence, training-equivalence, confidence-interval or full-epoch speed claim.
Reports use fresh paths and unique exclusive temporary files for atomic updates;
a failure keeps its report with partial results and stops the probe. Finite,
positive gradient norms are required for every update, checked after pass timing.

Offline safety/control tests (no torch/data/GPU required):

```sh
python -m unittest discover -s tests -p test_benchmark_real_batches.py -v
```

Instrumentation changelog (`fwt_v2/` and `train.py` are unchanged):

| New file | Purpose |
| --- | --- |
| `scripts/benchmark_real_batches.py` | Isolated opt-in real-data throughput probe and resource observations. |
| `tests/test_benchmark_real_batches.py` | Offline safety, exposure, cache and report checks. |
| `scripts/BENCHMARK_REAL_BATCHES.md` | Commands, timing boundaries and measurement limitations. |
