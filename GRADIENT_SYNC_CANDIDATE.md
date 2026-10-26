# Gradient diagnostic transfer candidate

Prepared on 2026-10-26 from commit `5da517f51b8d6b6ca06ff008b7b42a1b170baa64` in a separate local clone. This candidate has not been deployed, profiled on the target GPU, or used for training. The active H1/P1 sources, environment, data and jobs are unchanged.

## Change and intended effect

`engine.py` now imports two read-only helpers from `gradient_diagnostics.py`. For a list of dense, floating point gradients on one CUDA device, the helper still runs `torch.isfinite(g).all()` separately for every tensor, then transfers a stacked boolean vector to CPU. Only when all flags pass does the existing engine call the norm helper, which still computes each tensor's float64 `torch.linalg.vector_norm`, transfers a stacked float64 vector and combines the resulting Python floats with `math.hypot` in the original parameter order.

The potential saving is fewer host waits and scalar transfers. This does not remove the individual reduction kernels, change the optimizer, change precision, clip gradients, enlarge the batch or consume random numbers. Loss and other synchronization points remain.

CPU model construction using the frozen `configs/h1.json` found H1 has 255 trainable parameter tensors / 5,638,074 elements, and P1 has 80 / 1,015,761. If every trainable parameter has a gradient, the two diagnostics currently trigger 510 or 160 explicit scalar readbacks per step; the candidate uses 2. These counts are architecture inventory, not measured GPU timing. Missing gradients reduce the original count.

## Behavior and limits

- CPU, mixed device and unsupported layout/type inputs use the original per-tensor path. Empty standalone helpers return `True` and `0.0`; the engine still rejects missing gradients.
- NaN/Inf gradients are rejected before norm computation and `optimizer.step`. Finite all-zero gradients keep the old behavior.
- Every norm reduction remains float64 and the Python reduction order is preserved. No new rejection of an overflowing norm is added.
- The CUDA finite branch evaluates all reductions before transferring flags. It no longer short circuits GPU work on the first failed flag. Therefore, the ordering of unrelated errors from later tensors, failed allocations or broken devices may differ.
- Different allocation/synchronization timing does not guarantee bitwise-identical future training with nondeterministic CUDA operators.
- The candidate changes the numerical source digest. It must get a fresh deployment, provenance and any required compatibility evidence before use. Do not modify or resume the active experiment under its old binding.

## Validation performed locally

Use the local CPU interpreter with `CUDA_VISIBLE_DEVICES=-1` (an empty PowerShell environment value removes the variable and does not reliably hide GPUs). CUDA-specific tests additionally require `FWT_RUN_CUDA_EQUIVALENCE=1`.

```
python -B -m unittest discover -s tests -p 'test_gradient_*.py' -v
```

The CPU cases check exact finite/norm results for empty, zero, ordinary, large finite, mixed dtype, noncontiguous, NaN and Inf tensors; unchanged input tensors and RNG; exact one-step AdamW state and parameter equality; and the actual engine gradient boundary refusing missing/nonfinite gradients without calling norm or optimizer. A forced branch on CPU checks the stack/list algorithm, but does not establish target CUDA equivalence. CUDA tests remain skipped in the recorded CPU run.

Recorded final run: 13 tests discovered, 11 passed and 2 CUDA cases explicitly skipped, exit 0. The adjacent `../gradient-cpu-test-report.json` records stdout/stderr, command, visibility settings and hashes of all tested implementation/test files. `../gradient-shape-inventory.json` records the CPU architecture counts. The local optional huggingface_hub integration prints a missing-filelock warning; it is not imported by this candidate or required by the passing tests.

The benchmark gate fixtures check CPU-only training preflight, another CUDA PID, malformed/failed nvidia-smi output and acceptance of only the benchmark's own CUDA PID. No remote benchmark or GPU profiling has run. A preliminary local test run accidentally exercised two short CUDA equivalence cases because an empty visibility variable was treated as unset; these were on the workstation, not the server, and do not provide target GPU validation.

## Future server validation, after training is complete

Deploy this clone to a new directory after the active pipeline completes. Use the already validated server environment; do not rebuild it. Running the script without `--execute` only prints a plan and imports no torch:

```
python -B scripts/benchmark_gradient_diagnostics.py
python -B scripts/benchmark_gradient_diagnostics.py --execute \
  --output /NEW_RESULT_DIRECTORY/gradient-diagnostics.json \
  --profile-dir /NEW_RESULT_DIRECTORY/traces
```

Execution requires Linux and an empty external GPU process list, and refuses recognized training processes even while they are still in CPU preflight. Checks repeat before/after every measured block and before profiling. Output paths must be new. This is an observational idle gate, not a scheduler lock: externally starting a job concurrently can race it; use only after the known pipeline is fully complete.

The script builds the actual H1 and P1 architectures on CPU, allocates deterministic synthetic gradients matching their parameter shapes, checks exact norm equality, alternates legacy/candidate timing order and measures each with CUDA synchronization around the entire repeated block. It can export separate real torch-profiler CPU/CUDA traces for each phase/implementation. It checks unchanged gradients and CUDA RNG and writes hashes, shapes, raw timings, idle observations and diagnostic-only timing ratios. It performs no optimizer step, data access, checkpoint loading, model forward/backward or training.

Before adoption, also run the opt-in CUDA edge-case equivalence tests only on the verified idle target, and a fresh disposable real-data whole-step comparison. The isolated diagnostic ratio must not be reported as full training speedup; validation and data preparation are excluded. Adopt only if the end-to-end gain is meaningful and semantic checks pass.
