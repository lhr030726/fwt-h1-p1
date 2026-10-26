# P1 validation sampling in groups

`FWT_P1_VALIDATION_PARALLEL_SAMPLES` is a positive integer, default `1`.
Training reads it once and records it as
`configuration.json.performance.p1_validation_parallel_samples`, also included
in checkpoint provenance and the strict training-state binding. This performance
setting does not change K, DDIM steps, seed, loss, optimizer, or batch size.

For direct validation/benchmarking use
`raw_probability(..., parallel_samples=N)` or
`validation_records(..., parallel_samples=N)`. The effective group size is at
most the current validation batch. The final short group is handled identically.

For DDIM, each sample still receives its own CPU generator seeded by the exact
existing SHA-256 base-seed/sample-ID rule. Each generator draws the original
`(K, 1, H, W)` noise tensor, preserving K ordering. CPU draws are concatenated
in sample-major order and transferred once. The existing sampler runs unchanged
DDIM math over `(group_size * K)` conditions, including per-draw residual clipping
and probability clipping before averaging. `sample(..., sample_generators=...)`
is a DDIM-only API; supplying both generator arguments is rejected. DDPM keeps
the original serial per-sample sampling path, including all per-step noise draws.

The denoiser uses GroupNorm rather than batch statistics. Nonetheless, changed
batch shapes can select different floating-point kernels, so GPU numerical
equivalence and end-to-end timing must be measured before choosing a group size.
This code does not bypass or relax source/checkpoint compatibility. An old
training state cannot resume directly under these new source hashes; any
authorized migration must separately verify and record its exact lineage.

Local CPU verification (`CUDA_VISIBLE_DEVICES=-1`):

- `python -B -m unittest discover -s tests -p test_parallel_validation.py -v`
  passed 8 tests: exact original initial noises and sample/K order; real 8-draw,
  4-step sampler outputs for groups 1/2/4/8; tail groups; sample reordering and
  loader boundaries; unchanged global/caller RNG; clamp-before-mean; DDPM exact
  fallback; invalid controls/generator combinations. Real output comparisons
  use rtol 2e-5 and atol 2e-6; initial noises use exact tensor equality.
- Six existing zero-output initialization tests passed.
- Gradient diagnostic suite: 11 passed, 2 CUDA tests explicitly skipped.

No local or remote GPU run was performed for this change. CPU test throughput
is not evidence of server speedup. Existing engine/diffusion CRLF bytes were
preserved, and `git -c core.whitespace=cr-at-eol diff --check` is used accordingly.
