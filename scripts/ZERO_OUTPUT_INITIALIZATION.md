# P1 zero-output initialization variant

Added 2026-10-26: `train.py --p1-zero-init` opts into a new P1 initialization recipe.
Use this flag for **both** H1 and P1 in a fresh run directory. The portable binding
records `recipe.p1_output_initialization=zero`; omitting the flag records `random`.
The bundle configuration must still match its original bytes and fixed architecture.
Only an in-memory configuration copy changes after the repeated execution preflight.

`DiffusionConfig.output_initialization` defaults to `random`, retaining previous
constructor RNG draws and parameter/buffer keys when the field is absent. `zero` is
accepted only for the clean-probability diffusion head. Construction finishes
normally, then only the final convolution weight and bias are set to zero. This
preserves constructor RNG consumption and all other initial tensors. The existing
source-version guards on checkpoints remain in force.

P1 configuration/checkpoint provenance records
`initialization=new_random_initialization_zero_output` and
`diffusion_config.output_initialization=zero`; diagnostics also record
`output_initialization=zero`. H1 retains its existing random initialization.

Initially the learned logit correction and final DDIM residual are zero. The K=8
probability reduction can introduce floating-point rounding. Calibration logits
retain the existing clipping behavior and need not equal H1 raw logits. The first
update has zero upstream data gradients; AdamW can still decay upstream weights.
The output layer can learn on that update, allowing upstream gradients afterward.
The untrained initial state is **not** added to checkpoint candidates. Validation
AP selection remains restricted to completed epochs. No quality gain is guaranteed.

On an idle configured Linux GPU host, inspect then explicitly run the bounded witness:

```bash
python scripts/check_zero_init_real.py --data-root /path/to/data
python scripts/check_zero_init_real.py --data-root /path/to/data --output profile_output/zero-init-real.json --execute
```

The default does not import torch, open data, or query the GPU. Execution preserves
the full portable metadata and POWER-source checks, hashes only two selected Tr
sample pairs, and uses fresh disposable random H1/head weights at full 256x256
resolution, FP32, batch 1, seed 0 and two updates. It checks constructor RNG/state,
initial correction/residual/probabilities, gradient opening, first-step weight
decay, frozen H1, prediction RNG isolation, and final source/data hashes. It reads
no checkpoint, V arrays or held-out arrays and writes only a fresh JSON report.
This validates the initialization mechanism, not model quality or training speed.

CPU mechanism tests use synthetic small tensors; they do not replace this real-data
witness or a complete new training run.
