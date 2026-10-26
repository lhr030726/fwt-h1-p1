# Additional H1 segment and extended fresh P1

The previous zero-output, batch-16 run selected H1 epoch 29 (AP 0.229085),
while P1 selected epoch 6 (AP 0.205108) and later fluctuated. This experimental
recipe lowers both learning rates to 0.0001 and spends more updates on training.
It is a development comparison, not a guarantee of improved held-out quality.

H1 starts from the **exact previous selected checkpoint weights** and trains
30 additional epochs with a fresh AdamW optimizer and seed 0. Optimizer moments
and RNG state are not resumed. The original run completed 30 epochs, but the
selected weights came from epoch 29; this is not described as an exact 60-epoch
resume. The new segment selects its best epoch among its 30 trained epochs;
the original checkpoint remains an independent preserved reference.

P1 is then freshly initialized with a zero output layer and trained for 60
epochs against the new segment's selected, frozen H1. Reusing an old P1 head
would change its conditioning backbone, so this recipe deliberately starts a
new P1 optimization. Batch 16, validation batch 8, FP32, cache 40 GiB, CPU 2,
seed 0, AdamW weight decay 0.01, all objectives and sampling are unchanged.

Both commands must supply the same recipe and lineage flags:

```text
--warm-start-h1-run ORIGINAL_RUN
--warm-start-h1-source-tree ORIGINAL_REPO
--h1-learning-rate 0.0001 --p1-learning-rate 0.0001
--h1-epochs 30 --p1-epochs 60 --p1-zero-init
--batch-size 16 --validation-batch-size 8 --sample-cache-gib 40
```

H1 uses `--phase det --epochs 30`; P1 uses `--phase diff --epochs 60` after H1
fully completes. `--execute` is necessary for each stage. Preflight remains
read-only and imports no torch/numpy. Original seven completion artifacts are
hash checked, all data/model identities and selection rules are checked, and
source compatibility requires every package module except `engine.py` and
`backbone_compat.py` to remain byte-identical. Engine setup and every function
except `train` and the new `load_warm_start_h1` helper must retain the same AST.
Original source is inspected as text only; no original code is executed.

The engine verifies checkpoint tensor shape, dtype, finiteness and exact loaded
values before training, records the initial state hash and fresh optimizer
evidence, and preserves the original checkpoint. P1 verifies its newly selected
H1 remains frozen throughout training. There is no fabricated H1 completion
receipt, no checkpoint source rebinding, and no held-out evaluation.
