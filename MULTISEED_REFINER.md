# Fixed BCE refiner confirmation runs

`tools/train_refiner_multiseed.py` trains **fresh seeds 1, 2, 3**, sequentially,
with the selected original C H1 frozen. Each run uses the selected direct
conditional logit refiner architecture, BCE, FP32, AdamW learning rate `1e-4`,
weight decay `0.01`, batch size 16, and exactly 10 epochs / 510 updates. Validation
batch size is 8. Historical seed 0 is preserved as an exploratory reference and
is not a fourth confirmation run.

Only refiner initialization and Tr sample permutations vary across seeds. Each
seed explicitly resets Python/NumPy/PyTorch RNG, constructs and loads the same
H1 (preserving the historical constructor RNG convention), then constructs its
own zero-output refiner. A separate CPU PyTorch generator supplies each seed's
ten complete 811-sample permutations without advancing global RNG. All three
plans are frozen before execution; their hashes and initial model hashes must
differ. Zero initial correction preserves the initial H1 predictions.

This measures refiner stability **conditional on one selected H1**, not complete
end-to-end model seed variability. CUDA nondeterministic algorithms remain
permitted as in the historical recipe, with cuDNN benchmarking disabled and the
actual runtime/backend settings recorded.

## Source and data preservation

The old pilot runner, old checkpoints, all existing `fwt_v2` modules, and
`train.py` are unchanged. The runner verifies the original C completed receipts,
H1 checkpoint/source, development-only portable data bundle, normalization,
weather manifest, exact Tr/V sample order, and published input/label array
hashes. Full-V H1 AP must reproduce the original value within `1e-6` before any
refiner updates. Original source/input hashes, file identities, and frozen H1
state/absence of gradients are rechecked through training.

Data validation, the complete H1 baseline, and a shared 16 GiB CPU tensor cache
are prepared once for all three runs. Training uses exactly the original 811 Tr
samples. The original 568-sample V alone chooses each seed's earliest maximum
pooled-AP checkpoint. **No independent or held-out arrays are opened by this
entry.** Independent evaluation is a separate stage, after all selected weights
and development thresholds have been frozen. Test scores must not choose a seed,
checkpoint, threshold, or further training recipe.

## Invocation

Supply the same immutable development/H1/C evidence paths used by the paired
pilot. The required flags are:

```text
--manifest --normalizer --backbone --h1-validation --h1-receipt
--c-diff-configuration --c-diff-diagnostics --c-diff-receipt
--reference-source --output
```

Without `--execute`, the command verifies metadata and emits a read-only plan;
it does not construct datasets, initialize CUDA, or create output. Plan generation
uses CPU PyTorch to construct the frozen permutations. `--execute` requires an
idle, single visible CUDA GPU and a **new** output directory within the deployed
repository (the existing artifact helpers enforce workspace containment).

`--resume-run OLD_RUN --output NEW_RUN --execute` restores each available seed's
last complete epoch into a fresh directory, including parameters, AdamW moments,
RNG, history, and selected-weight references. Source, input paths/bytes, recipe,
seeds, and batch plans must match exactly. Interrupted seeds without a saved
epoch begin from the same reproducible initialization; completed seeds are
copied and reverified. The budget remains ten epochs and old artifacts are never
overwritten. This is full-state continuation within this schema, not an import
of the historical diffusion optimizer or a conversion to `fwt_checkpoint_v2`.

## Output and evaluation contract

The root contains `plan.json`, `baseline.json`, `input-binding.json`,
`status.json`, `resource-history.jsonl`, and a final `receipt.json`. Resource
snapshots contain process RSS/peak RSS, cgroup memory/current limit, GPU usage,
and GPU memory. Per-epoch checks also contain peak torch GPU allocated/reserved
bytes, finite/nonzero gradient checks, observed batch-order hash, training time,
and training-plus-validation/metrics time. Timing excludes integrity checks and
checkpoint writes. Resource telemetry failure is recorded but does not change
training.

Each `seed-001`, `seed-002`, and `seed-003` directory contains:

- `initial-head.pth`, `configuration.json`, `history.json`, and `diagnostics.json`;
- `best.pth`, `last.pth`, and immutable `selected-epoch-NNN.pth` snapshots;
- final `validation_selected.json` and `receipt.json`.

Training state schema is `fwt_logit_refiner_multiseed_state_v1`. Each state binds
the seed, fixed recipe/source/input/array hashes, training/V identities,
validation target/mask digest, initial head digest, batch-plan hash, H1 state
digest, head configuration, and optimizer settings. Epoch fields inside states
and selected reports are zero based; filenames and status epochs are one based.
The selected state contains its actual completed-epoch history prefix; only the
separate completed receipt certifies the full ten-epoch selection budget.

Per-seed receipt schema is `fwt_logit_refiner_seed_completed_v1`; it records seed,
`completed_epochs=10`, `total_updates=510`, earliest `best_epoch`, selected/last
validation AP, immutable selected filename, and all checkpoint/report hashes.
Root receipt schema is `fwt_logit_refiner_multiseed_completed_v1` and has
`seeds={"1": receipt, "2": receipt, "3": receipt}`, per-seed initial-state and
batch-plan hash maps, H1 baseline AP, and the common input-binding hash.

## Focused verification

`tests/test_refiner_multiseed.py` checks independent/reproducible seed plans and
initializations, complete sample coverage, no global RNG consumption by planning,
zero initial correction, original array-byte binding, busy-GPU rejection,
cross-seed restore rejection, optimizer shape/integrity checks, exact next-step
AdamW/RNG recovery, interruption between best/last publication, and protection of
existing output. The existing eight refiner model tests remain applicable.
