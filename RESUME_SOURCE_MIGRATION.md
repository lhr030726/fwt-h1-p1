# Performance source continuation

Use a **new run directory**. Preserve the original source and an immutable copy
of the original `last.pth`. The original full-epoch checkpoint is never rewritten.

The continuation manifest has schema `fwt_performance_continuation_v1`:

```json
{
  "schema": "fwt_performance_continuation_v1",
  "original_source_tree": "/preserved/original/source",
  "original_files": {"train.py": "sha256", "fwt_v2/engine.py": "sha256"},
  "current_files": {"train.py": "sha256", "fwt_v2/engine.py": "sha256"},
  "changed_files": ["fwt_v2/engine.py", "fwt_v2/source_resume.py", "train.py"],
  "original_entry_binding": {},
  "checkpoint": {"path": "/preserved/last.pth", "sha256": "sha256", "phase": "det", "completed_epochs": 11},
  "approved_p1_validation_parallel_samples": 4
}
```

The file maps above are abbreviated. Generate the complete maps using
`source_resume.source_inventory(old_root)` and `source_inventory(new_root)`;
`changed_files` must be the sorted actual difference. `original_entry_binding`
must be the complete original launch receipt binding. Create this manifest only
after reviewing and testing the final merged source bytes. Bind benchmark/test
evidence separately in the deployment receipt.

Both stage commands use `--continuation-manifest FILE` and the unchanged original
scientific flags, including `--h1-epochs 30 --p1-epochs 60`. H1 additionally uses
`--resume-training-state /preserved/last.pth`. Omit fresh `--warm-start-h1-*`
arguments; the original initialization lineage is inherited descriptively from
the full training state. Set `FWT_P1_VALIDATION_PARALLEL_SAMPLES` to exactly the
approved manifest value in both processes. P1 still validates the new run's
completed H1 receipt before starting with a fresh zero-output head.

The engine restores model tensors, AdamW moments and steps, Python/NumPy/CPU/CUDA
RNG, complete histories, global update/epoch counters, and the best selection.
The old total epoch budget stays fixed. All recipe, input identity, runtime,
optimizer and sampling bindings remain exact; only the reviewed source change
and approved validation parallelism are permitted. Historical selected weights
and measurements keep their original metadata in `selection_origin`, including
in `validation_selected.json` if an old best remains selected.

Original source protection includes a file allowlist, exact old/new byte maps,
and AST equality outside the explicitly reviewed engine training/diagnostic/
validation functions and diffusion sampling method. No architecture, loss,
normalization, dataset, diffusion training or evaluation metric change is
permitted by this path. Original hashes are checked again before execution and
completion; runtime source/input checks remain active every epoch.

Validation: six CPU migration tests pass. The main test actually loads the
original engine from git commit 5da517, uses dropout and shuffled batches, and
compares uninterrupted three epochs against one original epoch plus two migrated
epochs. Model/AdamW/RNG states and nine global updates match exactly; an original
best survives, its historical origin remains exposed, and original checkpoint
bytes stay unchanged. Eight prior entry tests and four warm-start tests also
pass. CUDA numerical trajectory equality is not claimed; target-GPU benchmark
and equivalence checks remain deployment prerequisites.

The old running process has no runtime graceful epoch-stop hook. A preserved
completed epoch is resumable; computation after that saved boundary may be
repeated. SIGSTOP alone preserves RAM but cannot apply new Python code and still
occupies GPU memory. Do not edit live source or silently relabel old receipts.
