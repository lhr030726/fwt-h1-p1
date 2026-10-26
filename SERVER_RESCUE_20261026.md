# Training rescue: 2026-10-26

The exact running training source is commit `574322e8c798e0a70cbfa74a949d12bb5ea906d4`. This follow-up only adds recovery documentation, scripts, and verification records.

- [Download verified checkpoints, logs, frozen parent models, and provenance](https://github.com/lhr030726/fwt-h1-p1/releases/tag/rescue-history-refiner-20261026).
- [Recovery instructions (Chinese)](migration/20261026/RECOVERY.md).
- Restore the base archive, then the final progress overlay, and the weather evidence archive.
- Stable paired progress: seeds **4 = 50/50**, **17 = 50/50**, **29 = 35/50** for both arms. Training was intentionally paused for migration.
- Raw arrays remain in the user's original dataset; all 2,758 required Tr/V arrays match the server SHA256 values.
- The relocation adapter passed three CPU checks. Full resume on a different GPU has not yet been tested.

GitHub stores the artifacts; it does not supply a GPU or continue the training process.
