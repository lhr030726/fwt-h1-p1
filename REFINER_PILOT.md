The paired refiner pilot adds a deterministic clean-logit control without changing any existing H1/P1 implementation or the completed C run.

`fwt_v2/logit_refiner.py` reuses P1's 1,015,761-parameter conditional UNet. It receives frozen H1 features, probability and current-fire distance, with zero residual input and timestep zero. The zero-initialized output adds a correction to H1 logits. BCE and focal arms share exact initial weights, original C per-batch Tr sample order, optimizer, and ten-epoch budget. Focal uses positive alpha 0.75 and gamma 2. The comparison between the two arms changes only the loss; comparison to stochastic diffusion is not compute-matched.

`tools/train_refiner_pilot.py` prints a read-only plan unless `--execute` is supplied. All output must be in a NEW directory beneath this checkout. It binds the completed C source, H1 checkpoint, original data bundle/index, Tr811/V568 identities, target/mask bytes and replay order. Original sources and checkpoints are never changed. Full-V H1 AP must reproduce within 1e-6 before training starts. The existing runtime and dataset are reused.

The pilot uses its own checkpoint schema. These checkpoints are not accepted by the existing formal `load_predictor` or `train.py` entry. `best.pth` and `last.pth` contain full state at their actual saved epochs. Final selection is authorized by `validation_selected.json` and the completed receipt, not by relabeling an early snapshot as a ten-epoch training state. Immutable selected-epoch files preserve consistency if a process stops between best and last saves. `--resume-run` restores a completed epoch into another new output directory under the unchanged ten-epoch recipe.

Validation is development data already used to select and tune models. Any improvement here is a candidate for subsequent independent evaluation, not a held-out generalization result. Zero-epoch H1 is reported as a separate baseline and is not silently added to either arm's checkpoint selection.

CPU mechanism and state tests:

```text
python -B -m unittest discover -s tests -p "test_*refiner*.py" -v
```

Required paths and the resolved plan are recorded by the deployment wrapper. Run `python -B tools/train_refiner_pilot.py --help` for the explicit input flags. No environment installation or automatic external notification is performed.
