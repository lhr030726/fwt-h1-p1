"""CPU-only independent seed/order and full-state recovery mechanism checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refiner_multiseed", ROOT / "tools/train_refiner_multiseed.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, optimizer_state_integrity


class SeedOrderTests(unittest.TestCase):
    def test_checkpoint_tuple_metadata_matches_json_lists(self):
        self.assertEqual(pilot.canonical_sha({"nodata_values": (-9999.,), "widths": (16, 32)}),
                         pilot.canonical_sha({"nodata_values": [-9999.], "widths": [16, 32]}))

    def test_independent_orders_reproduce_without_advancing_global_rng(self):
        ids = [f"event/sample-{i}" for i in range(811)]
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        hashes = []
        for seed in (1, 2, 3):
            order = pilot.seeded_order(seed, ids)
            self.assertEqual(order, pilot.seeded_order(seed, ids))
            self.assertEqual(len(order), 10)
            self.assertEqual(len(order[0]), 51)
            self.assertEqual(len(order[0][-1]), 11)
            self.assertEqual(len({pilot.canonical_sha(epoch) for epoch in order}), 10)
            hashes.append(pilot.canonical_sha(order))
        self.assertEqual(len(set(hashes)), 3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rejects_invalid_seeds_ids_and_changed_batch_shape(self):
        ids = [f"event/sample-{i}" for i in range(811)]
        for seed in (0, 4, True, "1"):
            with self.assertRaisesRegex(ValueError, "confirmation seeds"):
                pilot.seeded_order(seed, ids)
        for altered in (ids[:-1], ids[:-1] + [ids[0]]):
            with self.assertRaisesRegex(ValueError, "811 unique"):
                pilot.seeded_order(1, altered)
        order = pilot.seeded_order(1, ids)
        order[2][3][0] = order[2][3][1]
        with self.assertRaisesRegex(ValueError, "duplicated, missing, or foreign"):
            pilot.validate_order(order, ids)
        order = pilot.seeded_order(2, ids)
        order[4][5].pop()
        with self.assertRaisesRegex(ValueError, "batch shape"):
            pilot.validate_order(order, ids)

    def test_seeded_initializations_differ_reproduce_and_start_with_zero_correction(self):
        from fwt_v2.diffusion import DiffusionConfig
        config = DiffusionConfig(backbone_feature_channels=(4, 4, 4, 4),
            relational_channels=4, hidden_channels=(4, 4, 4, 4), condition_channels=2,
            prediction_type="clean_probability", output_initialization="zero")
        fixed = torch.nn.Linear(3, 2)
        state = {key: value.clone() for key, value in fixed.state_dict().items()}
        digest = model_state_digest(fixed)
        def fake_load(*args):
            model = torch.nn.Linear(3, 2)
            model.load_state_dict(state)
            return model, {}
        hashes = []
        with patch("fwt_v2.engine.load_det", side_effect=fake_load):
            for seed in (1, 2, 3):
                head, actual, rng = pilot.initialize_head(seed, "unused", "cpu", config, digest)
                repeated, expected, repeat_rng = pilot.initialize_head(seed, "unused", "cpu", config, digest)
                self.assertEqual(actual, expected)
                self.assertEqual(head.loss_name, "bce")
                self.assertEqual(int(torch.count_nonzero(head.denoiser.output.weight)), 0)
                self.assertEqual(int(torch.count_nonzero(head.denoiser.output.bias)), 0)
                self.assertTrue(torch.equal(rng["torch_cpu"], repeat_rng["torch_cpu"]))
                hashes.append(actual)
            with self.assertRaisesRegex(RuntimeError, "different fixed H1"):
                pilot.initialize_head(1, "unused", "cpu", config, "wrong")
        self.assertEqual(len(set(hashes)), 3)

    def test_busy_or_unparseable_gpu_query_fails_closed(self):
        from types import SimpleNamespace
        for value in ("12345\n", "N/A\n"):
            with patch.object(pilot.subprocess, "run", return_value=SimpleNamespace(stdout=value)):
                with self.assertRaises(RuntimeError):
                    pilot.assert_gpu_idle()
        with patch.object(pilot.subprocess, "run", return_value=SimpleNamespace(stdout="")):
            pilot.assert_gpu_idle()

    def test_actual_array_bytes_must_match_original_c_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {field: root / field for field in ("input_path", "label_path")}
            for field, path in paths.items():
                path.write_bytes(field.encode())
            sample = dict(sample_id="one", role="residual_train", **{k: str(p) for k, p in paths.items()})
            index = dict(schema_version="fwt_portable_array_index_v1", files=[
                dict(sample_id="one", field=field, role="residual_train", path="data/" + field,
                     size=path.stat().st_size, sha256=pilot.sha(path)) for field, path in paths.items()])
            hashes = {str(path): pilot.sha(path) for path in paths.values()}
            self.assertEqual(pilot.verify_selected_array_index(index, [sample], hashes, root), 2)
            paths["input_path"].write_bytes(b"mutated")
            hashes[str(paths["input_path"])] = pilot.sha(paths["input_path"])
            with self.assertRaisesRegex(ValueError, "differ from original C"):
                pilot.verify_selected_array_index(index, [sample], hashes, root)


class ResumeTests(unittest.TestCase):
    def make_checkpoint(self, path):
        torch.manual_seed(19)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        for _ in range(51):
            self.step(model, optimizer)
        checks = dict(epoch=0, updates=51, cumulative_updates=51)
        state = dict(schema=pilot.SCHEMA, binding={"seed": 1, "loss": "bce", "recipe": pilot.RECIPE},
            completed_epochs=1, updates=51, model_state_dict=pilot.tensor_cpu_tree(model.state_dict()),
            model_state_sha256=model_state_digest(model),
            optimizer_state_dict=pilot.tensor_cpu_tree(optimizer.state_dict()), rng_state=capture_rng("cpu"),
            history=[dict(epoch=0, training_checks=checks)],
            diagnostics=dict(completed_epochs=1, epochs=[checks]))
        state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
        torch.save(state, path)
        return model, optimizer, state

    @staticmethod
    def step(model, optimizer):
        x, target = torch.randn(4, 3), torch.randn(4, 1)
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        optimizer.step()

    def test_full_resume_matches_the_next_adamw_and_rng_step(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pth"
            reference, reference_optimizer, state = self.make_checkpoint(path)
            self.step(reference, reference_optimizer)
            expected_rng = torch.get_rng_state().clone()
            restored = torch.nn.Linear(3, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=.01)
            pilot.restore_multiseed_state(path, restored, restored_optimizer, state["binding"], "cpu")
            self.step(restored, restored_optimizer)
            self.assertEqual(model_state_digest(reference), model_state_digest(restored))
            self.assertEqual(optimizer_state_integrity(reference_optimizer.state_dict()),
                             optimizer_state_integrity(restored_optimizer.state_dict()))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_resume_rejects_other_seed_and_corrupted_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pth"
            model, optimizer, state = self.make_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "schema/binding"):
                pilot.restore_multiseed_state(path, model, optimizer, dict(state["binding"], seed=2), "cpu")
            next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"].add_(1)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "optimizer integrity"):
                pilot.restore_multiseed_state(path, model, optimizer, state["binding"], "cpu")

    def test_resume_rejects_wrong_shaped_moments_even_with_updated_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pth"
            model, optimizer, state = self.make_checkpoint(path)
            next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"] = torch.zeros(8)
            state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "moment tensor structure"):
                pilot.restore_multiseed_state(path, model, optimizer, state["binding"], "cpu")

    def test_previous_full_epoch_survives_new_best_before_last_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = root / "selected-epoch-001.pth"
            previous.write_bytes(b"old complete selected checkpoint")
            state = dict(selected_epoch=0, selected_checkpoint_file=previous.name,
                         selected_checkpoint_sha256=pilot.sha(previous))
            # Simulate interruption after new best publication but before last.
            (root / "selected-epoch-002.pth").write_bytes(b"new selected checkpoint")
            pilot.atomic_copy(root / "selected-epoch-002.pth", root / "best.pth")
            self.assertEqual(pilot.selected_snapshot_path(root, state), previous)
            restored = root / "copied-best.pth"
            pilot.atomic_copy(previous, restored)
            self.assertEqual(pilot.sha(restored), state["selected_checkpoint_sha256"])

    def test_existing_output_is_not_modified_when_execution_rejects_it(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            existing = output / "history.json"
            existing.write_text("original")
            args = SimpleNamespace(output=output, execute=True)
            with patch.object(pilot, "parse_args", return_value=args), patch.object(pilot, "make_plan", return_value={}), \
                    patch.object(pilot, "execute", side_effect=FileExistsError("existing output")):
                with self.assertRaises(FileExistsError):
                    pilot.main([])
            self.assertEqual(existing.read_text(), "original")
            self.assertEqual(list(output.iterdir()), [existing])


if __name__ == "__main__":
    unittest.main()
