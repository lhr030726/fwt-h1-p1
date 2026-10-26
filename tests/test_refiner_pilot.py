"""CPU-only checks of replay integrity and the independent pilot resume contract."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refiner_pilot", ROOT / "tools/train_refiner_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, optimizer_state_integrity


def replay_fixture():
    ids = [f"event/sample-{i}" for i in range(811)]
    epochs = []
    for epoch in range(10):
        ordered = ids[epoch:] + ids[:epoch]
        steps = [dict(epoch_batch_index=i, update_index=epoch * 51 + i + 1,
                      sample_ids=ordered[i * 16:(i + 1) * 16]) for i in range(51)]
        epochs.append(dict(epoch=epoch, updates=51, diffusion_training_steps=steps))
    return ids, dict(epochs=epochs)


class ReplayTests(unittest.TestCase):
    def test_checkpoint_tuple_metadata_matches_json_lists(self):
        self.assertEqual(pilot.canonical_sha({"nodata_values": (-9999.,), "widths": (16, 32)}),
                         pilot.canonical_sha({"nodata_values": [-9999.], "widths": [16, 32]}))

    def test_replays_original_permutations_and_rejects_identity_corruption(self):
        ids, diagnostics = replay_fixture()
        order = pilot.checked_order(diagnostics, ids)
        self.assertEqual(order[0][-1], ids[800:])
        self.assertEqual(order[7][0], ids[7:23])
        diagnostics["epochs"][4]["diffusion_training_steps"][0]["sample_ids"][0] = ids[-1]
        with self.assertRaisesRegex(ValueError, "duplicated, missing, or foreign"):
            pilot.checked_order(diagnostics, ids)

    def test_rejects_changed_update_order_and_batch_shape(self):
        for field, value in (("epoch_batch_index", 9), ("update_index", 123), ("sample_ids", ["x"])):
            ids, diagnostics = replay_fixture()
            diagnostics["epochs"][2]["diffusion_training_steps"][3][field] = value
            with self.assertRaisesRegex(ValueError, "Replay batch"):
                pilot.checked_order(diagnostics, ids)

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
        state = dict(schema=pilot.SCHEMA, binding={"arm": "bce", "recipe": pilot.RECIPE},
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
            pilot.restore_pilot_state(path, restored, restored_optimizer, state["binding"], "cpu")
            self.step(restored, restored_optimizer)
            self.assertEqual(model_state_digest(reference), model_state_digest(restored))
            self.assertEqual(optimizer_state_integrity(reference_optimizer.state_dict()),
                             optimizer_state_integrity(restored_optimizer.state_dict()))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_resume_rejects_other_arm_and_corrupted_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pth"
            model, optimizer, state = self.make_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "schema/binding"):
                pilot.restore_pilot_state(path, model, optimizer, {"arm": "focal"}, "cpu")
            next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"].add_(1)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "optimizer integrity"):
                pilot.restore_pilot_state(path, model, optimizer, state["binding"], "cpu")

    def test_resume_rejects_wrong_shaped_moments_even_with_updated_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pth"
            model, optimizer, state = self.make_checkpoint(path)
            next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"] = torch.zeros(8)
            state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "moment tensor structure"):
                pilot.restore_pilot_state(path, model, optimizer, state["binding"], "cpu")

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
