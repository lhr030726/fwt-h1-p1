"""CPU contracts: past-only input pairing, burn context and frozen-parent branch."""
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fwt_v2.data import DataConfig
from fwt_v2.history_refiner import HistoryRefiner, build_history_pairs, load_history_arrays


def row(day, *, role="residual_train", event="2020_UID_1", shape=(8, 8)):
    return dict(sample_id=f"{event}/{day}", event_id=event, date=day, role=role,
        group_id="group1", georeference=dict(crs="EPSG:32610",
        affine_transform=[50., 0., 0., 0., -50., 0.], shape=list(shape)))


def inputs(batch=2):
    return (dict(decoder_feature=torch.randn(batch, 4, 12, 12, requires_grad=True)),
        torch.randn(batch, 1, 12, 12, requires_grad=True),
        torch.zeros(batch, 1, 12, 12), torch.ones(batch, 1, 12, 12, dtype=torch.bool))


class HistoryPairTests(unittest.TestCase):
    def test_exact_past_date_order_and_event(self):
        samples = [row("2020-01-03"), row("2020-01-01"), row("2020-01-04"),
                   row("2020-01-02", event="2020_UID_2")]
        indices, audit = build_history_pairs(samples)
        self.assertEqual(indices, [None, None, 0, None])
        self.assertFalse(audit["arrays_read"])
        self.assertEqual(audit["paired_samples"], 1)

    def test_role_group_crs_affine_shape_mismatch_and_missing_metadata(self):
        mutations = [lambda r: r.update(role="validation"),
                     lambda r: r.update(group_id="other"), lambda r: r.pop("group_id"),
                     lambda r: r["georeference"].update(crs="EPSG:32611"),
                     lambda r: r["georeference"].update(affine_transform=[50., 0., 1., 0., -50., 0.]),
                     lambda r: r["georeference"].update(shape=[9, 8]),
                     lambda r: r["georeference"].pop("crs")]
        for mutate in mutations:
            samples = [row("2020-01-01"), row("2020-01-02")]
            mutate(samples[0])
            with self.subTest(samples=samples):
                indices, audit = build_history_pairs(samples)
                self.assertEqual(indices, [None, None])
                self.assertEqual(audit["paired_samples"], 0)

    def test_duplicate_identity_and_malformed_date_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_history_pairs([row("2020-01-01"), row("2020-01-01")])
        with self.assertRaises(ValueError):
            build_history_pairs([row("not-a-date")])

    def test_loader_never_reads_label_keeps_burned_context_and_negative_change(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = [row("2020-01-01"), row("2020-01-02"), row("2020-01-04")]
            arrays = [np.zeros((12, 8, 8), np.float32) for _ in samples]
            arrays[0][0, 1, 1] = 1.  # Observed burn disappearance remains -1.
            arrays[1][0, 2, 2] = 1.  # New burned context must not be domain-masked.
            arrays[1][0, 2, 3] = .5  # Same strict > .5 burn definition.
            arrays[0][1, 3, 3] = -9999.
            arrays[1][0, 3, 3] = 1.
            arrays[1][2, 4, 4] = np.nan
            for i, (sample, raw) in enumerate(zip(samples, arrays)):
                path = Path(directory) / f"input{i}.npy"
                np.save(path, raw)
                sample["input_path"] = str(path.resolve())
                sample["label_path"] = "FORBIDDEN_LABEL"
            config = DataConfig(wind_convention="grid_towards_ccw_from_east",
                wind_reference="projected_grid", wind_metadata_review="unit-test declared convention")
            dataset = SimpleNamespace(samples=samples, config=config, role="residual_train")
            original_load, accessed = np.load, []

            def guarded_load(path, *args, **kwargs):
                accessed.append(str(path))
                self.assertIn(str(path), [s["input_path"] for s in samples])
                self.assertIs(kwargs["allow_pickle"], False)
                return original_load(path, *args, **kwargs)

            with patch("fwt_v2.history_refiner.np.load", side_effect=guarded_load):
                result = load_history_arrays(dataset)
            self.assertEqual(accessed, [s["input_path"] for s in samples])
            self.assertEqual(result["has_history"].tolist(), [False, True, False])
            self.assertEqual(result["history_delta"][1, 0, 2, 2], 1.)
            self.assertTrue(result["history_valid"][1, 0, 2, 2])
            self.assertEqual(result["history_delta"][1, 0, 1, 1], -1.)
            self.assertEqual(result["history_delta"][1, 0, 2, 3], 0.)
            self.assertFalse(result["history_valid"][1, 0, 3, 3])
            self.assertFalse(result["history_valid"][1, 0, 4, 4])
            self.assertEqual(result["history_delta"][1, 0, 3, 3], 0.)
            self.assertFalse(result["history_valid"][[0, 2]].any())
            self.assertEqual(result["audit"]["label_arrays_read"], 0)
            self.assertEqual(result["audit"]["totals"]["positive_change_pixels"], 1)
            self.assertEqual(result["audit"]["totals"]["negative_change_pixels"], 1)

    def test_loader_refuses_unresolved_paths_or_nondevelopment_role(self):
        sample = row("2020-01-01")
        sample["input_path"] = "relative.npy"
        dataset = SimpleNamespace(samples=[sample], config=None, role="residual_train")
        with self.assertRaisesRegex(ValueError, "absolute"):
            load_history_arrays(dataset)
        dataset.role = "final"
        with self.assertRaises(PermissionError):
            load_history_arrays(dataset)


class HistoryModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_zero_output_equals_parent_and_only_output_layer_zero(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        data = inputs()
        data[2][:, :, 4:8, 4:8] = 1.
        self.assertTrue(torch.equal(model(*data), data[1]))
        self.assertTrue(bool(model.features[0].weight.detach().abs().sum() > 0))
        self.assertEqual(int(torch.count_nonzero(model.output.weight)), 0)

    def test_gradients_reach_branch_after_zero_start_but_not_parent_or_feature(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        data = inputs()
        data[2][:, :, 4:8, 4:8] = 1.
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        (model(*data) - 2.).square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(model.features[0].weight.grad.abs().sum()), 0.)
        self.assertIsNone(data[0]["decoder_feature"].grad)
        self.assertIsNone(data[1].grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (model(*data) - 2.).square().mean().backward()
        self.assertGreater(float(model.features[0].weight.grad.abs().sum()), 0.)

    def test_unavailable_pixel_fallback_even_after_training_and_unknown_delta_ignored(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        data = inputs()
        data[3][0] = False
        data[3][1, :, :3, :3] = False
        data[2][~data[3]] = float("nan")
        with torch.no_grad():
            model.output.bias.fill_(3.)
            model.output.weight.normal_()
        output = model(*data)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[~data[3]], data[1][~data[3]]))
        self.assertFalse(torch.equal(output[data[3]], data[1][data[3]]))

    def test_all_missing_history_batch_has_finite_backward_with_zero_branch_gradients(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        data = inputs()
        data[3].fill_(False)
        data[2].fill_(float("nan"))
        with torch.no_grad():
            model.output.weight.normal_()
            model.output.bias.fill_(1.)
        output = model(*data)
        self.assertTrue(torch.equal(output, data[1]))
        output.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_history_changes_can_influence_unburned_neighbors(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        data = inputs(batch=1)
        with torch.no_grad():
            model.output.weight.fill_(1.)
        zero_prediction = model(*data)
        data[2][0, 0, 5, 5] = 1.  # Currently burned input, no target mask enters branch.
        changed_prediction = model(*data)
        # Neighbor remains unburned but receives the changed context through the branch.
        self.assertGreater(float((changed_prediction - zero_prediction)[0, 0, 5, 6].abs()), 1e-6)

    def test_state_roundtrip_and_same_state_zero_history_control(self):
        history = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        control = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        control.load_state_dict(deepcopy(history.state_dict()))
        data = inputs()
        with torch.no_grad():
            history.output.weight.normal_()
        buffer = io.BytesIO()
        torch.save(history.state_dict(), buffer)
        buffer.seek(0)
        control.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=False))
        self.assertTrue(torch.equal(history(*data), control(*data)))
        self.assertEqual(history.configuration(), control.configuration())

    def test_mask_requires_bool_and_shapes_match(self):
        model = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        outputs, logits, delta, valid = inputs()
        with self.assertRaisesRegex(ValueError, "bool"):
            model(outputs, logits, delta, valid.float())
        with self.assertRaisesRegex(ValueError, "history_delta"):
            model(outputs, logits, delta[..., :-1], valid)


if __name__ == "__main__":
    unittest.main()
