"""CPU checks of compatibility, frozen-feature gradients, and masked objectives."""
from copy import deepcopy
from pathlib import Path
import sys
from types import MethodType
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2.literature_refiner import DecoderFeatureRefiner, masked_refinement_loss
from fwt_v2.models import FWTConfig, FWTDet


def legacy_decode(self, pyramid):
    """Decoder body from pre-feature-return commit 677e974, as a parity oracle."""
    x = pyramid[-1]
    for index, up in enumerate(self.upsamples):
        skip = pyramid[-2 - index]
        x_up = up(x, skip.shape[-2:])
        attended = self.attentions[index](x_up, skip)
        x = self.decoders[index](torch.cat([attended, x_up], dim=1))
    return self.Conv_1x1(x)


class LiteratureRefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        config = FWTConfig(schema_version="rcda22-semantic21-v3", widths=(2, 4, 8, 16),
            decoder_widths=(4, 8, 16, 32), prop_channels=4, propagation_steps=1,
            propagation_radius=1, normalization="group", norm_max_groups=2,
            use_wind_prior=False, extra_weather_channels=4)
        self.backbone = FWTDet(config).eval()
        self.x = torch.randn(2, 22, 16, 16, requires_grad=True)
        self.physics = dict(distance=torch.zeros(2, 1, 16, 16, requires_grad=True),
            wind_xy=torch.zeros(2, 2, 16, 16), slope=torch.ones(2, 1, 16, 16),
            upslope_xy=torch.zeros(2, 2, 16, 16))

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        else:
            self.assertEqual(actual, expected)

    def test_default_h1_matches_legacy_decoder_and_strict_state_loading(self):
        legacy = deepcopy(self.backbone)
        legacy.decode = MethodType(legacy_decode, legacy)
        original_state = deepcopy(legacy.state_dict())
        self.backbone.load_state_dict(original_state, strict=True)
        self.assertEqual(original_state.keys(), self.backbone.state_dict().keys())
        before_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            expected = legacy(self.x, self.physics)
            actual = self.backbone(self.x, self.physics)
            exposed = self.backbone(self.x, self.physics, return_decoder_features=True)
        self.assert_nested_equal(actual, expected)
        feature = exposed.pop("decoder_feature")
        self.assert_nested_equal(exposed, expected)
        self.assertEqual(tuple(feature.shape), (2, 4, 16, 16))
        self.assertTrue(torch.equal(self.backbone.Conv_1x1(feature), expected["logits_det"]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assert_nested_equal(self.backbone.state_dict(), original_state)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.backbone(self.x, self.physics, return_decoder_features="yes")

    def test_default_head_capacity_and_schema_are_explicit(self):
        head = DecoderFeatureRefiner()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 28449)
        self.assertEqual(head.configuration()["schema"], "fwt_decoder_feature_refiner_v1")
        self.assertEqual(head.configuration()["condition_order"],
                         ["decoder_feature", "logits_det", "distance"])
        self.assertEqual(head.configuration()["sampling"], "none")
        for kwargs in ({"decoder_channels": 0}, {"hidden_channels": True}):
            with self.assertRaises(ValueError):
                DecoderFeatureRefiner(**kwargs)

    def test_zero_initial_head_is_exact_h1_in_train_and_eval_without_rng(self):
        outputs = self.backbone(self.x, self.physics, return_decoder_features=True)
        outputs["logits_det"] = outputs["logits_det"].detach().clone()
        outputs["logits_det"][0, 0, 0, :2] = torch.tensor([-100., 100.])
        head = DecoderFeatureRefiner(decoder_channels=4, hidden_channels=8)
        before_rng = torch.get_rng_state().clone()
        for training in (True, False):
            head.train(training)
            self.assertTrue(torch.equal(head(outputs, self.physics), outputs["logits_det"]))
            self.assertTrue(torch.equal(head.probability(outputs, self.physics),
                                        outputs["logits_det"].sigmoid()))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_two_updates_open_upstream_gradients_but_never_touch_h1(self):
        before = deepcopy(self.backbone.state_dict())
        # Deliberately leave H1/physics trainable: head detachment is independently tested.
        outputs = self.backbone(self.x, self.physics, return_decoder_features=True)
        outputs["decoder_feature"].retain_grad()
        outputs["logits_det"].retain_grad()
        head = DecoderFeatureRefiner(decoder_channels=4, hidden_channels=8)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        target = (torch.rand(2, 1, 16, 16) > .8).float().requires_grad_()
        mask = torch.ones_like(target, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_refinement_loss(head(outputs, self.physics), target, mask, aux_weight=.1)
            loss["total"].backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in head.parameters()))
            upstream = sum(int(torch.count_nonzero(p.grad)) for p in head.features.parameters())
            self.assertEqual(upstream, 0) if step == 0 else self.assertGreater(upstream, 0)
            self.assertGreater(int(torch.count_nonzero(head.output.weight.grad)), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))
        self.assertIsNone(outputs["decoder_feature"].grad)
        self.assertIsNone(outputs["logits_det"].grad)
        self.assertIsNone(self.x.grad)
        self.assertIsNone(self.physics["distance"].grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(mask.grad)
        self.assert_nested_equal(self.backbone.state_dict(), before)

    def test_head_rejects_missing_or_incompatible_conditions(self):
        outputs = self.backbone(self.x, self.physics, return_decoder_features=True)
        head = DecoderFeatureRefiner(decoder_channels=4, hidden_channels=8)
        for bad in (None, {"logits_det": outputs["logits_det"]},
                    dict(outputs, decoder_feature=outputs["decoder_feature"][:, :2]),
                    dict(outputs, logits_det=outputs["logits_det"].double())):
            with self.assertRaises(ValueError):
                head(bad, self.physics)
        with self.assertRaisesRegex(ValueError, "distance"):
            head(outputs, dict(self.physics, distance=torch.full_like(self.physics["distance"], 2.)))


class RefinementLossTests(unittest.TestCase):
    def test_masked_loss_matches_hand_computed_counts_and_is_batch_pooled(self):
        probabilities = torch.tensor([.8, .4, .6, .1], dtype=torch.float64).reshape(2, 1, 1, 2)
        logits = torch.logit(probabilities).requires_grad_()
        target = torch.tensor([1., 0., 1., 0.], dtype=torch.float64).reshape_as(logits)
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool).reshape_as(logits)
        result = masked_refinement_loss(logits, target, mask, aux_weight=.2)
        expected_bce = (-torch.log(torch.tensor(.8, dtype=torch.float64))
                        - torch.log(torch.tensor(.6, dtype=torch.float64))
                        - torch.log(torch.tensor(.9, dtype=torch.float64))) / 3
        expected_tversky = 1 - (.8 + 1e-6) / (.8 + .3 * .5 + .7 * .2 + 1e-6)
        self.assertAlmostEqual(result["bce"].item(), expected_bce.item(), places=14)
        self.assertAlmostEqual(result["tversky"].item(), expected_tversky, places=14)
        self.assertAlmostEqual(result["total"].item(), expected_bce.item() + .2 * expected_tversky, places=14)
        result["total"].backward()
        self.assertEqual(logits.grad.flatten()[2].item(), 0.)

    def test_masked_unknown_labels_have_no_effect_and_bce_only_matches_pytorch(self):
        logits = torch.tensor([-.4, .7, 4., -8.], dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
        target = torch.tensor([0., 1., float("nan"), -9999.], dtype=torch.float64).reshape_as(logits)
        mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool).reshape_as(logits)
        result = masked_refinement_loss(logits, target, mask)
        expected = F.binary_cross_entropy_with_logits(logits.flatten()[:2], target.flatten()[:2])
        self.assertTrue(torch.equal(result["total"], expected))
        replacement = torch.where(mask, target, torch.zeros_like(target))
        changed = masked_refinement_loss(logits, replacement, mask.float(), aux_weight=.3)
        unchanged = masked_refinement_loss(logits, target, mask, aux_weight=.3)
        self.assertTrue(torch.equal(changed["total"], unchanged["total"]))
        unchanged["total"].backward()
        self.assertTrue(torch.equal(logits.grad.flatten()[2:], torch.zeros(2, dtype=torch.float64)))

    def test_empty_positive_auxiliary_is_differentiable_zero_and_bce_still_learns(self):
        logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)
        mask = torch.ones_like(logits, dtype=torch.bool)
        result = masked_refinement_loss(logits, target, mask, aux_weight=.3)
        self.assertEqual(result["tversky"].item(), 0.)
        auxiliary_grad = torch.autograd.grad(result["tversky"], logits, retain_graph=True)[0]
        self.assertEqual(int(torch.count_nonzero(auxiliary_grad)), 0)
        result["total"].backward()
        self.assertTrue(bool((logits.grad > 0).all()))
        self.assertTrue(torch.equal(result["total"], result["bce"]))

    def test_tversky_fp_fn_gradient_matches_analytic_derivative(self):
        probabilities = torch.tensor([.2, .8], dtype=torch.float64)
        logits = torch.logit(probabilities).reshape(1, 1, 1, 2).requires_grad_()
        target = torch.tensor([1., 0.], dtype=torch.float64).reshape_as(logits)
        result = masked_refinement_loss(logits, target, torch.ones_like(logits), aux_weight=.1)
        actual = torch.autograd.grad(result["tversky"], logits)[0].flatten()
        numerator = .2 + 1e-6
        denominator = .2 + .3 * .8 + .7 * .8 + 1e-6
        # Positive derivative: denominator changes by (1-beta_fn); negative by alpha_fp.
        d_positive = -(denominator - numerator * (1 - .7)) / denominator**2
        d_negative = numerator * .3 / denominator**2
        expected = torch.tensor([d_positive * .2 * .8, d_negative * .8 * .2], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
        self.assertLess(actual[0].item(), 0.)
        self.assertGreater(actual[1].item(), 0.)
        # More FN weight gives a stronger FN-vs-FP gradient ratio at this witness.
        balanced = masked_refinement_loss(logits, target, torch.ones_like(logits), alpha_fp=.5, beta_fn=.5)
        balanced_grad = torch.autograd.grad(balanced["tversky"], logits)[0].flatten()
        self.assertGreater(abs(actual[0] / actual[1]), abs(balanced_grad[0] / balanced_grad[1]))

    def test_total_loss_passes_double_precision_finite_difference_gradient_check(self):
        logits = torch.tensor([-.8, .3, 1.2, -.1], dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
        target = torch.tensor([1., 0., 1., 0.], dtype=torch.float64).reshape_as(logits)
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool).reshape_as(logits)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: masked_refinement_loss(z, target, mask, aux_weight=.1)["total"],
            (logits,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_invalid_supervision_and_hyperparameters_are_rejected(self):
        logits = torch.zeros(1, 1, 2, 2)
        target = torch.zeros_like(logits)
        mask = torch.ones_like(logits)
        for bad in (torch.zeros_like(mask), torch.full_like(mask, .5), torch.full_like(mask, float("nan"))):
            with self.assertRaisesRegex(ValueError, "mask"):
                masked_refinement_loss(logits, target, bad)
        for bad in (target.double(), torch.full_like(target, float("nan")), target + .5):
            with self.assertRaisesRegex(ValueError, "target"):
                masked_refinement_loss(logits, bad, mask)
        for kwargs in ({"aux_weight": -1.}, {"alpha_fp": float("nan")},
                       {"beta_fn": True}, {"alpha_fp": 0., "beta_fn": 0.}):
            with self.assertRaises(ValueError):
                masked_refinement_loss(logits, target, mask, **kwargs)


if __name__ == "__main__":
    unittest.main()
