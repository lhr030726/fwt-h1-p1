"""CPU mechanism tests for a matched deterministic clean-logit control."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig
from fwt_v2.logit_refiner import ConditionalLogitRefiner


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 4, 1)
        self.logit = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        feature = self.feature(x)
        return dict(logits_det=self.logit(feature),
                    fused_pyramid=tuple(F.avg_pool2d(feature, 2**s) for s in range(4)),
                    relational_feature=F.avg_pool2d(feature, 4))


class LogitRefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.config = DiffusionConfig(backbone_feature_channels=(4, 4, 4, 4),
            relational_channels=4, hidden_channels=(4, 4, 4, 4), condition_channels=2,
            prediction_type="clean_probability", training_timestep_sampling="antithetic",
            output_initialization="zero")
        self.backbone = TinyBackbone().eval()
        self.x = torch.randn(2, 3, 16, 16, requires_grad=True)
        self.outputs = self.backbone(self.x)
        self.physics = dict(distance=torch.zeros(2, 1, 16, 16, requires_grad=True))
        self.target = (torch.rand(2, 1, 16, 16) > .7).float()
        self.mask = torch.ones_like(self.target, dtype=torch.bool)
        self.mask[:, :, :3] = False

    def test_same_seed_matches_actual_c_denoiser_and_constructor_rng(self):
        config = DiffusionConfig(prediction_type="clean_probability",
            training_timestep_sampling="antithetic", output_initialization="zero")
        torch.manual_seed(0)
        diffusion = ConditionalResidualDiffusion(config)
        expected_rng = torch.get_rng_state().clone()
        expected = diffusion.denoiser.state_dict()
        for loss in ("bce", "focal"):
            torch.manual_seed(0)
            refiner = ConditionalLogitRefiner(config, loss=loss)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(expected.keys(), refiner.denoiser.state_dict().keys())
            for name, value in refiner.denoiser.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            self.assertEqual(sum(p.numel() for p in refiner.parameters()), 1015761)
            self.assertFalse(list(refiner.named_buffers()))

    def test_loss_arms_load_same_state_and_publish_distinct_metadata(self):
        bce = ConditionalLogitRefiner(self.config, loss="bce")
        focal = ConditionalLogitRefiner(self.config, loss="focal")
        focal.load_state_dict(bce.state_dict(), strict=True)
        self.assertTrue(all(name.startswith("denoiser.") for name in bce.state_dict()))
        self.assertEqual(bce.configuration()["schema"], "fwt_conditional_logit_refiner_v1")
        self.assertEqual(bce.configuration()["loss"], "bce")
        self.assertEqual(focal.configuration()["loss"], "focal")
        self.assertEqual(focal.configuration()["sampling"], "none")
        self.assertEqual(focal.configuration()["residual_input"], "zeros")

    def test_zero_correction_is_exact_h1_and_forward_consumes_no_rng(self):
        head = ConditionalLogitRefiner(self.config)
        outputs = dict(self.outputs)
        outputs["logits_det"] = outputs["logits_det"].detach().clone()
        outputs["logits_det"][0, 0, 0, :2] = torch.tensor([-100., 100.])
        seen = []
        hook = head.denoiser.register_forward_pre_hook(
            lambda module, args: seen.append((args[0].clone(), args[1].clone())))
        before_rng = torch.get_rng_state().clone()
        for training in (False, True):
            head.train(training)
            self.assertTrue(torch.equal(head(outputs, self.physics), outputs["logits_det"]))
            self.assertTrue(torch.equal(head.probability(outputs, self.physics),
                                        outputs["logits_det"].sigmoid()))
        hook.remove()
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(all(int(torch.count_nonzero(r)) == 0 and int(torch.count_nonzero(t)) == 0
                            for r, t in seen))

    def test_masked_bce_and_focal_match_formulas_and_ignore_masked_targets(self):
        for loss in ("bce", "focal"):
            head = ConditionalLogitRefiner(self.config, loss=loss)
            with torch.no_grad():
                head.denoiser.output.bias.fill_(.2)
            logits = head(self.outputs, self.physics)
            expected = F.binary_cross_entropy_with_logits(logits, self.target, reduction="none")
            if loss == "focal":
                p = logits.sigmoid()
                pt = torch.where(self.target > .5, p, 1 - p)
                alpha = torch.where(self.target > .5, .75, .25)
                expected = alpha * (1 - pt).square() * expected
            actual = head.training_loss(self.target, self.outputs, self.physics, self.mask)
            torch.testing.assert_close(actual, expected[self.mask].mean(), rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(head.training_loss(self.target, self.outputs, self.physics),
                                       expected.mean(), rtol=0, atol=0)
            changed = torch.where(self.mask, self.target, 1 - self.target)
            self.assertTrue(torch.equal(actual,
                head.training_loss(changed, self.outputs, self.physics, self.mask)))
            weights = self.mask.float() * .5
            torch.testing.assert_close(head.training_loss(self.target, self.outputs, self.physics, weights),
                                       actual, rtol=0, atol=0)

    def test_gradient_opening_and_every_h1_condition_are_detached(self):
        before_backbone = {name: value.clone() for name, value in self.backbone.state_dict().items()}
        for loss in ("bce", "focal"):
            head = ConditionalLogitRefiner(self.config, loss=loss)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                target = self.target.clone().requires_grad_()
                mask = self.mask.float().requires_grad_()
                value = head.training_loss(target, self.outputs, self.physics, mask)
                value.backward()
                self.assertIsNone(target.grad)
                self.assertIsNone(mask.grad)
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    for p in head.parameters()))
                output_nonzero = sum(int(torch.count_nonzero(p.grad))
                                     for p in head.denoiser.output.parameters())
                upstream_nonzero = sum(int(torch.count_nonzero(p.grad))
                    for name, p in head.named_parameters() if not name.startswith("denoiser.output."))
                self.assertGreater(output_nonzero, 0)
                self.assertEqual(upstream_nonzero, 0) if step == 0 else self.assertGreater(upstream_nonzero, 0)
                optimizer.step()
            self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))
            self.assertIsNone(self.x.grad)
            self.assertIsNone(self.physics["distance"].grad)
            for name, value in self.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, before_backbone[name]), name)

    def test_rejects_invalid_configuration_or_loss(self):
        self.assertEqual(ConditionalLogitRefiner().loss_name, "bce")
        with self.assertRaisesRegex(ValueError, "loss"):
            ConditionalLogitRefiner(self.config, loss="mse")
        with self.assertRaisesRegex(TypeError, "DiffusionConfig"):
            ConditionalLogitRefiner({})
        for config in (DiffusionConfig(), replace(self.config, output_initialization="random")):
            with self.assertRaisesRegex(ValueError, "clean_probability and zero"):
                ConditionalLogitRefiner(config)

    def test_rejects_bad_batch_shapes_and_conditions(self):
        head = ConditionalLogitRefiner(self.config)
        for logits in (torch.zeros(0, 1, 16, 16), torch.zeros(2, 2, 16, 16),
                       torch.zeros(2, 1, 7, 16), torch.zeros(2, 16, 16),
                       torch.zeros(2, 1, 16, 16, dtype=torch.long)):
            with self.assertRaisesRegex(ValueError, "logits_det"):
                head(dict(self.outputs, logits_det=logits), self.physics)
        with self.assertRaisesRegex(ValueError, "mappings"):
            head(None, self.physics)
        with self.assertRaisesRegex(ValueError, "distance"):
            head(self.outputs, {})
        with self.assertRaisesRegex(ValueError, "multiscale"):
            head({"logits_det": self.outputs["logits_det"]}, self.physics)
        for key, bad in (("distance", torch.full((2, 1, 16, 16), float("nan"))),
                         ("distance", torch.zeros(1, 1, 16, 16))):
            with self.assertRaises(ValueError):
                head(self.outputs, dict(self.physics, **{key: bad}))

    def test_rejects_bad_targets_and_masks(self):
        head = ConditionalLogitRefiner(self.config)
        for target in (None, self.target[:1], self.target.double(),
                       torch.full_like(self.target, float("nan")), self.target + 1):
            with self.assertRaisesRegex(ValueError, "target"):
                head.training_loss(target, self.outputs, self.physics, self.mask)
        for mask in (self.mask[:1], torch.zeros_like(self.target),
                     torch.full_like(self.target, float("nan")), torch.full_like(self.target, 2)):
            with self.assertRaisesRegex(ValueError, "mask"):
                head.training_loss(self.target, self.outputs, self.physics, mask)


if __name__ == "__main__":
    unittest.main()
