"""CPU mechanism checks; synthetic inputs, no training-quality claim."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2.diffusion import (ConditionalResidualDiffusion, DiffusionConfig,
                             DeterministicResidualHead, compute_probability)
from scripts.check_zero_init_real import verify_gradients


class TinyFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 4, 1)
        self.logit = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        feature = self.feature(x)
        return dict(logits_det=self.logit(feature),
                    fused_pyramid=tuple(F.avg_pool2d(feature, 2**s) for s in range(4)),
                    relational_feature=F.avg_pool2d(feature, 4))


class ZeroInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(0)
        self.config = DiffusionConfig(backbone_feature_channels=(4, 4, 4, 4),
            relational_channels=4, hidden_channels=(4, 4, 4, 4), condition_channels=2,
            prediction_type='clean_probability', training_timestep_sampling='antithetic')
        self.backbone = TinyFrozenBackbone().requires_grad_(False).eval()
        self.x = torch.randn(2, 3, 16, 16, requires_grad=True)
        self.outputs = self.backbone(self.x)
        self.physics = dict(distance=torch.zeros(2, 1, 16, 16))

    def test_default_and_legacy_configs_preserve_random_constructor_and_state(self):
        legacy = asdict(self.config)
        legacy.pop('output_initialization')
        torch.manual_seed(91)
        original = ConditionalResidualDiffusion(DiffusionConfig(**legacy))
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        explicit = ConditionalResidualDiffusion(replace(self.config, output_initialization='random'))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        # Existing parameter/buffer keys remain strictly loadable with the new default.
        explicit.load_state_dict(original.state_dict(), strict=True)
        self.assertGreater(int(torch.count_nonzero(original.denoiser.output.weight)), 0)

    def test_zero_changes_only_output_and_preserves_constructor_rng(self):
        torch.manual_seed(37)
        random = ConditionalResidualDiffusion(self.config)
        random_rng = torch.get_rng_state().clone()
        torch.manual_seed(37)
        zero = ConditionalResidualDiffusion(replace(self.config, output_initialization='zero'))
        self.assertTrue(torch.equal(random_rng, torch.get_rng_state()))
        self.assertEqual(random.state_dict().keys(), zero.state_dict().keys())
        for name, value in zero.state_dict().items():
            if name in ('denoiser.output.weight', 'denoiser.output.bias'):
                self.assertEqual(int(torch.count_nonzero(value)), 0)
            else:
                self.assertTrue(torch.equal(value, random.state_dict()[name]), name)

    def test_invalid_variants_and_non_diffusion_head_rejected(self):
        with self.assertRaisesRegex(ValueError, 'random or zero'):
            replace(self.config, output_initialization='unknown')
        for prediction in ('epsilon', 'v'):
            with self.assertRaisesRegex(ValueError, 'requires clean_probability'):
                replace(self.config, prediction_type=prediction, output_initialization='zero')
        with self.assertRaisesRegex(ValueError, 'only supported by the diffusion head'):
            DeterministicResidualHead(replace(self.config, output_initialization='zero'))

    def test_real_witness_gradient_gate_accepts_zero_upstream_only_on_first_update(self):
        output, upstream = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))
        parameters = [('denoiser.output.weight', output), ('denoiser.encoders.0.weight', upstream)]
        output.grad, upstream.grad = torch.ones_like(output), torch.zeros_like(upstream)
        first = verify_gradients(parameters, 1, torch)
        self.assertEqual(first['upstream_nonzero_gradient_elements'], 0)
        with self.assertRaisesRegex(RuntimeError, 'gradient-opening'):
            verify_gradients(parameters, 2, torch)
        upstream.grad.fill_(1)
        self.assertGreater(verify_gradients(parameters, 2, torch)['total_gradient_l2'], 0)
        with self.assertRaisesRegex(RuntimeError, 'gradient-opening'):
            verify_gradients(parameters, 1, torch)
        output.grad.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'Missing/nonfinite'):
            verify_gradients(parameters, 2, torch)

    def test_zero_correction_clean_residual_and_final_ddim_including_saturation(self):
        head = ConditionalResidualDiffusion(replace(self.config, output_initialization='zero')).eval()
        outputs = dict(self.outputs)
        outputs['logits_det'] = self.outputs['logits_det'].detach().clone()
        outputs['logits_det'][0, 0, 0, :2] = torch.tensor([-100., 100.])
        conditions = head.prepare_conditions(outputs, self.physics)
        rt = torch.randn_like(outputs['logits_det'])
        for step, previous in ((199, 133), (133, 66), (66, 0), (0, -1)):
            t = torch.full((2,), step, dtype=torch.long)
            correction = head.denoiser(rt, t, conditions)
            self.assertEqual(int(torch.count_nonzero(correction)), 0)
            clean = head.predict_clean_residual(rt, t, correction, outputs['logits_det'])
            self.assertEqual(int(torch.count_nonzero(clean)), 0)
            rt = head.ddim_step(rt, t, torch.full_like(t, previous), correction, outputs['logits_det'])
            self.assertTrue(bool(torch.isfinite(rt).all()))
        self.assertEqual(int(torch.count_nonzero(rt)), 0)
        before_rng = torch.get_rng_state().clone()
        samples = head.sample(outputs, self.physics, K=8, steps=4,
                              generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(int(torch.count_nonzero(samples)), 0)
        # Reduction can introduce roundoff even when all K residuals are zero.
        torch.testing.assert_close(compute_probability(outputs, samples),
                                   outputs['logits_det'].sigmoid(), atol=1e-7, rtol=0)

    def test_two_updates_bce_gradient_opening_weight_decay_and_frozen_backbone(self):
        head = ConditionalResidualDiffusion(replace(self.config, output_initialization='zero'))
        initial_backbone = {k: v.clone() for k, v in self.backbone.state_dict().items()}
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        target = (torch.rand(2, 1, 16, 16) > .7).float()
        mask = torch.ones_like(target)
        mask[:, :, :3] = 0
        t = torch.tensor([0, 199])
        noise = torch.randn_like(target)
        upstream = {name: p for name, p in head.named_parameters()
                    if not name.startswith('denoiser.output.')}
        initial_upstream = {name: p.detach().clone() for name, p in upstream.items()}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = head.training_loss(target, self.outputs, self.physics, mask=mask,
                                      timesteps=t, noise=noise)
            self.assertTrue(bool(torch.isfinite(loss)))
            if step == 0:
                expected = F.binary_cross_entropy_with_logits(
                    self.outputs['logits_det'].detach(), target, reduction='none')
                self.assertTrue(torch.equal(loss.detach(), (expected * mask).sum() / mask.sum()))
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in head.parameters()))
            self.assertGreater(sum(int(torch.count_nonzero(p.grad))
                                   for p in head.denoiser.output.parameters()), 0)
            nonzero = sum(int(torch.count_nonzero(p.grad)) for p in upstream.values())
            self.assertEqual(nonzero, 0) if step == 0 else self.assertGreater(nonzero, 0)
            optimizer.step()
            if step == 0:
                for name, p in upstream.items():
                    self.assertTrue(torch.equal(p, initial_upstream[name] * (1 - .001 * .01)), name)
                self.assertTrue(any(not torch.equal(p, initial_upstream[name])
                                    for name, p in upstream.items()))
            self.assertIsNone(self.x.grad)  # Conditions detach even with a grad-enabled input.
            self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))
            for name, value in self.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, initial_backbone[name]), name)


if __name__ == '__main__':
    unittest.main()
