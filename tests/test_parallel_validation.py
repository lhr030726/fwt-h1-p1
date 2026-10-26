"""CPU sampler checks; GPU performance/numerics require a separate real benchmark."""
import hashlib
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fwt_v2.diffusion import ConditionalResidualDiffusion, DiffusionConfig
from fwt_v2.engine import raw_probability, validation_records
from fwt_v2.validation_sampling import validation_parallel_samples


class FixedBackbone(nn.Module):
    def forward(self, x, physics):
        return dict(logits_det=x[:, :1],
                    fused_pyramid=[x[:, :2, ::2**s, ::2**s] for s in range(4)],
                    relational_feature=x[:, :2, ::4, ::4])


def batch_fixture(count=5):
    x = torch.linspace(-1, 1, count*2*8*8).reshape(count, 2, 8, 8)
    shape = (count, 1, 8, 8)
    return dict(x_norm=x, physics={'distance': torch.full(shape, .25)},
                sample_id=['fire-a/2020-01-%02d' % i for i in range(count)],
                event_id=['fire-a']*count, target=torch.zeros(shape),
                valid_mask=torch.ones(shape, dtype=torch.bool),
                domain_mask=torch.ones(shape, dtype=torch.bool))


def subset(batch, indices):
    return {key: ({k: v[indices] for k, v in value.items()} if key == 'physics'
                 else [value[i] for i in indices] if isinstance(value, list)
                 else value[indices]) for key, value in batch.items()}


class ParallelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)
        self.head = ConditionalResidualDiffusion(DiffusionConfig(
            timesteps=4, backbone_feature_channels=(2, 2, 2, 2),
            relational_channels=2, hidden_channels=(2, 2, 2, 2),
            condition_channels=2, prediction_type='clean_probability')).eval()
        self.backbone = FixedBackbone().eval()
        self.batch = batch_fixture()
        self.sampling = dict(K=8, steps=4, method='ddim', seed=19)

    def predict(self, parallel, batch=None, sampling=None):
        return raw_probability(self.backbone, self.head, batch or self.batch,
                               sampling or self.sampling, return_logits=True,
                               parallel_samples=parallel)

    def test_real_sampler_parallel_output_noise_order_and_tail(self):
        captured = []
        handle = self.head.denoiser.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].clone()))
        try:
            serial, serial_logits = self.predict(1)
            serial_noise = torch.cat(captured[::4])
            for width in (2, 4, 8):
                captured.clear()
                actual, logits = self.predict(width)
                self.assertTrue(torch.equal(serial_noise, torch.cat(captured[::4])))
                self.assertEqual([t.shape[0] for t in captured[::4]],
                                 [min(width, 5-i)*8 for i in range(0, 5, width)])
                torch.testing.assert_close(actual, serial, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(logits, serial_logits, rtol=2e-5, atol=2e-6)
        finally:
            handle.remove()

    def test_original_sample_id_noise_formula(self):
        captured = []
        handle = self.head.denoiser.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].clone()))
        try:
            self.predict(4)
        finally:
            handle.remove()
        expected = []
        for sample_id in self.batch['sample_id']:
            seed = int.from_bytes(hashlib.sha256(f'19:{sample_id}'.encode()).digest()[:8],
                                  'big') % (2**63-1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            expected.append(torch.randn((8, 1, 8, 8), generator=generator))
        self.assertTrue(torch.equal(torch.cat(expected), torch.cat(captured[::4])))

    def test_global_and_caller_generator_rng_are_unchanged(self):
        caller = torch.Generator(device='cpu').manual_seed(333)
        caller_before = caller.get_state().clone()
        global_before = torch.get_rng_state().clone()
        raw_probability(self.backbone, self.head, self.batch, self.sampling,
                        generator=caller, parallel_samples=4)
        self.assertTrue(torch.equal(caller_before, caller.get_state()))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_reorder_and_loader_boundaries_keep_sample_predictions(self):
        expected, _ = self.predict(1)
        order = [4, 2, 0, 3, 1]
        actual, _ = self.predict(3, subset(self.batch, order))
        torch.testing.assert_close(actual, expected[order], rtol=2e-5, atol=2e-6)
        records = validation_records(self.backbone, self.head,
            [subset(self.batch, [0, 1, 2]), subset(self.batch, [3, 4])], 'cpu',
            self.sampling, parallel_samples=2)
        self.assertEqual([r['sample_id'] for r in records], self.batch['sample_id'])
        torch.testing.assert_close(torch.stack([torch.from_numpy(r['probability']) for r in records]),
                                   expected, rtol=2e-5, atol=2e-6)

    def test_ddpm_falls_back_to_original_serial_stream(self):
        sampling = dict(self.sampling, method='ddpm')
        original = self.head.sample
        with patch.object(self.head, 'sample', wraps=original) as sample:
            expected, _ = self.predict(1, sampling=sampling)
            actual, _ = self.predict(4, sampling=sampling)
            self.assertEqual(sample.call_count, 10)
            self.assertTrue(all('sample_generators' not in call.kwargs for call in sample.call_args_list))
        self.assertTrue(torch.equal(expected, actual))

    def test_clamp_per_draw_before_mean(self):
        batch = batch_fixture(3)
        batch['x_norm'].fill_(torch.logit(torch.tensor(.2)))
        def residual(outputs, physics, K, **kwargs):
            samples = torch.tensor([-1., 1.] * 4).reshape(1, K, 1, 1, 1)
            return samples.expand(outputs['logits_det'].shape[0], K, 1, 8, 8)
        with patch.object(self.head, 'sample', side_effect=residual):
            actual, _ = self.predict(2, batch)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, .5)))

    def test_default_is_serial_and_invalid_controls_fail(self):
        expected, _ = self.predict(1)
        actual = raw_probability(self.backbone, self.head, self.batch, self.sampling)
        self.assertTrue(torch.equal(expected, actual))
        for invalid in (0, -1, True, 1.5, '2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.predict(invalid)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(validation_parallel_samples(), 1)
        for text in ('2', '4', '8'):
            with patch.dict(os.environ, FWT_P1_VALIDATION_PARALLEL_SAMPLES=text):
                self.assertEqual(validation_parallel_samples(), int(text))
        for text in ('0', '-1', '1.5', 'true', '', ' 2', '２'):
            with patch.dict(os.environ, FWT_P1_VALIDATION_PARALLEL_SAMPLES=text):
                with self.assertRaises(ValueError):
                    validation_parallel_samples()

    def test_sample_generators_reject_ambiguous_or_unsupported_streams(self):
        outputs = self.backbone(self.batch['x_norm'], self.batch['physics'])
        generators = [torch.Generator(device='cpu').manual_seed(i) for i in range(5)]
        for options in (dict(method='ddpm'), dict(generator=generators[0]),
                        dict(sample_generators=generators[:1]),
                        dict(sample_generators=[None]*5)):
            kwargs = dict(K=8, steps=4, sample_generators=generators)
            kwargs.update(options)
            with self.assertRaises(ValueError):
                self.head.sample(outputs, self.batch['physics'], **kwargs)


if __name__ == '__main__':
    unittest.main()
