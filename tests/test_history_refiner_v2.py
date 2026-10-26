"""CPU numerical contracts for the proposed event-risk and bounded interventions."""
from copy import deepcopy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from fwt_v2.history_refiner import HistoryRefiner
from fwt_v2.history_refiner_v2 import (
    BoundedHistoryRefiner, build_event_support, correction_diagnostics, mixed_event_bce,
)


def training_samples():
    # Unequal event durations AND unequal mask sizes; one missing-history day
    # remains in event A. This distinguishes pixel-, sample- and event-averaging.
    specs = [('a', 4, 1), ('a', 2, 0), ('a', 1, 1), ('b', 3, 2), ('c', 0, 0)]
    result = []
    for i, (event, count, positive) in enumerate(specs):
        target, mask = torch.full((1, 2, 2), float('nan')), torch.zeros(1, 2, 2)
        mask.view(-1)[:count] = 1
        target.view(-1)[:count] = 0
        target.view(-1)[:positive] = 1
        result.append(dict(event_id=event, sample_id=str(i), target=target, valid_mask=mask,
                           has_history=i != 0))
    return result


class EventLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.samples = training_samples()
        self.support = build_event_support(self.samples)
        self.target = torch.stack([s['target'] for s in self.samples])
        self.mask = torch.stack([s['valid_mask'] for s in self.samples])
        self.ids = [s['event_id'] for s in self.samples]
        self.logits = torch.linspace(-2, 2, 20).reshape(5, 1, 2, 2).requires_grad_()

    def test_complete_training_support_includes_missing_history_and_zero_support_events(self):
        self.assertEqual(self.support['n_samples'], 5)
        self.assertEqual(self.support['n_events'], 2)
        self.assertEqual(self.support['events']['a'], dict(samples=3, valid_pixels=7, positive_pixels=2))
        self.assertEqual(self.support['events']['c']['valid_pixels'], 0)
        self.assertFalse(self.support['history_filter_applied'])
        self.assertEqual(self.support, build_event_support(self.samples))
        with self.assertRaisesRegex(ValueError, 'unique'):
            build_event_support([self.samples[0], self.samples[0]])
        with self.assertRaisesRegex(ValueError, 'residual_train'):
            build_event_support([dict(self.samples[0], role='validation')])

    def test_full_batch_equals_manually_computed_event_mean_and_original_pixel_bce(self):
        result = mixed_event_bce(self.logits, self.target, self.mask, self.ids, self.support)
        event_means = []
        for event in ('a', 'b'):
            rows = torch.tensor([e == event for e in self.ids])
            valid = self.mask[rows].bool()
            event_means.append(F.binary_cross_entropy_with_logits(self.logits[rows][valid], self.target[rows][valid]))
        expected_event = torch.stack(event_means).mean()
        expected_pixel = F.binary_cross_entropy_with_logits(self.logits[self.mask.bool()], self.target[self.mask.bool()])
        torch.testing.assert_close(result['event_bce'], expected_event)
        torch.testing.assert_close(result['pixel_bce'], expected_pixel)
        torch.testing.assert_close(result['total'], (expected_pixel + expected_event) / 2)

    def test_short_batch_weighted_estimates_recover_exact_full_event_objective_and_gradients(self):
        full = mixed_event_bce(self.logits, self.target, self.mask, self.ids, self.support, event_mix=1)['total']
        parts = []
        # End with a batch of size one (event b); zero-support sample stays legal
        # within a supervised batch, as in the complete uniform-sample schedule.
        for indices in ([0, 4], [1, 2], [3]):
            part = mixed_event_bce(self.logits[indices], self.target[indices], self.mask[indices],
                                   [self.ids[i] for i in indices], self.support, event_mix=1)['total']
            parts.append(part * (len(indices) / 5))
        recomposed = sum(parts)
        torch.testing.assert_close(recomposed, full)
        g_full = torch.autograd.grad(full, self.logits, retain_graph=True)[0]
        g_recomposed = torch.autograd.grad(recomposed, self.logits)[0]
        torch.testing.assert_close(g_full, g_recomposed)
        self.assertEqual(float(g_full[~self.mask.bool()].abs().sum()), 0.)

    def test_unknown_labels_ignored_and_labels_never_get_gradients(self):
        target = self.target.clone().requires_grad_()
        result = mixed_event_bce(self.logits, target, self.mask, self.ids, self.support)
        result['total'].backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertEqual(float(self.logits.grad[~self.mask.bool()].abs().sum()), 0.)
        self.assertIsNone(target.grad)

    def test_zero_mix_matches_original_and_rejects_unknown_event_or_empty_batch(self):
        result = mixed_event_bce(self.logits, self.target, self.mask, self.ids, self.support, event_mix=0)
        torch.testing.assert_close(result['total'], result['pixel_bce'])
        with self.assertRaisesRegex(ValueError, 'absent'):
            mixed_event_bce(self.logits, self.target, self.mask, ['new'] * 5, self.support)
        with self.assertRaisesRegex(ValueError, 'no valid'):
            mixed_event_bce(self.logits, self.target, torch.zeros_like(self.mask), self.ids, self.support)
        with self.assertRaises(ValueError):
            mixed_event_bce(self.logits, self.target, self.mask, self.ids, self.support, event_mix=float('nan'))


class BoundedHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.head = BoundedHistoryRefiner(decoder_channels=4, hidden_channels=4, cap=2.)
        self.feature = torch.randn(2, 4, 12, 12, requires_grad=True)
        self.parent = torch.randn(2, 1, 12, 12, requires_grad=True)
        self.delta = torch.ones_like(self.parent)
        self.valid = torch.ones_like(self.parent, dtype=torch.bool)
        self.valid[0] = False
        self.valid[1, :, :2] = False
        self.delta[~self.valid] = float('nan')

    def forward(self, head=None):
        return (head or self.head)(dict(decoder_feature=self.feature), self.parent, self.delta, self.valid)

    def test_parameter_compatibility_zero_initialization_and_local_gradient_match_v1(self):
        original = HistoryRefiner(decoder_channels=4, hidden_channels=4)
        original.load_state_dict(deepcopy(self.head.state_dict()), strict=True)
        self.assertTrue(torch.equal(self.forward(), self.parent))
        self.forward().sum().backward()
        self.forward(original).sum().backward()
        torch.testing.assert_close(self.head.output.weight.grad, original.output.weight.grad)
        self.assertIsNone(self.parent.grad)
        self.assertIsNone(self.feature.grad)

    def test_large_signed_corrections_are_bounded_and_missing_history_is_exact_parent(self):
        for bias in (100., -100., .7):
            with torch.no_grad():
                self.head.output.bias.fill_(bias)
            output = self.forward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(output[~self.valid], self.parent[~self.valid]))
            self.assertLessEqual(float((output - self.parent.detach()).abs().max()), 2. + 1e-6)
        self.assertEqual(self.head.configuration()['correction_cap'], 2.)
        with self.assertRaises(ValueError):
            BoundedHistoryRefiner(cap=0)

    def test_all_missing_history_has_finite_zero_gradients(self):
        self.valid.fill_(False)
        self.delta.fill_(float('nan'))
        output = self.forward()
        self.assertTrue(torch.equal(output, self.parent))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() == 0
                            for p in self.head.parameters()))


class DiagnosticTests(unittest.TestCase):
    @staticmethod
    def record(sid, event, logits, labels, mask):
        return dict(sample_id=sid, event_id=event, calibration_logit=np.array(logits, float),
                    target=np.array(labels, float), valid_mask=np.array(mask, bool))

    def test_per_event_weighting_and_known_drift_without_mutating_records(self):
        parent = [self.record('1', 'a', [0, 0, 0], [1, 0, 0], [1, 1, 1]),
                  self.record('2', 'a', [0, 0, 0], [1, 0, 0], [1, 0, 0]),
                  self.record('3', 'b', [0, 0], [0, 0], [0, 0])]
        records = deepcopy(parent)
        records[0]['calibration_logit'] = np.array([1., -1., -1.])
        records[1]['calibration_logit'] = np.array([2., 100., 100.])
        report = correction_diagnostics(records, parent)
        event = report['events']['a']
        self.assertEqual(event['valid_pixels'], 4)
        self.assertEqual(event['positive_pixels'], 2)
        self.assertEqual(event['correction_abs_mean'], 1.25)
        self.assertEqual(event['correction_abs_max'], 2.)
        self.assertLess(event['nll_change'], 0)
        self.assertLess(event['brier_change'], 0)
        self.assertIsNone(report['events']['b']['nll'])
        self.assertFalse(report['used_for_training_or_checkpoint_selection'])
        self.assertTrue(np.array_equal(parent[0]['calibration_logit'], [0, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'order'):
            correction_diagnostics(records[::-1], parent)

    def test_rejects_malformed_prediction_records_instead_of_reporting_invalid_scores(self):
        row = self.record('1', 'a', [0., 1.], [1., 0.], [1, 1])
        for target in ([2., 0.], [.2, 0.], [float('nan'), 0.]):
            bad = dict(row, target=np.array(target))
            with self.assertRaises(ValueError):
                correction_diagnostics([bad], [bad])
        with self.assertRaises(ValueError):
            correction_diagnostics([row, row], [row, row])
        with self.assertRaises(ValueError):
            correction_diagnostics([dict(row, calibration_logit=np.array([0.]))], [row])


if __name__ == '__main__':
    unittest.main()
