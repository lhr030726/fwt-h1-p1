"""CPU contracts for an explicit best49 fork; no parent source is patched."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compact_continuation', ROOT / 'tools/continue_compact_refiner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, restore_rng, optimizer_state_integrity


def step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = torch.tensor([[.7, -.3, .4], [.2, .8, -.5]])
    target = torch.tensor([[.3], [-.6]])
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def parent_fixture():
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    step(model, optimizer)
    # The fixture has the real AdamW tensor structure and inherited step count;
    # the production validator additionally binds the original checkpoint SHA.
    for item in optimizer.state.values():
        item['step'].fill_(2499)
    history = [dict(epoch=i, validation={'pooled_ap': .2 + .001 * i}) for i in range(49)]
    binding = dict(seed=4, arm='compact_bce', source_files={'frozen.py': 'original'})
    state = dict(schema=runner.extension.SCHEMA, binding=binding, completed_epochs=49,
        epoch=48, updates=2499, history=history, model_state_dict=runner.tensor_cpu_tree(model.state_dict()),
        model_state_sha256=model_state_digest(model), optimizer_state_dict=runner.tensor_cpu_tree(optimizer.state_dict()),
        rng_state=capture_rng('cpu'), validation_ap=history[-1]['validation']['pooled_ap'])
    state['optimizer_integrity'] = optimizer_state_integrity(state['optimizer_state_dict'])
    return state, binding


def fork_fixture():
    parent, parent_binding = parent_fixture()
    heads, optimizers, bindings, migrations = {}, {}, {}, {}
    baseline = dict(pooled_ap=.26, nll=.1, brier=.03)
    states = {}
    for arm in runner.ARMS:
        head = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
        runner.restore_parent_state(parent, head, optimizer, parent_binding, 'cpu', parent['history'][:10])
        migrations[arm] = runner.migrate_optimizer_lr(optimizer, arm)
        binding = dict(branch_id=arm, architecture_arm='compact_bce', seed=4,
            parent_checkpoint_sha256='original-best49', optimizer_param_groups=deepcopy(optimizer.state_dict()['param_groups']))
        heads[arm], optimizers[arm], bindings[arm] = head, optimizer, binding
        states[arm] = runner.make_state(head, optimizer, binding, [], baseline, 'cpu')
    transaction = runner.build_transaction(states, {a: runner.serialize_state(s) for a, s in states.items()},
                                            {a: None for a in runner.ARMS}, capture_rng('cpu'))
    return parent, heads, optimizers, bindings, migrations, transaction


def advance_epoch(heads, optimizers, bindings, transaction, ap=.2):
    decoded = runner.decode_transaction(transaction)
    states, best, new_best = {}, {}, {}
    for arm in runner.ARMS:
        for _ in range(51):
            step(heads[arm], optimizers[arm])
        previous = decoded[arm]['last']
        n = previous['completed_child_epochs'] + 1
        history = deepcopy(previous['history'])
        history.append(dict(child_epoch=n, absolute_epoch=49 + n, lifetime_updates=2499 + 51 * n,
            child_updates=51 * n, learning_rate=runner.LEARNING_RATES[arm],
            validation=dict(pooled_ap=ap, nll=.11, brier=.035)))
        state = runner.make_state(heads[arm], optimizers[arm], bindings[arm], history,
                                   previous['baseline_validation'], 'cpu')
        states[arm] = state
        best[arm] = (runner.serialize_state(state) if ap > decoded[arm]['best']['validation']['pooled_ap']
                      else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best'))
        new_best[arm] = (runner.serialize_state(state) if 'new_best' not in decoded[arm]
            or ap > decoded[arm]['new_best']['validation']['pooled_ap']
            else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'new_best'))
    return runner.build_transaction(states, best, new_best, capture_rng('cpu'))


class ContinuationTests(unittest.TestCase):
    def test_development_candidate_gate_filter_margin_and_ties(self):
        def receipts(high_ap, low_ap, high_mean=.2, low_mean=.2, high_ok=True, low_ok=True):
            return {arm: dict(last_three_mean_ap=mean, new_best=dict(validation={'pooled_ap': ap},
                qualifies_development_replacement_gate=ok, checkpoint_sha256=arm + '-new-best'))
                for arm, ap, mean, ok in (('constant_lr', high_ap, high_mean, high_ok),
                                          ('lower_lr', low_ap, low_mean, low_ok))}
        result = runner.select_development_candidate(receipts(.3, .4, high_ok=False, low_ok=False))
        self.assertEqual(result['decision'], 'retained_parent')
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['checkpoint_sha256'], runner.PARENT_BEST_SHA)
        result = runner.select_development_candidate(receipts(.9, .3, high_ok=False))
        self.assertEqual(result['candidate'], 'lower_lr')
        # Use exactly the comparison literal at the boundary, avoiding a test
        # that silently imposes decimal rounding absent from the frozen rule.
        result = runner.select_development_candidate(receipts(.001, 0., high_mean=0., low_mean=1.))
        self.assertEqual(result['candidate'], 'constant_lr')
        result = runner.select_development_candidate(receipts(.000999, 0., high_mean=0., low_mean=1.))
        self.assertEqual(result['candidate'], 'lower_lr')
        self.assertEqual(runner.select_development_candidate(receipts(.3, .3))['candidate'], 'lower_lr')
        self.assertEqual(runner.select_development_candidate(receipts(.3, .3, .4, .2))['candidate'], 'constant_lr')
        self.assertTrue(result['development_only'])
        self.assertFalse(result['independent_evaluation'])

    def test_order_exact_fifty_prefix_and_59_budget_without_rng_consumption(self):
        ids = [f'sample-{i}' for i in range(811)]
        ten = runner.pilot.seeded_order(4, ids)
        fifty = runner.extension.extended_order(ids, ten)
        before = torch.get_rng_state().clone()
        result = runner.continuation_order(ids, fifty)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(result[:50], fifty)
        self.assertEqual(len(result), 59)
        self.assertEqual(result[49], fifty[49])
        self.assertEqual(sum(len(e) for e in result[49:59]), 510)
        bad = deepcopy(fifty)
        bad[49][0][0], bad[49][0][1] = bad[49][0][1], bad[49][0][0]
        with self.assertRaisesRegex(ValueError, 'first fifty'):
            runner.continuation_order(ids, bad)
        with self.assertRaisesRegex(ValueError, 'fifty'):
            runner.continuation_order(ids, fifty[:49])

    def test_full_moments_step_rng_preserved_only_lr_migrates(self):
        parent, heads, optimizers, bindings, migrations, transaction = fork_fixture()
        for arm in runner.ARMS:
            self.assertEqual(model_state_digest(heads[arm]), parent['model_state_sha256'])
            saved = runner.tensor_cpu_tree(optimizers[arm].state_dict())
            self.assertTrue(all(v['step'].item() == 2499 for v in saved['state'].values()))
            for group in saved['param_groups']:
                self.assertEqual(group['lr'], runner.LEARNING_RATES[arm])
                group['lr'] = 1e-4
            self.assertEqual(optimizer_state_integrity(saved), parent['optimizer_integrity'])
            self.assertEqual(migrations[arm]['source_optimizer_integrity'], parent['optimizer_integrity'])
        for left, right in zip(heads['constant_lr'].parameters(), heads['lower_lr'].parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        for l, r in zip(optimizers['constant_lr'].state.values(), optimizers['lower_lr'].state.values()):
            self.assertNotEqual(l['exp_avg'].data_ptr(), r['exp_avg'].data_ptr())
            self.assertNotEqual(l['step'].data_ptr(), r['step'].data_ptr())
        restore_rng(parent['rng_state'], 'cpu')
        expected = (random.random(), np.random.random(), torch.rand(3))
        runner.restore_transaction(transaction, heads, optimizers, bindings, 'cpu')
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_real_updates_diverge_with_preserved_adamw_steps(self):
        _, heads, optimizers, _, _, _ = fork_fixture()
        self.assertEqual(model_state_digest(heads['constant_lr']), model_state_digest(heads['lower_lr']))
        for arm in runner.ARMS:
            self.assertTrue(np.isfinite(step(heads[arm], optimizers[arm])))
            self.assertTrue(all(v['step'].item() == 2500 for v in optimizers[arm].state.values()))
            if arm == 'constant_lr':
                self.assertTrue(all(v['step'].item() == 2499 for v in optimizers['lower_lr'].state.values()))
        self.assertNotEqual(model_state_digest(heads['constant_lr']), model_state_digest(heads['lower_lr']))

    def test_restore_ownership_even_if_optimizer_retains_input_tensor_references(self):
        # Emulate the ownership behavior seen on the remote PyTorch 2.7 runtime
        # even when this test runs on a version whose loader deep-copies inputs.
        original_load = torch.optim.AdamW.load_state_dict

        def retaining_load(optimizer, saved):
            result = original_load(optimizer, saved)
            for serialized, group in zip(saved['param_groups'], optimizer.param_groups):
                for identity, parameter in zip(serialized['params'], group['params']):
                    if identity in saved['state']:
                        optimizer.state[parameter] = saved['state'][identity]
            return result

        with patch.object(torch.optim.AdamW, 'load_state_dict', retaining_load):
            parent, heads, optimizers, bindings, _, transaction = fork_fixture()
            parent_integrity = optimizer_state_integrity(parent['optimizer_state_dict'])
            parent_tensors = deepcopy(parent['optimizer_state_dict']['state'])
            decoded = runner.restore_transaction(transaction, heads, optimizers, bindings, 'cpu')
            snapshot = runner.make_state(heads['constant_lr'], optimizers['constant_lr'],
                bindings['constant_lr'], [], decoded['constant_lr']['last']['baseline_validation'], 'cpu')
            snapshot_integrity = optimizer_state_integrity(snapshot['optimizer_state_dict'])
            decoded_integrity = {arm: {kind: optimizer_state_integrity(state['optimizer_state_dict'])
                for kind, state in states.items()} for arm, states in decoded.items()}
            other_integrity = optimizer_state_integrity(optimizers['lower_lr'].state_dict())
            for arm in runner.ARMS:
                live = optimizers[arm].state_dict()['state']
                for identity, tensors in live.items():
                    for key, tensor in tensors.items():
                        self.assertNotEqual(tensor.data_ptr(), parent['optimizer_state_dict']['state'][identity][key].data_ptr())
                        self.assertNotEqual(tensor.data_ptr(), decoded[arm]['last']['optimizer_state_dict']['state'][identity][key].data_ptr())
                        if arm == 'constant_lr':
                            self.assertNotEqual(tensor.data_ptr(), snapshot['optimizer_state_dict']['state'][identity][key].data_ptr())
            for left, right in zip(optimizers['constant_lr'].state.values(), optimizers['lower_lr'].state.values()):
                for key in left:
                    self.assertNotEqual(left[key].data_ptr(), right[key].data_ptr())
            step(heads['constant_lr'], optimizers['constant_lr'])
            self.assertTrue(all(v['step'].item() == 2500 for v in optimizers['constant_lr'].state.values()))
            self.assertTrue(all(v['step'].item() == 2499 for v in optimizers['lower_lr'].state.values()))
            self.assertEqual(optimizer_state_integrity(optimizers['lower_lr'].state_dict()), other_integrity)
            self.assertEqual(optimizer_state_integrity(parent['optimizer_state_dict']), parent_integrity)
            self.assertEqual(optimizer_state_integrity(snapshot['optimizer_state_dict']), snapshot_integrity)
            for identity, tensors in parent_tensors.items():
                for key, tensor in tensors.items():
                    self.assertTrue(torch.equal(tensor, parent['optimizer_state_dict']['state'][identity][key]))
            for arm, states in decoded.items():
                for kind, state in states.items():
                    self.assertEqual(optimizer_state_integrity(state['optimizer_state_dict']), decoded_integrity[arm][kind])
            # Reusing the same authoritative transaction must still restore an
            # unmodified 2499-step checkpoint, including on a second resume.
            runner.restore_transaction(transaction, heads, optimizers, bindings, 'cpu')
            for arm in runner.ARMS:
                self.assertTrue(all(v['step'].item() == 2499 for v in optimizers[arm].state.values()))
                step(heads[arm], optimizers[arm])
                self.assertTrue(all(v['step'].item() == 2500 for v in optimizers[arm].state.values()))

    def test_parent_fallback_and_new_only_best_are_distinct(self):
        _, heads, optimizers, bindings, _, transaction = fork_fixture()
        transaction = advance_epoch(heads, optimizers, bindings, transaction, ap=.2)
        decoded = runner.decode_transaction(transaction)
        for arm in runner.ARMS:
            self.assertEqual(decoded[arm]['best']['completed_child_epochs'], 0)
            self.assertEqual(decoded[arm]['new_best']['completed_child_epochs'], 1)
            self.assertEqual(decoded[arm]['last']['lifetime_updates'], 2550)
        transaction = advance_epoch(heads, optimizers, bindings, transaction, ap=.3)
        transaction = advance_epoch(heads, optimizers, bindings, transaction, ap=.3)
        for states in runner.decode_transaction(transaction).values():
            self.assertEqual(states['best']['completed_child_epochs'], 2)
            self.assertEqual(states['new_best']['completed_child_epochs'], 2)
            self.assertEqual(states['last']['child_updates'], 153)
            self.assertEqual(states['last']['lifetime_updates'], 2652)

    def test_reject_single_arm_wrong_epoch_broken_optimizer_and_lr(self):
        _, heads, optimizers, bindings, _, transaction = fork_fixture()
        wrong = deepcopy(transaction)
        del wrong['arms']['lower_lr']
        with self.assertRaisesRegex(ValueError, 'arms'):
            runner.decode_transaction(wrong)
        wrong = deepcopy(transaction)
        wrong['completed_child_epochs'] = 1
        with self.assertRaises(ValueError):
            runner.decode_transaction(wrong)
        decoded = runner.decode_transaction(transaction)
        last = {arm: deepcopy(value['last']) for arm, value in decoded.items()}
        best = {a: runner.pilot.transaction_checkpoint_bytes(transaction['arms'][a], 'best') for a in runner.ARMS}
        key = next(iter(last['lower_lr']['optimizer_state_dict']['state']))
        last['lower_lr']['optimizer_state_dict']['state'][key]['exp_avg'].add_(1)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            runner.build_transaction(last, best, {a: None for a in runner.ARMS}, capture_rng('cpu'))
        wrong = deepcopy(decoded['constant_lr']['last'])
        wrong['learning_rate'] = 1e-5
        with self.assertRaisesRegex(ValueError, 'LR'):
            runner.check_state(wrong, 'constant_lr')

    def test_resume_restores_next_real_update_and_rejects_changed_binding(self):
        _, heads, optimizers, bindings, _, transaction = fork_fixture()
        transaction = advance_epoch(heads, optimizers, bindings, transaction)
        for arm in runner.ARMS:
            step(heads[arm], optimizers[arm])
        expected_model = {a: model_state_digest(h) for a, h in heads.items()}
        expected_opt = {a: optimizer_state_integrity(runner.tensor_cpu_tree(o.state_dict())) for a, o in optimizers.items()}
        runner.restore_transaction(transaction, heads, optimizers, bindings, 'cpu')
        for arm in runner.ARMS:
            step(heads[arm], optimizers[arm])
            self.assertEqual(model_state_digest(heads[arm]), expected_model[arm])
            self.assertEqual(optimizer_state_integrity(runner.tensor_cpu_tree(optimizers[arm].state_dict())), expected_opt[arm])
        changed = deepcopy(bindings)
        changed['lower_lr']['parent_checkpoint_sha256'] = 'different-parent'
        with self.assertRaisesRegex(ValueError, 'binding'):
            runner.restore_transaction(transaction, heads, optimizers, changed, 'cpu')

    def test_exact_ten_epoch_budget_finishes_at_lifetime3009(self):
        _, heads, optimizers, bindings, _, transaction = fork_fixture()
        for _ in range(10):
            transaction = advance_epoch(heads, optimizers, bindings, transaction)
        self.assertEqual(transaction['completed_child_epochs'], 10)
        for arm, states in runner.decode_transaction(transaction).items():
            self.assertEqual(states['last']['child_updates'], 510)
            self.assertEqual(states['last']['lifetime_updates'], 3009)
            self.assertEqual(states['last']['absolute_completed_epochs'], 59)
            self.assertTrue(all(v['step'].item() == 3009 for v in optimizers[arm].state.values()))
        invalid = deepcopy(transaction)
        invalid['completed_child_epochs'] = 11
        with self.assertRaisesRegex(ValueError, 'progress'):
            runner.decode_transaction(invalid)

    def test_transaction_recovers_incomplete_mirrors(self):
        _, heads, optimizers, bindings, _, transaction = fork_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            for arm in runner.ARMS:
                (directory / arm).mkdir()
            runner.publish_transaction(directory, transaction)
            updated = advance_epoch(heads, optimizers, bindings, transaction)
            with patch.object(runner.pilot, 'atomic_bytes', side_effect=OSError('interrupted mirror')):
                with self.assertRaisesRegex(OSError, 'mirror'):
                    runner.publish_transaction(directory, updated)
            committed = torch.load(directory / 'transaction.pth', map_location='cpu', weights_only=True)
            self.assertEqual(committed['completed_child_epochs'], 1)
            runner.restore_transaction(committed, heads, optimizers, bindings, 'cpu')
            runner.publish_transaction(directory, committed)
            for arm in runner.ARMS:
                self.assertEqual(runner.sha(directory / arm / 'last.pth'), committed['arms'][arm]['last_sha256'])
                self.assertTrue((directory / arm / 'new_best.pth').is_file())

    def test_raw_ap_mask_class_proper_scores_and_full_parent_drift(self):
        z = np.array([100., 90., -100., -90., 999.], dtype=np.float32)
        y = np.array([1., 0., 1., 0., 0.], dtype=np.float32)
        mask = np.array([1, 1, 1, 1, 0], dtype=bool)
        p = torch.from_numpy(z).sigmoid().numpy()
        row = dict(sample_id='one', event_id='event', probability=p, calibration_logit=z,
            parent_calibration_logit=z - 2., target=y, valid_mask=mask, domain_mask=np.ones(5, bool))
        metric = runner.extended_metrics([row])
        self.assertEqual(metric['n_valid_pixel_days'], 4)
        self.assertAlmostEqual(metric['pooled_ap'], average_precision_score(y[mask], p[mask]))
        self.assertAlmostEqual(metric['pooled_raw_logit_ap'], average_precision_score(y[mask], z[mask]))
        self.assertNotEqual(metric['pooled_ap'], metric['pooled_raw_logit_ap'])
        self.assertAlmostEqual(metric['pooled_nll'], np.mean(np.logaddexp(0., z[mask].astype(float)) - y[mask] * z[mask]))
        self.assertEqual(metric['raw_logit_statistics']['positive']['count'], 2)
        self.assertEqual(metric['raw_logit_statistics']['negative']['count'], 2)
        self.assertEqual(metric['raw_logit_statistics']['all']['mean'], 0.)
        for value in metric['parent_logit_drift'].values():
            self.assertEqual(value['mean'], 2.)
            self.assertEqual(value['mean_absolute'], 2.)
            self.assertEqual(value['rms'], 2.)
        self.assertAlmostEqual(metric['raw_logit_statistics']['positive']['nll'], 50.)
        bad = deepcopy(row)
        bad['domain_mask'][0] = False
        with self.assertRaisesRegex(ValueError, 'domain'):
            runner.extended_metrics([bad])

    def test_shared_validation_one_h1_per_batch_and_single_head_parity(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x, physics, return_decoder_features=False):
                self.calls += 1
                return dict(logits_det=x)
        class Head(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset
            def forward(self, outputs, physics):
                return outputs['logits_det'] + self.offset
        batches = [dict(x_norm=torch.tensor([[[[1., -1.]]]]), physics={}, target=torch.tensor([[[[1., 0.]]]]),
                        valid_mask=torch.ones(1, 1, 1, 2), domain_mask=torch.ones(1, 1, 1, 2),
                        sample_id=[f'sample{i}'], event_id=['event']) for i in range(2)]
        backbone = Backbone()
        heads = {'constant_lr': Head(.2), 'lower_lr': Head(.1)}
        result = runner.paired_validation_records(backbone, heads, Head(0.), batches, 'cpu')
        self.assertEqual(backbone.calls, 2)
        for arm, head in heads.items():
            reference = runner.pilot.validation_records(backbone, head, batches, 'cpu', compact=True)
            self.assertEqual(runner.pilot.epoch_metrics(reference), runner.pilot.epoch_metrics(result[arm]))
            for left, right in zip(reference, result[arm]):
                np.testing.assert_array_equal(left['calibration_logit'], right['calibration_logit'])
                np.testing.assert_array_equal(left['probability'], right['probability'])
        self.assertTrue(all(not head.training for head in heads.values()))


if __name__ == '__main__':
    unittest.main()
