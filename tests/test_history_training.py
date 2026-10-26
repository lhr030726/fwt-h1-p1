"""CPU contracts for fresh paired history branches and recoverable progress."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('history_training', ROOT / 'tools/train_history_refiner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, optimizer_state_integrity


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.tensor([[.3, -.2], [.7, .4]])).square().mean()
    loss.backward()
    optimizer.step()


def fixture():
    torch.manual_seed(4)
    template = torch.nn.Linear(2, 1)
    heads = {arm: deepcopy(template) for arm in runner.ARMS}
    optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=1e-4, weight_decay=.01) for arm in runner.ARMS}
    bindings = {arm: dict(seed=4, arm=arm, epochs=50, updates_per_epoch=2,
        frozen_parent_checkpoint_sha256='frozen-best49', batch_order_sha256='fixed-order',
        optimizer_param_groups=deepcopy(optimizers[arm].state_dict()['param_groups'])) for arm in runner.ARMS}
    baseline = dict(pooled_ap=.26, nll=.05, brier=.01)
    states = {arm: runner.make_state(heads[arm], optimizers[arm], bindings[arm], [], baseline, 'cpu') for arm in runner.ARMS}
    best = {arm: runner.serialize_state(state) for arm, state in states.items()}
    transaction = runner.build_transaction(states, best, {arm: None for arm in runner.ARMS}, capture_rng('cpu'))
    return heads, optimizers, bindings, transaction


def advance(heads, optimizers, bindings, transaction, ap=.25):
    old = runner.decode_transaction(transaction)
    states, best, new_best = {}, {}, {}
    for arm in runner.ARMS:
        for _ in range(bindings[arm]['updates_per_epoch']):
            update(heads[arm], optimizers[arm])
        last = old[arm]['last']
        epoch = last['completed_epochs'] + 1
        history = deepcopy(last['history']) + [dict(epoch=epoch, updates=epoch * bindings[arm]['updates_per_epoch'],
            validation=dict(pooled_ap=ap, nll=.05, brier=.01))]
        states[arm] = runner.make_state(heads[arm], optimizers[arm], bindings[arm], history, last['baseline_validation'], 'cpu')
        data = runner.serialize_state(states[arm])
        best[arm] = data if ap > old[arm]['best']['validation']['pooled_ap'] else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
        new_best[arm] = data if 'new_best' not in old[arm] or ap > old[arm]['new_best']['validation']['pooled_ap'] else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'new_best')
    return runner.build_transaction(states, best, new_best, capture_rng('cpu'))


class HistoryTrainingTests(unittest.TestCase):
    def test_orders_cover_all_samples_without_consuming_global_rng(self):
        ids = [str(index) for index in range(811)]
        before = torch.get_rng_state().clone()
        orders = runner.epoch_orders(17, ids)
        self.assertEqual(len(orders), 50)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(orders, runner.epoch_orders(17, ids))
        self.assertNotEqual(orders, runner.epoch_orders(29, ids))
        for epoch in orders:
            self.assertEqual([len(batch) for batch in epoch], [16] * 50 + [11])
            self.assertEqual(sorted(item for batch in epoch for item in batch), sorted(ids))

    def test_pair_same_initialization_fresh_isolated_optimizers(self):
        heads, optimizers = runner.initialize_pair(17, 'cpu')
        self.assertEqual(model_state_digest(heads['true_delta']), model_state_digest(heads['zero_delta']))
        self.assertFalse(optimizers['true_delta'].state)
        self.assertFalse(optimizers['zero_delta'].state)
        for left, right in zip(heads['true_delta'].parameters(), heads['zero_delta'].parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        another, _ = runner.initialize_pair(29, 'cpu')
        self.assertNotEqual(model_state_digest(heads['true_delta']), model_state_digest(another['true_delta']))

    def test_control_changes_only_delta(self):
        seen = []
        def head(outputs, anchor, delta, valid):
            seen.append((outputs, anchor, delta.clone(), valid))
            return anchor
        anchor = torch.randn(2, 1, 4, 4)
        valid = torch.ones_like(anchor, dtype=torch.bool)
        delta = torch.randn_like(anchor)
        batch = dict(history_delta=delta, history_valid=valid)
        for arm in runner.ARMS:
            runner.corrected_logits(head, arm, {'feature': 'same'}, anchor, batch)
        self.assertTrue(torch.equal(seen[0][2], delta))
        self.assertEqual(torch.count_nonzero(seen[1][2]), 0)
        self.assertIs(seen[0][3], seen[1][3])
        self.assertIs(seen[0][1], seen[1][1])

    def test_history_wrapper_preserves_original_target_mask_and_owns_history(self):
        item = dict(target=torch.tensor([1.]), valid_mask=torch.tensor([True]), sample_id='id')
        history = dict(history_delta=np.ones((1, 1, 2, 2), dtype=np.float32),
            history_valid=np.ones((1, 1, 2, 2), dtype=np.bool_))
        wrapped = runner.HistoryDataset([item], history)
        result = wrapped[0]
        self.assertIs(result['target'], item['target'])
        self.assertIs(result['valid_mask'], item['valid_mask'])
        self.assertNotIn('history_delta', item)
        result['history_delta'].zero_()
        self.assertEqual(history['history_delta'].sum(), 4)

    def test_fresh_state_and_snapshot_own_optimizer_tensors(self):
        heads, optimizers, bindings, initial = fixture()
        decoded = runner.decode_transaction(initial)
        self.assertEqual(decoded['true_delta']['last']['updates'], 0)
        self.assertEqual(decoded['true_delta']['last']['frozen_parent_updates'], 2499)
        transaction = advance(heads, optimizers, bindings, initial)
        decoded = runner.decode_transaction(transaction)
        saved = decoded['true_delta']['last']
        integrity = deepcopy(saved['optimizer_integrity'])
        update(heads['true_delta'], optimizers['true_delta'])
        self.assertEqual(optimizer_state_integrity(saved['optimizer_state_dict']), integrity)

    def test_transaction_keeps_parent_fallback_new_best_and_earliest_tie(self):
        heads, optimizers, bindings, transaction = fixture()
        transaction = advance(heads, optimizers, bindings, transaction, .25)
        decoded = runner.decode_transaction(transaction)
        self.assertEqual(decoded['true_delta']['best']['completed_epochs'], 0)
        self.assertEqual(decoded['true_delta']['new_best']['completed_epochs'], 1)
        transaction = advance(heads, optimizers, bindings, transaction, .27)
        transaction = advance(heads, optimizers, bindings, transaction, .27)
        decoded = runner.decode_transaction(transaction)
        self.assertEqual(decoded['true_delta']['best']['completed_epochs'], 2)
        self.assertEqual(decoded['true_delta']['new_best']['completed_epochs'], 2)

    def test_tampered_payload_and_unpaired_progress_rejected(self):
        heads, optimizers, bindings, transaction = fixture()
        transaction = advance(heads, optimizers, bindings, transaction)
        broken = deepcopy(transaction)
        broken['arms']['true_delta']['last_data'][0] ^= 1
        with self.assertRaises(ValueError):
            runner.decode_transaction(broken)
        broken = deepcopy(transaction)
        broken['completed_epochs'] = 2
        with self.assertRaises(ValueError):
            runner.decode_transaction(broken)

    def test_restore_is_independent_and_next_step_matches_uninterrupted(self):
        heads, optimizers, bindings, transaction = fixture()
        transaction = advance(heads, optimizers, bindings, transaction)
        saved = runner.decode_transaction(transaction)['true_delta']['last']
        before = deepcopy(saved['optimizer_integrity'])
        clones, clone_optimizers = {}, {}
        for name in ('left', 'right'):
            clones[name] = torch.nn.Linear(2, 1)
            clone_optimizers[name] = torch.optim.AdamW(clones[name].parameters(), lr=1e-4, weight_decay=.01)
            runner.restore_state(saved, clones[name], clone_optimizers[name], bindings['true_delta'], 'cpu')
        other_integrity = optimizer_state_integrity(clone_optimizers['right'].state_dict())
        update(clones['left'], clone_optimizers['left'])
        update(heads['true_delta'], optimizers['true_delta'])
        self.assertEqual(model_state_digest(clones['left']), model_state_digest(heads['true_delta']))
        self.assertEqual(optimizer_state_integrity(clone_optimizers['right'].state_dict()), other_integrity)
        self.assertEqual(optimizer_state_integrity(saved['optimizer_state_dict']), before)
        self.assertEqual(optimizer_state_integrity(clone_optimizers['left'].state_dict()), optimizer_state_integrity(optimizers['true_delta'].state_dict()))

    def test_completed_seed_can_restore_full_fifty_epoch_transaction(self):
        import tempfile
        heads, optimizers, bindings, transaction = fixture()
        for _ in range(50):
            transaction = advance(heads, optimizers, bindings, transaction)
        decoded = runner.decode_transaction(transaction, bindings)
        self.assertEqual(transaction['completed_epochs'], 50)
        for arm in runner.ARMS:
            runner.restore_state(decoded[arm]['last'], heads[arm], optimizers[arm], bindings[arm], 'cpu')
            self.assertEqual(decoded[arm]['last']['updates'], 100)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            histories = {arm: decoded[arm]['last']['history'] for arm in runner.ARMS}
            runner.publish_progress(directory, transaction, histories)
            for arm in runner.ARMS:
                self.assertEqual(runner.read_json(directory / arm / 'history.json'), histories[arm])
                self.assertEqual(runner.sha(directory / arm / 'last.pth'), transaction['arms'][arm]['last_sha256'])

    def test_detailed_metrics_keep_predefined_subsets_and_recall_points(self):
        def row(identity, z):
            z = np.asarray(z, dtype=np.float32)
            return dict(sample_id=identity, event_id=identity, probability=torch.from_numpy(z).sigmoid().numpy(),
                calibration_logit=z, target=np.array([1, 0, 0], dtype=np.float32),
                valid_mask=np.ones(3, dtype=bool), domain_mask=np.ones(3, dtype=bool))
        rows = [row('history', [1., -.5, -1.]), row('missing', [-.5, 1., -1.])]
        metrics = runner.detailed_metrics(rows, {'history'})
        self.assertEqual(metrics['n_valid_pixel_days'], 6)
        self.assertEqual(metrics['n_positive_events'], 2)
        for subset in metrics['history_subsets'].values():
            self.assertEqual(subset['n_samples'], 1)
            self.assertEqual(subset['n_valid_pixel_days'], 3)
            self.assertIn('pooled_raw_logit_ap', subset)
        self.assertGreaterEqual(metrics['operating_points']['validation_recall_70']['recall'], .7)
        self.assertGreaterEqual(metrics['operating_points']['validation_recall_80']['recall'], .8)

    def test_full_validation_zero_equivalence_and_one_shared_parent_pass(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x, physics, return_decoder_features=False):
                self.calls += 1
                return dict(logits_det=x)
        class Parent(torch.nn.Module):
            def forward(self, outputs, physics):
                return outputs['logits_det']
        class Branch(torch.nn.Module):
            def forward(self, outputs, parent, delta, valid):
                return parent + delta * 0.
        batches = [dict(x_norm=torch.tensor([[[[1., -1.]]]]), physics={},
            history_delta=torch.ones(1, 1, 1, 2), history_valid=torch.tensor([[[[True, False]]]]),
            target=torch.tensor([[[[1., 0.]]]]), valid_mask=torch.ones(1, 1, 1, 2),
            domain_mask=torch.ones(1, 1, 1, 2), sample_id=['sample'], event_id=['event'])]
        backbone, parent = Backbone(), Parent()
        result = runner.validation_records(backbone, parent, {arm: Branch() for arm in runner.ARMS}, batches, 'cpu',
            include_parent=True, require_parent_equal=True, collect_heads=False)
        self.assertEqual(backbone.calls, 1)
        self.assertEqual(set(result), {'parent'})
        self.assertEqual(len(result['parent']), 1)

    def test_cli_fixed_budget_and_exclusive_modes(self):
        argv = ['--parent-run', 'parent', '--output', 'output', '--diagnostic-report', 'diag', '--experiment-plan', 'plan']
        args = runner.parse_args(argv)
        self.assertFalse(args.execute or args.preflight)
        self.assertEqual(args.seeds, [4, 17, 29])
        with self.assertRaises(ValueError):
            runner.parse_args(argv + ['--epochs', '100'])
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                runner.parse_args(argv + ['--execute', '--preflight'])

    def test_aggregate_acceptance_requires_paired_gain_and_precision_safeguard(self):
        baseline = dict(pooled_ap=.26, nll=.06, brier=.02, positive_event_ap=.35,
            operating_points={'validation_recall_70': {'precision': .12}})
        receipts = {str(seed): {
            'true_delta': dict(last_five_mean_ap=.263,
                last_five_means=dict(pooled_ap=.263, nll=.06, brier=.02, positive_event_ap=.35),
                last_five_mean_precision_at_recall70=.121),
            'zero_delta': dict(last_five_mean_ap=.261)} for seed in runner.SEEDS}
        result = runner.development_comparison(receipts, baseline)
        self.assertTrue(result['all_development_gates_passed'])
        self.assertFalse(result['automatic_parent_replacement'])
        self.assertFalse(result['independent_evaluation'])
        # An AP gain cannot compensate for a recall70 precision regression.
        receipts['4']['true_delta']['last_five_mean_precision_at_recall70'] = .10
        result = runner.development_comparison(receipts, baseline)
        self.assertFalse(result['all_development_gates_passed'])
        self.assertFalse(result['development_gate_checks']['parent_precision70'])
        receipts['4']['true_delta']['last_five_mean_precision_at_recall70'] = .121
        for arm in receipts.values():
            arm['zero_delta']['last_five_mean_ap'] = .264
        self.assertFalse(runner.development_comparison(receipts, baseline)['all_development_gates_passed'])
        del receipts['29']
        with self.assertRaises(ValueError):
            runner.development_comparison(receipts, baseline)

    def test_status_preserves_seed_epoch_and_resources(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            args = SimpleNamespace(output=Path(temp), _started_at=1.)
            with patch.object(runner.pilot.old, 'resource_snapshot', return_value={'gpu': 'snapshot'}):
                runner.write_status(args, 'training', seed=17, epoch=3, batch=10, completed_epochs=2)
            status = runner.read_json(Path(temp) / 'status.json')
            self.assertEqual((status['seed'], status['epoch'], status['batch']), (17, 3, 10))
            self.assertEqual(status['resources'], {'gpu': 'snapshot'})
            self.assertEqual(len((Path(temp) / 'resource-history.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
