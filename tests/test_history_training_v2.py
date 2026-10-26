"""CPU contracts for three-arm training, loss dispatch and strict recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('history_training_v2', ROOT / 'tools/train_history_refiner_v2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, optimizer_state_integrity
from fwt_v2.history_refiner_v2 import build_event_support


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[.3, -.2], [.7, .4]])).square().mean().backward()
    optimizer.step()


def fixture():
    torch.manual_seed(4)
    template = torch.nn.Linear(2, 1)
    heads = {arm: deepcopy(template) for arm in runner.ARMS}
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
                  for arm, head in heads.items()}
    bindings = {arm: dict(seed=4, arm=arm, epochs=50, updates_per_epoch=2,
        event_support_sha256='support-v2', runtime={'machine': 'fixed'},
        frozen_parent_checkpoint_sha256='frozen-best49', batch_order_sha256='fixed-order',
        optimizer_param_groups=deepcopy(optimizers[arm].state_dict()['param_groups'])) for arm in runner.ARMS}
    baseline = dict(pooled_ap=.26, nll=.05, brier=.01)
    states = {arm: runner.make_state(heads[arm], optimizers[arm], bindings[arm], [], baseline, 'cpu')
              for arm in runner.ARMS}
    best = {arm: runner.serialize_state(state) for arm, state in states.items()}
    transaction = runner.build_transaction(states, best, {arm: None for arm in runner.ARMS}, capture_rng('cpu'))
    return heads, optimizers, bindings, transaction


def advance(heads, optimizers, bindings, transaction, ap=.27):
    old = runner.decode_transaction(transaction)
    states, best, new_best = {}, {}, {}
    for arm in runner.ARMS:
        for _ in range(bindings[arm]['updates_per_epoch']):
            update(heads[arm], optimizers[arm])
        last = old[arm]['last']
        epoch = last['completed_epochs'] + 1
        history = deepcopy(last['history']) + [dict(epoch=epoch, updates=epoch * bindings[arm]['updates_per_epoch'],
                                                   validation=dict(pooled_ap=ap, nll=.05, brier=.01))]
        states[arm] = runner.make_state(heads[arm], optimizers[arm], bindings[arm], history,
                                       last['baseline_validation'], 'cpu')
        data = runner.serialize_state(states[arm])
        best[arm] = data if ap > old[arm]['best']['validation']['pooled_ap'] else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
        new_best[arm] = data if 'new_best' not in old[arm] or ap > old[arm]['new_best']['validation']['pooled_ap'] else runner.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'new_best')
    return runner.build_transaction(states, best, new_best, capture_rng('cpu'))


class HistoryV2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_cli_does_not_execute(self):
        args = runner.parse_args(['--parent-run', '.', '--output', 'unused',
                                  '--diagnostic-report', 'diagnostic.json', '--experiment-plan', 'plan.md'])
        self.assertFalse(args.execute)
        self.assertFalse(args.preflight)
        self.assertEqual(args.seeds, [4, 17, 29])
        self.assertEqual(args.epochs, 50)
        base = ['--parent-run', '.', '--output', 'unused',
                '--diagnostic-report', 'diagnostic.json', '--experiment-plan', 'plan.md']
        with self.assertRaises(ValueError):
            runner.parse_args(base + ['--execute'])
        with self.assertRaises(ValueError):
            runner.parse_args(base + ['--preflight', '--preflight-run', 'previous'])

    def test_execute_requires_matching_successful_v2_preflight(self):
        from fwt_v2.artifacts import atomic_json
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            plan = dict(schema=runner.PLAN_SCHEMA, mode='preflight', output=str(directory),
                resume_run=None, preflight_run=None, preflight_artifacts=None,
                source_files={'module': 'frozen-digest'}, recipe={'event_mix': .5, 'cap': 2.},
                parent_selected_validation={'pooled_ap': .2612})
            binding = dict(source_files=plan['source_files'], recipe=plan['recipe'])
            atomic_json(directory / 'plan.json', plan)
            atomic_json(directory / 'input-binding.json', binding)
            report = dict(schema='fwt_history_event_preflight_v2', status='passed',
                zero_initialization_exact=True, frozen_parent_unchanged=True, training_checkpoints_created=False,
                disposable_optimizer_steps_per_arm=1, arms={arm: {} for arm in runner.ARMS},
                parent_replay={'pooled_ap': .2612}, plan_sha256=runner.sha(directory / 'plan.json'),
                input_binding_sha256=runner.sha(directory / 'input-binding.json'))
            atomic_json(directory / 'preflight.json', report)
            execution = dict(plan, mode='execute', output='new-output', preflight_run=str(directory))
            self.assertEqual(set(runner.validate_preflight(directory, execution)),
                             {'preflight.json', 'plan.json', 'input-binding.json'})
            for changes in ({'status': 'failed'}, {'schema': 'old_preflight'},
                            {'input_binding_sha256': 'tampered'}, {'disposable_optimizer_steps_per_arm': 0},
                            {'parent_replay': {'pooled_ap': .1}}, {'arms': {'pixel_bce': {}}}):
                atomic_json(directory / 'preflight.json', dict(report, **changes))
                with self.assertRaises(ValueError):
                    runner.validate_preflight(directory, execution)
            atomic_json(directory / 'preflight.json', report)
            for key, value in [('source_files', {'module': 'new-digest'}), ('recipe', {'event_mix': .6, 'cap': 2.})]:
                with self.assertRaises(ValueError):
                    runner.validate_preflight(directory, dict(execution, **{key: value}))

    def test_three_arms_same_initial_weights_and_isolated_optimizers(self):
        heads, optimizers = runner.initialize_pair(17, 'cpu')
        self.assertEqual(len(heads), 3)
        self.assertEqual(len({model_state_digest(head) for head in heads.values()}), 1)
        self.assertTrue(all(not optimizer.state for optimizer in optimizers.values()))
        pointers = [next(head.parameters()).data_ptr() for head in heads.values()]
        self.assertEqual(len(set(pointers)), 3)
        self.assertEqual(heads['event_bce_bounded'].configuration()['correction_cap'], 2.)
        other, _ = runner.initialize_pair(29, 'cpu')
        self.assertNotEqual(model_state_digest(heads['pixel_bce']), model_state_digest(other['pixel_bce']))

    def test_all_arms_use_real_delta(self):
        seen = []
        def head(outputs, anchor, delta, valid):
            seen.append((delta, valid))
            return anchor
        anchor = torch.zeros(2, 1, 4, 4)
        batch = dict(history_delta=torch.ones_like(anchor), history_valid=torch.ones_like(anchor, dtype=torch.bool))
        for arm in runner.ARMS:
            runner.corrected_logits(head, arm, {}, anchor, batch)
        self.assertTrue(all(delta is batch['history_delta'] and valid is batch['history_valid'] for delta, valid in seen))

    def test_dispatch_preserves_A_and_identical_B_C_objectives(self):
        from fwt_v2.literature_refiner import masked_refinement_loss
        from fwt_v2.history_refiner_v2 import mixed_event_bce
        logits = torch.tensor([[[[.1, -.2]]], [[[1., -.5]]]], requires_grad=True)
        labels = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        masks = torch.ones_like(labels)
        samples = [dict(sample_id=str(i), event_id=event, target=labels[i], valid_mask=masks[i])
                   for i, event in enumerate(['short', 'long'])]
        samples += [dict(sample_id='extra', event_id='long', target=labels[1], valid_mask=masks[1])]
        support = build_event_support(samples)
        batch = dict(target=labels, valid_mask=masks, event_id=['short', 'long'])
        a = runner.arm_loss('pixel_bce', logits, batch, support)['total']
        b = runner.arm_loss('event_bce', logits, batch, support)['total']
        c = runner.arm_loss('event_bce_bounded', logits, batch, support)['total']
        self.assertTrue(torch.equal(a, masked_refinement_loss(logits, labels, masks, aux_weight=0.)['bce']))
        self.assertTrue(torch.equal(b, mixed_event_bce(logits, labels, masks, batch['event_id'], support, .5)['total']))
        self.assertTrue(torch.equal(b, c))
        self.assertNotEqual(float(a), float(b))
        b.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_cpu_shared_parent_training_and_validation_missing_history_fallback(self):
        self._exercise_shared_parent('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Local CUDA is optional; server preflight remains mandatory')
    def test_cuda_three_arm_device_transfer_loss_and_missing_history_fallback(self):
        self._exercise_shared_parent('cuda:0')

    def _exercise_shared_parent(self, device):
        from torch.utils.data import DataLoader
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(.5))
                self.calls = 0
            def forward(self, x, physics, return_decoder_features=False):
                self.calls += 1
                z = x[:, :1] * self.scale
                return dict(logits_det=z, decoder_feature=z.expand(-1, 64, -1, -1).contiguous())
        class Parent(torch.nn.Module):
            def forward(self, outputs, physics):
                return outputs['logits_det']
        samples = [dict(x_norm=torch.randn(22, 16, 16), target=torch.zeros(1, 16, 16),
            valid_mask=torch.ones(1, 16, 16), domain_mask=torch.ones(1, 16, 16), physics={},
            sample_id=f'sample-{i}', event_id=f'event-{i}') for i in range(2)]
        for sample in samples:
            sample['target'][:, 3:7, 3:7] = 1.
        delta = np.ones((2, 1, 16, 16), dtype=np.float32)
        valid = np.ones((2, 1, 16, 16), dtype=np.bool_)
        valid[0] = False
        delta[0] = 0
        dataset = runner.HistoryDataset(samples, dict(history_delta=delta, history_valid=valid))
        support = build_event_support(dataset)
        self.assertEqual(support['n_samples'], 2)  # Missing history contributes to the full event denominator.
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        cpu_batch = next(iter(loader))
        batch = runner.history_to_device(cpu_batch, device)
        backbone, parent = Backbone().to(device).requires_grad_(False).eval(), Parent().to(device).eval()
        frozen = model_state_digest(backbone)
        heads, optimizers = runner.initialize_pair(4, device)
        with torch.no_grad():
            outputs = backbone(batch['x_norm'], batch['physics'], return_decoder_features=True)
            anchor = parent(outputs, batch['physics'])
        for arm in runner.ARMS:
            optimizers[arm].zero_grad(set_to_none=True)
            logits = runner.corrected_logits(heads[arm], arm, outputs, anchor, batch)
            self.assertTrue(torch.equal(logits, anchor))
            runner.arm_loss(arm, logits, batch, support)['total'].backward()
            self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                for parameter in heads[arm].parameters()))
            optimizers[arm].step()
            with torch.no_grad():
                after = runner.corrected_logits(heads[arm], arm, outputs, anchor, batch)
            self.assertTrue(torch.equal(after[0], anchor[0]))
            self.assertTrue(bool((after[1] != anchor[1]).any()))
        before = backbone.calls
        records = runner.validation_records(backbone, parent, heads, loader, device, include_parent=True)
        self.assertEqual(backbone.calls - before, 1)
        self.assertEqual(set(records), {'parent', *runner.ARMS})
        for arm in runner.ARMS:
            np.testing.assert_array_equal(records[arm][0]['calibration_logit'], records['parent'][0]['calibration_logit'])
        self.assertEqual(model_state_digest(backbone), frozen)
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))

    def test_complete_three_arm_roundtrip_and_next_step(self):
        heads, optimizers, bindings, transaction = fixture()
        transaction = advance(heads, optimizers, bindings, transaction)
        decoded = runner.decode_transaction(transaction, bindings)
        self.assertEqual(set(decoded), set(runner.ARMS))
        for arm in runner.ARMS:
            clone = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-4, weight_decay=.01)
            state = decoded[arm]['last']
            integrity = deepcopy(state['optimizer_integrity'])
            runner.restore_state(state, clone, optimizer, bindings[arm], 'cpu')
            update(clone, optimizer)
            update(heads[arm], optimizers[arm])
            self.assertEqual(model_state_digest(clone), model_state_digest(heads[arm]))
            self.assertEqual(optimizer_state_integrity(optimizer.state_dict()), optimizer_state_integrity(optimizers[arm].state_dict()))
            self.assertEqual(optimizer_state_integrity(state['optimizer_state_dict']), integrity)

    def test_reject_v1_missing_arm_tampered_bytes_and_cross_arm_state(self):
        heads, optimizers, bindings, transaction = fixture()
        transaction = advance(heads, optimizers, bindings, transaction)
        for mutate in (lambda value: value.update(schema='fwt_history_pair_transaction_v1'),
                       lambda value: value['arms'].pop('event_bce_bounded'),
                       lambda value: value['arms']['pixel_bce']['last_data'].__setitem__(0, value['arms']['pixel_bce']['last_data'][0] ^ 1)):
            broken = deepcopy(transaction)
            mutate(broken)
            with self.assertRaises(ValueError):
                runner.decode_transaction(broken)
        state = runner.decode_transaction(transaction)['pixel_bce']['last']
        with self.assertRaises(ValueError):
            runner.restore_state(state, heads['event_bce'], optimizers['event_bce'], bindings['event_bce'], 'cpu')
        state = deepcopy(state)
        state['schema'] = 'fwt_history_branch_training_state_v1'
        with self.assertRaises(ValueError):
            runner.check_state(state)

    def test_resume_rejects_support_runtime_or_bound_changes(self):
        heads, optimizers, bindings, transaction = fixture()
        state = runner.decode_transaction(transaction)['pixel_bce']['last']
        for key, value in [('event_support_sha256', 'different'), ('runtime', {'machine': 'other'})]:
            changed = dict(bindings['pixel_bce'], **{key: value})
            with self.assertRaises(ValueError):
                runner.restore_state(state, heads['pixel_bce'], optimizers['pixel_bce'], changed, 'cpu')
        real, real_optimizers = runner.initialize_pair(4, 'cpu')
        binding = dict(bindings['event_bce_bounded'],
            branch_configuration=real['event_bce_bounded'].configuration(),
            optimizer_param_groups=deepcopy(real_optimizers['event_bce_bounded'].state_dict()['param_groups']))
        baseline = dict(pooled_ap=.26, nll=.05, brier=.01)
        state = runner.make_state(real['event_bce_bounded'], real_optimizers['event_bce_bounded'], binding, [], baseline, 'cpu')
        real['event_bce_bounded'].cap = 3.
        with self.assertRaises(ValueError):
            runner.restore_state(state, real['event_bce_bounded'], real_optimizers['event_bce_bounded'], binding, 'cpu')

    def test_ties_and_committed_history_survive_publication(self):
        heads, optimizers, bindings, transaction = fixture()
        for ap in (.25, .27, .27):
            transaction = advance(heads, optimizers, bindings, transaction, ap)
        decoded = runner.decode_transaction(transaction)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            runner.publish_progress(directory, transaction, {arm: decoded[arm]['last']['history'] for arm in runner.ARMS})
            for arm in runner.ARMS:
                self.assertEqual(decoded[arm]['best']['completed_epochs'], 2)
                self.assertEqual(decoded[arm]['new_best']['completed_epochs'], 2)
                self.assertEqual(runner.sha(directory / arm / 'last.pth'), transaction['arms'][arm]['last_sha256'])
                self.assertEqual(len(runner.read_json(directory / arm / 'history.json')), 3)

    def test_fresh_machine_allowance_retains_content_and_numeric_constraints(self):
        stat = dict(sample_id='s', field='input_path', path='/data/s.npy', size=64,
                    mtime_ns=1, device=2, inode=3)
        runtime = dict(python='3.10.19 original build', torch='2.5.1+cu124', gpu='old',
                       cpu_threads=2, interop_threads=2, matmul_tf32=False, cudnn_tf32=True)
        old = dict(array_files_sha256={'/data/s.npy': 'sha'}, array_stat_identity=[stat],
                   frozen_backbone_state_sha256='h1', runtime=runtime)
        current_stat = dict(stat, mtime_ns=4, device=5, inode=6)
        current_runtime = dict(runtime, python='3.10.14 new build', gpu='new')
        result = runner.validate_fresh_parent_binding(old, old['array_files_sha256'], [current_stat], 'h1', current_runtime)
        self.assertEqual(result['array_stat_changed_entries'], 1)
        self.assertEqual(set(result['runtime_changes']), {'python', 'gpu'})
        with self.assertRaises(ValueError):
            runner.validate_fresh_parent_binding(old, {'/data/s.npy': 'tampered'}, [current_stat], 'h1', current_runtime)
        for key, value in [('path', '/different/s.npy'), ('size', 65), ('sample_id', 'other')]:
            with self.assertRaises(ValueError):
                runner.validate_fresh_parent_binding(old, old['array_files_sha256'], [dict(current_stat, **{key: value})], 'h1', current_runtime)
        for key, value in [('python', '3.11.10 build'), ('torch', '2.6.0+cu124'), ('matmul_tf32', True)]:
            with self.assertRaises(ValueError):
                runner.validate_fresh_parent_binding(old, old['array_files_sha256'], [current_stat], 'h1', dict(current_runtime, **{key: value}))

    def test_scientific_versions_remain_frozen(self):
        with patch('importlib.metadata.version', side_effect=runner.SCIENTIFIC_DEPENDENCIES.__getitem__):
            self.assertEqual(runner.validate_scientific_environment(), runner.SCIENTIFIC_DEPENDENCIES)
        for package in runner.SCIENTIFIC_DEPENDENCIES:
            altered = dict(runner.SCIENTIFIC_DEPENDENCIES, **{package: 'different'})
            with patch('importlib.metadata.version', side_effect=altered.__getitem__):
                with self.assertRaises(ValueError):
                    runner.validate_scientific_environment()

    def test_addition_allowlist_does_not_relax_frozen_hashes(self):
        original = {'fwt_v2/model.py': 'old'}
        current = dict(original, **{name: 'new' for name in runner.ADDITIONAL_PARENT_SOURCES})
        runner.previous.validate_parent_source_files(original, current, runner.ADDITIONAL_PARENT_SOURCES)
        with self.assertRaises(ValueError):
            runner.previous.validate_parent_source_files(original, dict(current, **{'fwt_v2/model.py': 'changed'}), runner.ADDITIONAL_PARENT_SOURCES)
        with self.assertRaises(ValueError):
            runner.previous.validate_parent_source_files(original, dict(current, **{'fwt_v2/unapproved.py': 'extra'}), runner.ADDITIONAL_PARENT_SOURCES)

    def test_comparison_reports_B_A_C_B_C_A_and_rejects_false_improvement(self):
        baseline = dict(pooled_ap=.26, nll=.06, brier=.014, positive_event_ap=.34)
        receipts = {}
        for seed in runner.SEEDS:
            receipts[str(seed)] = {}
            for index, arm in enumerate(runner.ARMS):
                receipts[str(seed)][arm] = dict(
                    last_five_means=dict(pooled_ap=.37 + index * .001, nll=.055 - index * .001,
                                         brier=.013 - index * .0001, positive_event_ap=.38 + index * .001),
                    last_five_mean_precision_at_recall70=.19 + index * .001,
                    last_five_event_ap={'first': .3 + index * .01, 'second': .5 - index * .005})
        report = runner.development_comparison(receipts, baseline)
        self.assertEqual(len(report['contrasts']), 3)
        self.assertFalse(report['automatic_parent_replacement'])
        self.assertTrue(all(value['all_development_gates_passed'] for value in report['contrasts'].values()))
        self.assertEqual(report['contrasts']['event_bce_minus_pixel_bce']['event_ap_degraded'], 1)
        receipts['4']['event_bce']['last_five_means']['pooled_ap'] = .30
        result = runner.development_comparison(receipts, baseline)['contrasts']['event_bce_minus_pixel_bce']
        self.assertFalse(result['all_development_gates_passed'])
        self.assertFalse(result['development_gate_checks']['pooled_ap_noninferior'])


if __name__ == '__main__':
    unittest.main()
