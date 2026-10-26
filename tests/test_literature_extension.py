"""CPU mechanisms for a declared budget extension without changing old acceptance."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('literature_extension', ROOT / 'tools/extend_literature_ablation.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
from tests.test_literature_ablation import state_fixture, step
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, restore_rng, optimizer_state_integrity


def parent_fixture(source=None, inputs=None):
    states, best_bytes, originals = {}, {}, {}
    for arm in extension.ARMS:
        _, _, state, best = state_fixture(4, arm, epochs=10, selected_epoch=2)
        original = dict(seed=4, arm=arm, recipe=deepcopy(extension.pilot.BASE_RECIPE), stage='pilot', candidate=None,
            source_files=source or {'fwt_v2/data.py': 'original-source'}, input_files_sha256=inputs or {'original-data': 'hash'},
            batch_order_sha256='original-order', auxiliary_weight=.05 if arm.endswith('tversky') else 0.,
            head_configuration={'architecture': arm.split('_')[0]}, runtime={'torch': str(torch.__version__)})
        state['binding'] = original
        selected = extension.deserialize_state(best)
        selected['binding'] = original
        states[arm], best_bytes[arm], originals[arm] = state, extension.serialize_state(selected), original
    transaction = extension.pilot.build_transaction(4, states, best_bytes, capture_rng('cpu'))
    first = originals['old_bce']
    plan = dict(recipe=dict(extension.pilot.BASE_RECIPE, epochs=50),
        parent_source_files=first['source_files'],
        source_files=dict(first['source_files'], **{'tools/extend_literature_ablation.py': 'new-source'}),
        input_files_sha256=dict(first['input_files_sha256'], **{'parent-transaction': 'hash'}),
        original_binding_sha256={a: extension.canonical_sha(b) for a, b in originals.items()},
        original_history_sha256={a: extension.canonical_sha(s['history']) for a, s in states.items()},
        batch_order_sha256={'4': 'extended-order'}, parent_batch_order_sha256={'4': 'original-order'},
        parent_plan_sha256='parent-plan', parent_transaction_sha256='parent-transaction')
    return transaction, originals, plan


def models():
    heads = {a: torch.nn.Linear(3, 1) for a in extension.ARMS}
    optimizers = {a: torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=.01) for a, h in heads.items()}
    return heads, optimizers


def advanced_epoch(transaction, heads, optimizers):
    decoded = extension.decode_transaction(transaction)
    states, best_bytes = {}, {}
    for arm, (last, best) in decoded.items():
        for _ in range(51):
            step(heads[arm], optimizers[arm])
        last['epoch'] += 1
        last['completed_epochs'] += 1
        last['updates'] += 51
        last['history'].append(dict(epoch=last['epoch'], validation={'pooled_ap': .4}))
        last.update(model_state_dict=extension.tensor_cpu_tree(heads[arm].state_dict()),
            model_state_sha256=model_state_digest(heads[arm]), validation_ap=.4,
            optimizer_state_dict=extension.tensor_cpu_tree(optimizers[arm].state_dict()), rng_state=capture_rng('cpu'))
        last['optimizer_integrity'] = optimizer_state_integrity(last['optimizer_state_dict'])
        states[arm] = last
        best_bytes[arm] = extension.pilot.transaction_checkpoint_bytes(transaction['arms'][arm], 'best')
    return extension.build_transaction(states, best_bytes, capture_rng('cpu'))


class ExtensionTests(unittest.TestCase):
    def test_order_prefix_is_exact_and_global_rng_is_unchanged(self):
        ids = [f'sample-{i}' for i in range(811)]
        original = extension.pilot.seeded_order(4, ids)
        before = torch.get_rng_state().clone()
        order = extension.extended_order(ids, original)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(order[:10], original)
        self.assertEqual(sum(len(b) for b in order), 2550)
        self.assertEqual(len(order), 50)
        self.assertEqual(len({extension.canonical_sha(e) for e in order}), 50)
        altered = deepcopy(original)
        altered[0][0][0], altered[0][0][1] = altered[0][0][1], altered[0][0][0]
        with self.assertRaisesRegex(ValueError, 'first ten orders differ'):
            extension.extended_order(ids, altered)

    def test_binding_migration_has_an_explicit_narrow_allowlist(self):
        _, originals, plan = parent_fixture()
        original = originals['old_bce']
        binding = extension.extension_binding(original, plan)
        changed = {k for k in original.keys() | binding.keys() if original.get(k) != binding.get(k)}
        self.assertEqual(changed, set(extension.ALLOWED_BINDING_CHANGES))
        self.assertEqual(binding['head_configuration'], original['head_configuration'])
        self.assertEqual(binding['runtime'], original['runtime'])
        for alteration in ('lr', 'source', 'data', 'order'):
            bad = deepcopy(plan)
            if alteration == 'lr': bad['recipe']['learning_rate'] *= 2
            if alteration == 'source': bad['source_files']['fwt_v2/data.py'] = 'modified'
            if alteration == 'data': bad['input_files_sha256']['original-data'] = 'modified'
            if alteration == 'order': bad['parent_batch_order_sha256']['4'] = 'modified'
            with self.assertRaisesRegex(ValueError, 'Unlisted recipe/source/input/order'):
                extension.extension_binding(original, bad)

    def test_migration_matches_original_last_next_adamw_and_all_rng_streams(self):
        parent, originals, plan = parent_fixture()
        decoded = extension.pilot.decode_transaction(parent, 4, extension.ARMS)
        reference, reference_opt = models()
        for arm, (last, _) in decoded.items():
            extension.pilot.restore_state(last, reference[arm], reference_opt[arm], originals[arm], 'cpu')
        restore_rng(parent['rng_state'], 'cpu')
        expected_draws = []
        for arm in extension.ARMS:
            expected_draws.append((random.random(), np.random.random()))
            step(reference[arm], reference_opt[arm])
        expected_rng = torch.get_rng_state().clone()
        heads, optimizers = models()
        migrated, bindings, histories = extension.migrate_parent(parent, heads, optimizers, originals, plan, 'cpu')
        actual = extension.decode_transaction(migrated)
        for arm, (last, best) in actual.items():
            self.assertEqual(last['model_state_sha256'], decoded[arm][0]['model_state_sha256'])
            self.assertEqual(model_state_digest(heads[arm]), last['model_state_sha256'])
            self.assertNotEqual(last['model_state_sha256'], best['model_state_sha256'])
            self.assertEqual(best['model_state_sha256'], decoded[arm][1]['model_state_sha256'])
            self.assertEqual(best['epoch'], 2)
            self.assertEqual(histories[arm], decoded[arm][0]['history'])
            self.assertEqual(last['inherited_checkpoint']['sha256'], parent['arms'][arm]['last_sha256'])
            self.assertEqual(best['inherited_checkpoint']['sha256'], parent['arms'][arm]['best_sha256'])
        draws = []
        for arm in extension.ARMS:
            draws.append((random.random(), np.random.random()))
            step(heads[arm], optimizers[arm])
            self.assertEqual(model_state_digest(heads[arm]), model_state_digest(reference[arm]))
            self.assertEqual(optimizer_state_integrity(optimizers[arm].state_dict()),
                             optimizer_state_integrity(reference_opt[arm].state_dict()))
        self.assertEqual(draws, expected_draws)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        # The old acceptance contract still refuses the independent new schema.
        with self.assertRaisesRegex(ValueError, 'schema/seed/arms/progress'):
            extension.pilot.decode_transaction(migrated, 4, extension.ARMS)

    def test_atomic_epoch11_resume_skips_completed_work_and_keeps_earlier_best(self):
        parent, originals, plan = parent_fixture()
        heads, optimizers = models()
        migrated, bindings, parent_histories = extension.migrate_parent(parent, heads, optimizers, originals, plan, 'cpu')
        advanced = advanced_epoch(migrated, heads, optimizers)
        self.assertEqual(advanced['completed_epochs'], 11)
        with tempfile.TemporaryDirectory() as temp, patch('fwt_v2.artifacts.WORKSPACE', Path(temp)):
            root = Path(temp) / 'source'
            for arm in extension.ARMS: (root / arm).mkdir(parents=True)
            extension.publish_transaction(root, migrated)
            real_atomic = extension.pilot.atomic_bytes
            calls = 0
            def interrupted(path, value):
                nonlocal calls
                calls += 1
                if calls == 2: raise OSError('interrupted between mirrors')
                return real_atomic(path, value)
            with patch.object(extension.pilot, 'atomic_bytes', side_effect=interrupted):
                with self.assertRaises(OSError): extension.publish_transaction(root, advanced)
            authority = torch.load(root / 'transaction.pth', weights_only=True)
            restored, restored_opt = models()
            start, history, best_bytes, aps, selected = extension.restore_transaction(
                authority, restored, restored_opt, bindings, parent_histories, 'cpu')
            self.assertEqual(start, 11)
            self.assertEqual(list(range(start, extension.EPOCHS)), list(range(11, 50)))
            for arm in extension.ARMS:
                self.assertEqual(model_state_digest(restored[arm]), model_state_digest(heads[arm]))
                self.assertEqual(optimizer_state_integrity(restored_opt[arm].state_dict()),
                                 optimizer_state_integrity(optimizers[arm].state_dict()))
                self.assertEqual(selected[arm], 2)
                self.assertEqual(aps[arm], .6)
                self.assertEqual(len(history[arm]), 11)
                self.assertEqual(best_bytes[arm], extension.pilot.transaction_checkpoint_bytes(migrated['arms'][arm], 'best'))
            destination = Path(temp) / 'recovered'
            for arm in extension.ARMS: (destination / arm).mkdir(parents=True)
            extension.publish_transaction(destination, authority)
            for arm in extension.ARMS:
                self.assertEqual(extension.sha(destination / arm / 'last.pth'), authority['arms'][arm]['last_sha256'])

    def test_corruption_original_history_and_overbudget_are_rejected(self):
        parent, originals, plan = parent_fixture()
        heads, optimizers = models()
        transaction, bindings, history = extension.migrate_parent(parent, heads, optimizers, originals, plan, 'cpu')
        state = extension.decode_transaction(transaction)['old_bce'][0]
        changed = deepcopy(state)
        changed['history'][0]['validation']['pooled_ap'] = .9
        with self.assertRaisesRegex(ValueError, 'progress/original history'):
            extension.restore_state(changed, heads['old_bce'], optimizers['old_bce'], bindings['old_bce'], 'cpu', history['old_bce'])
        changed = deepcopy(state)
        next(iter(changed['optimizer_state_dict']['state'].values()))['exp_avg'].add_(1)
        with self.assertRaisesRegex(ValueError, 'AdamW integrity'):
            extension.restore_state(changed, heads['old_bce'], optimizers['old_bce'], bindings['old_bce'], 'cpu', history['old_bce'])
        over = dict(transaction, completed_epochs=51)
        with self.assertRaisesRegex(ValueError, 'schema/arms/progress'):
            extension.decode_transaction(over)
        parent['completed_epochs'] = 9
        with self.assertRaises(ValueError):
            extension.migrate_parent(parent, heads, optimizers, originals, plan, 'cpu')

    def test_fiftieth_epoch_has_2550_updates_and_no_remaining_loop(self):
        parent, originals, plan = parent_fixture()
        heads, optimizers = models()
        transaction, bindings, history = extension.migrate_parent(parent, heads, optimizers, originals, plan, 'cpu')
        for _ in range(40):
            transaction = advanced_epoch(transaction, heads, optimizers)
        recovered, recovered_opt = models()
        start, final_history, _, _, selected = extension.restore_transaction(
            transaction, recovered, recovered_opt, bindings, history, 'cpu')
        self.assertEqual(start, 50)
        self.assertEqual(list(range(start, extension.EPOCHS)), [])
        for arm, (last, _) in extension.decode_transaction(transaction).items():
            self.assertEqual(last['updates'], 2550)
            self.assertEqual(len(final_history[arm]), 50)
            self.assertEqual(selected[arm], 2)
            self.assertEqual({int(s['step']) for s in last['optimizer_state_dict']['state'].values()}, {2550})

    def test_existing_output_is_not_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'original').write_text('preserved')
            args = SimpleNamespace(output=root, execute=True)
            with patch.object(extension, 'parse_args', return_value=args), patch.object(extension, 'make_plan', return_value={}), \
                    patch.object(extension, 'execute', side_effect=FileExistsError('existing')):
                with self.assertRaises(FileExistsError): extension.main([])
            self.assertEqual([p.name for p in root.iterdir()], ['original'])


if __name__ == '__main__':
    unittest.main()
