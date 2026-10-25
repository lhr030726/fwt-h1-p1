"""Synthetic validation sequences only: no project model, data or GPU access."""
from copy import deepcopy
import subprocess
import sys
import unittest

import torch

from fwt_v2.training_control import (TrainingController, advance_state, initial_state,
    policy_for_epochs, replay_history, validate_completion)


def history_for(aps, epochs=100):
    policy = policy_for_epochs(epochs)
    state = initial_state([.001])
    rows = []
    for epoch, ap in enumerate(aps):
        state = advance_state(state, ap, policy)
        rows.append(dict(epoch=epoch, validation=dict(pooled_ap=ap), training_control=state,
                         training_checks=dict(updates=1, cumulative_updates=epoch + 1)))
    return rows


def completion_for(aps, epochs=100):
    rows = history_for(aps, epochs)
    state = rows[-1]['training_control']
    reason = 'early_stopping' if state['should_stop'] else 'max_epochs'
    common = dict(completed_epochs=len(rows), actual_epochs=len(rows), max_epochs=epochs,
                  selection_frozen=True, stop_reason=reason,
                  training_control=policy_for_epochs(epochs), training_control_state=state)
    diagnostic = dict(common, status='completed', selected_epoch=state['best_epoch'],
                      remaining_epochs=0, budget_remaining_epochs=epochs-len(rows),
                      epochs=[row['training_checks'] for row in rows])
    selection = dict(common, epoch=state['best_epoch'], validation_ap=state['best_validation_ap'],
                     requested_epochs=epochs)
    return diagnostic, selection, rows


def controller(epochs=100, lr=.001):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=lr)
    return TrainingController(optimizer, epochs, policy_for_epochs(epochs))


class TrainingControlTests(unittest.TestCase):
    def test_metadata_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; from fwt_v2.training_control import policy_for_epochs; "
            "assert policy_for_epochs(100)['max_epochs'] == 100; assert 'torch' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_policy_and_independent_return_values(self):
        self.assertIsNone(policy_for_epochs(2))
        self.assertIsNone(policy_for_epochs(30))
        policy = policy_for_epochs(100)
        policy['scheduler']['patience'] = 1
        self.assertEqual(policy_for_epochs(100)['scheduler']['patience'], 5)
        for value in (True, 1, 99, 100.0, '100'):
            with self.assertRaises(ValueError):
                policy_for_epochs(value)

    def test_exact_15_bad_epochs_and_earliest_tie(self):
        control = controller()
        for index in range(16):
            state = control.step(.2)
            self.assertEqual(state['bad_epochs'], index)
            self.assertEqual(state['best_epoch'], 0)
            self.assertEqual(state['should_stop'], index == 15)
        with self.assertRaises(ValueError):
            control.step(.3)

    def test_strict_improvement_resets_early_stop(self):
        control = controller()
        for ap in [.2] * 15 + [.2000000001] + [.2] * 14:
            state = control.step(ap)
        self.assertFalse(state['should_stop'])
        self.assertEqual(state['best_epoch'], 15)
        self.assertEqual(state['bad_epochs'], 14)
        self.assertTrue(control.step(.2)['should_stop'])

    def test_lr_changes_after_six_bad_epochs_not_five(self):
        control = controller()
        for _ in range(6):
            state = control.step(.2)
        self.assertEqual(state['learning_rates_after'], [.001])
        state = control.step(.2)
        self.assertEqual(state['learning_rates_before'], [.001])
        self.assertEqual(state['learning_rates_after'], [.0005])
        self.assertEqual(state['scheduler_bad_epochs'], 0)
        self.assertEqual(state['bad_epochs'], 6)

    def test_lr_floor_with_improvement_between_plateaus(self):
        control = controller()
        for cycle in range(9):
            control.step(.1 + cycle * .01)
            for _ in range(6):
                state = control.step(.01)
        self.assertEqual(state['learning_rates_after'], [1e-5])
        self.assertFalse(state['should_stop'])

    def test_invalid_ap_rejected_before_scheduler_mutation(self):
        control = controller()
        before = control.state_dict()
        for invalid in (None, True, float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                control.step(invalid)
            self.assertEqual(control.state_dict(), before)

    def test_external_lr_mutation_rejected(self):
        control = controller()
        control.optimizer.param_groups[0]['lr'] = .0009
        with self.assertRaises(ValueError):
            control.step(.5)

    def test_fixed_legacy_recipes_have_no_scheduler_or_early_stop(self):
        for budget in (2, 30):
            control = controller(budget)
            self.assertIsNone(control.scheduler)
            for _ in range(budget):
                state = control.step(.2)
            self.assertEqual(state['learning_rates_after'], [.001])
            self.assertFalse(state['should_stop'])
            with self.assertRaises(ValueError):
                control.step(.3)

    def test_replay_rejects_extra_epoch_and_bad_counter(self):
        history = history_for([.2] * 16)
        history.append(deepcopy(history[-1]))
        history[-1]['epoch'] = 16
        with self.assertRaises(ValueError):
            replay_history(history, 100, policy_for_epochs(100))
        history = history_for([.2] * 5)
        history[-1]['training_control']['bad_epochs'] = 0
        with self.assertRaises(ValueError):
            replay_history(history, 100, policy_for_epochs(100))

    def test_valid_early_and_full_budget_completion(self):
        for aps, budget, reason in (([.2] * 16, 100, 'early_stopping'),
                ([i / 100 for i in range(100)], 100, 'max_epochs'),
                ([i / 100 for i in range(85)] + [.84] * 15, 100, 'early_stopping'),
                ([.2] * 30, 30, 'max_epochs'), ([.2] * 2, 2, 'max_epochs')):
            d, s, h = completion_for(aps, budget)
            result = validate_completion(d, s, h, budget, policy_for_epochs(budget))
            self.assertEqual(result['actual_epochs'], len(aps))
            self.assertEqual(result['stop_reason'], reason)

    def test_history_bound_checked_before_row_processing(self):
        with self.assertRaises(ValueError):
            replay_history([None] * 101, 100, policy_for_epochs(100))
        with self.assertRaises(ValueError):
            replay_history([], 100, policy_for_epochs(100))

    def test_completion_rejects_forged_fields_and_premature_stop(self):
        d, s, h = completion_for([.2] * 15)
        with self.assertRaises(ValueError):
            validate_completion(d, s, h, 100, policy_for_epochs(100))
        d, s, h = completion_for([.2] * 16)
        for target, key, value in (('d', 'stop_reason', 'max_epochs'),
                ('d', 'actual_epochs', 100), ('s', 'completed_epochs', 100),
                ('s', 'epoch', 1), ('s', 'validation_ap', .3),
                ('d', 'selection_frozen', False)):
            changed_d, changed_s = deepcopy(d), deepcopy(s)
            (changed_d if target == 'd' else changed_s)[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_completion(changed_d, changed_s, h, 100, policy_for_epochs(100))

    def test_controller_roundtrip_after_lr_drop(self):
        control = controller()
        history = history_for([.2] * 8)
        for _ in history:
            control.step(.2)
        restored = controller()
        restored.optimizer.load_state_dict(control.optimizer.state_dict())
        restored.load_state_dict(control.state_dict(), history)
        for ap in [.2] * 3 + [.3] + [.3] * 8:
            self.assertEqual(control.step(ap), restored.step(ap))
        self.assertEqual(control.state_dict(), restored.state_dict())

    def test_controller_rejects_scheduler_tampering_and_terminal_state(self):
        control = controller()
        history = history_for([.2] * 8)
        for _ in history:
            control.step(.2)
        restored = controller()
        restored.optimizer.load_state_dict(control.optimizer.state_dict())
        saved = control.state_dict()
        saved['scheduler_state']['num_bad_epochs'] += 1
        with self.assertRaises(ValueError):
            restored.load_state_dict(saved, history)
        for _ in range(8):
            control.step(.2)
        restored.optimizer.load_state_dict(control.optimizer.state_dict())
        with self.assertRaises(ValueError):
            restored.load_state_dict(control.state_dict(), history_for([.2] * 16))


if __name__ == '__main__':
    unittest.main()
