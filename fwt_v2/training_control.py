"""Frozen validation-AP training control; metadata validation imports no ML stack."""
from copy import deepcopy
import math


def policy_for_epochs(epochs):
    if type(epochs) is not int or epochs not in (2, 30, 100):
        raise ValueError('Supported budgets are fixed 2/30 or adaptive 100 epochs')
    if epochs != 100:
        return None
    return dict(schema='fwt_training_control_v1', max_epochs=100,
                monitor='pooled_validation_new_pixel_day_AP_max', mode='max',
                early_stop_patience=15, min_delta=0.0,
                scheduler=dict(name='ReduceLROnPlateau', mode='max', factor=0.5,
                    patience=5, threshold=0.0, threshold_mode='abs', cooldown=0,
                    min_lr=1e-5, eps=1e-8))


def _check_policy(max_epochs, policy):
    if policy != policy_for_epochs(max_epochs):
        raise ValueError('Training-control policy differs from the frozen epoch recipe')


def _ap(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Validation AP must be finite in [0, 1]')
    return float(value)


def initial_state(initial_lrs):
    if (not isinstance(initial_lrs, list) or not initial_lrs or
            any(type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0 for lr in initial_lrs)):
        raise ValueError('Positive finite initial learning rates are required')
    return dict(completed_epochs=0, best_validation_ap=None, best_epoch=None,
                bad_epochs=0, scheduler_bad_epochs=0, should_stop=False,
                learning_rates_before=list(initial_lrs), learning_rates_after=list(initial_lrs))


def advance_state(state, ap, policy):
    """Pure metadata transition, also used to replay saved history independently."""
    ap = _ap(ap)
    if state['should_stop']:
        raise ValueError('Cannot advance after the registered early-stop boundary')
    result = deepcopy(state)
    result['learning_rates_before'] = list(state['learning_rates_after'])
    improved = state['best_validation_ap'] is None or ap > state['best_validation_ap']
    if improved:
        result.update(best_validation_ap=ap, best_epoch=state['completed_epochs'], bad_epochs=0,
                      scheduler_bad_epochs=0)
    else:
        result['bad_epochs'] += 1
        if policy is not None:
            result['scheduler_bad_epochs'] += 1
    if policy is not None:
        scheduler = policy['scheduler']
        if result['scheduler_bad_epochs'] > scheduler['patience']:
            result['learning_rates_after'] = [
                max(lr * scheduler['factor'], scheduler['min_lr'])
                if lr - max(lr * scheduler['factor'], scheduler['min_lr']) > scheduler['eps'] else lr
                for lr in result['learning_rates_before']]
            result['scheduler_bad_epochs'] = 0
        result['should_stop'] = result['bad_epochs'] >= policy['early_stop_patience']
    result['completed_epochs'] += 1
    return result


def replay_history(history, max_epochs, policy, initial_lrs=None):
    """Bound replay before iteration; reject edited counters, LR, AP and extra epochs."""
    _check_policy(max_epochs, policy)
    if not isinstance(history, list) or not 1 <= len(history) <= max_epochs:
        raise ValueError('History must contain 1..max_epochs completed epochs')
    state = initial_state([0.001] if initial_lrs is None else initial_lrs)
    for epoch, row in enumerate(history):
        if not isinstance(row, dict) or type(row.get('epoch')) is not int or row['epoch'] != epoch:
            raise ValueError('History epochs must be contiguous and zero based')
        validation = row.get('validation')
        if not isinstance(validation, dict):
            raise ValueError('Missing validation metadata')
        state = advance_state(state, validation.get('pooled_ap'), policy)
        # Fixed legacy histories lack control snapshots. New histories must agree.
        if (policy is not None or 'training_control' in row) and row.get('training_control') != state:
            raise ValueError('History training-control snapshot differs from AP/LR replay')
    return state


def validate_completion(diagnostic, selection, history, max_epochs, policy):
    """Validate a frozen completion using metadata only; return its exact budget."""
    state = replay_history(history, max_epochs, policy)
    actual = len(history)
    reason = 'early_stopping' if state['should_stop'] else 'max_epochs'
    if not state['should_stop'] and actual != max_epochs:
        raise ValueError('Premature completion without the registered early-stop condition')
    if (not isinstance(diagnostic, dict) or not isinstance(selection, dict) or
            diagnostic.get('status') != 'completed' or diagnostic.get('selection_frozen') is not True or
            selection.get('selection_frozen') is not True or
            diagnostic.get('completed_epochs') != actual or selection.get('completed_epochs') != actual or
            selection.get('epoch') != state['best_epoch'] or
            selection.get('validation_ap') != state['best_validation_ap'] or
            diagnostic.get('selected_epoch') != state['best_epoch'] or
            diagnostic.get('remaining_epochs') != 0):
        raise ValueError('Completion must freeze the actual maximum-AP / earliest-tie selection')
    for payload in (diagnostic, selection):
        if payload.get('training_control') != policy:
            raise ValueError('Completed training-control policy differs')
        for key, value in dict(actual_epochs=actual, max_epochs=max_epochs, stop_reason=reason,
                               training_control_state=state).items():
            if (policy is not None or key in payload) and payload.get(key) != value:
                raise ValueError('Completion metadata differs from history replay: ' + key)
    if (selection.get('requested_epochs') != max_epochs or
            ('budget_remaining_epochs' in diagnostic and
             diagnostic['budget_remaining_epochs'] != max_epochs - actual)):
        raise ValueError('Completion budget differs from the registered maximum')
    if ('epochs' in diagnostic and diagnostic['epochs'] != [row.get('training_checks') for row in history]):
        raise ValueError('Diagnostic epoch checks differ from completed history')
    return dict(actual_epochs=actual, max_epochs=max_epochs, stop_reason=reason,
                selected_epoch=state['best_epoch'], best_validation_ap=state['best_validation_ap'])


class TrainingController:
    """The sole Torch-dependent interface; constructing it never touches a GPU."""
    def __init__(self, optimizer, max_epochs, policy):
        _check_policy(max_epochs, policy)
        self.optimizer, self.max_epochs, self.policy = optimizer, max_epochs, deepcopy(policy)
        self.initial_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.state = initial_state(self.initial_lrs)
        self.scheduler = None
        if policy is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            settings = dict(policy['scheduler'])
            settings.pop('name')
            self.scheduler = ReduceLROnPlateau(optimizer, **settings)

    def step(self, ap):
        if self.state['completed_epochs'] >= self.max_epochs:
            raise ValueError('Training budget already completed')
        if [group['lr'] for group in self.optimizer.param_groups] != self.state['learning_rates_after']:
            raise ValueError('Optimizer learning rate changed outside the frozen scheduler')
        expected = advance_state(self.state, ap, self.policy)
        if self.scheduler is not None:
            self.scheduler.step(ap)
        if [group['lr'] for group in self.optimizer.param_groups] != expected['learning_rates_after']:
            raise ValueError('PyTorch scheduler disagrees with the registered LR policy')
        self.state = expected
        return deepcopy(expected)

    def state_dict(self):
        return dict(schema='fwt_training_control_state_v1', max_epochs=self.max_epochs,
                    policy=deepcopy(self.policy), initial_lrs=list(self.initial_lrs),
                    progress=deepcopy(self.state),
                    scheduler_state=deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None)

    def load_state_dict(self, saved, history):
        expected = replay_history(history, self.max_epochs, self.policy, self.initial_lrs)
        if (not isinstance(saved, dict) or saved.get('schema') != 'fwt_training_control_state_v1' or
                saved.get('max_epochs') != self.max_epochs or saved.get('policy') != self.policy or
                saved.get('initial_lrs') != self.initial_lrs or saved.get('progress') != expected or
                [group['lr'] for group in self.optimizer.param_groups] != expected['learning_rates_after']):
            raise ValueError('Training-control state differs from its recipe/history/optimizer')
        if expected['should_stop'] or expected['completed_epochs'] == self.max_epochs:
            raise ValueError('Completed/early-stopped training cannot be resumed')
        if self.scheduler is None:
            if saved.get('scheduler_state') is not None:
                raise ValueError('Fixed recipe must not restore a scheduler')
        else:
            # Version-specific auxiliary fields remain bound to this runtime's
            # fresh scheduler; only the replayed progress fields may differ.
            scheduler_state = deepcopy(self.scheduler.state_dict())
            scheduler_state.update(best=expected['best_validation_ap'],
                num_bad_epochs=expected['scheduler_bad_epochs'], cooldown_counter=0,
                last_epoch=expected['completed_epochs'], _last_lr=expected['learning_rates_after'])
            if saved.get('scheduler_state') != scheduler_state:
                raise ValueError('Saved scheduler state differs from validation-history replay')
            self.scheduler.load_state_dict(saved['scheduler_state'])
        self.state = expected
