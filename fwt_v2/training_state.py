"""Full-epoch restart state, deliberately distinct from a frozen predictor.

Only Python primitives and tensors are serialized. Restart never extends the
predeclared budget, changes the recipe, or infers which checkpoint to select.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .artifacts import atomic_checkpoint
from .training_control import policy_for_epochs, replay_history

SCHEMA = 'fwt_training_state_v2'


def optimizer_state_integrity(saved):
    """Bind the exact states that existed at save time, including their bytes.

    AdamW legitimately omits parameters that have never received a gradient.
    Record its actual state identities rather than requiring every parameter.
    This detects accidental corruption, not coordinated malicious rewriting of
    both the optimizer state and this digest inside the same checkpoint.
    """
    if (not isinstance(saved, dict) or not isinstance(saved.get('state'), dict) or
            not isinstance(saved.get('param_groups'), list) or
            any(type(index) is not int for index in saved['state'])):
        raise ValueError('Invalid AdamW state for integrity verification')
    identities = sorted(saved['state'])
    digest = hashlib.sha256()
    digest.update(json.dumps(saved['param_groups'], sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode())
    for index in identities:
        state = saved['state'][index]
        if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
            raise ValueError('Invalid AdamW parameter state for integrity verification')
        digest.update(f'\nparameter:{index}\n'.encode())
        for name in sorted(state):
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError('AdamW state must contain tensor moments and update steps')
            tensor = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                     separators=(',', ':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(schema='fwt_optimizer_integrity_v1', state_parameter_ids=identities,
                sha256=digest.hexdigest())


def verify_optimizer_integrity(bundle):
    expected = bundle.get('optimizer_integrity')
    if (not isinstance(expected, dict) or
            expected != optimizer_state_integrity(bundle['optimizer_state_dict'])):
        raise ValueError('AdamW integrity mismatch: saved parameter states or contents changed')


def sample_file_identity(samples):
    """Metadata-only identity for the actual training/validation input files.

    Detects ordinary replacement/editing without reading array contents. This
    is not a cryptographic digest of dataset bytes; immutable data is required.
    Calibration and final-role files are never opened or fingerprinted here.
    """
    result = []
    for sample in samples:
        for key in ('input_path', 'label_path'):
            path = Path(sample[key]).resolve()
            stat = path.stat()
            result.append(dict(sample_id=sample['sample_id'], field=key, path=str(path),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino))
    return result


def training_binding(provenance, config, sampling, samples):
    # JSON canonicalization removes tuple/list representation differences and
    # fails on nonfinite values. Output paths and controlled stop points are
    # intentionally absent; the original TOTAL requested epochs remains bound.
    value = dict(provenance=provenance, configuration=config, sampling=sampling,
        input_files=sample_file_identity(samples),
        runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=str(np.__version__),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            interop_threads=torch.get_num_interop_threads()))
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def capture_rng(device):
    numpy_state = np.random.get_state()
    result = dict(python=random.getstate(),
        numpy=dict(kind=numpy_state[0], keys=numpy_state[1].tolist(), position=numpy_state[2],
                   has_gauss=numpy_state[3], cached_gaussian=numpy_state[4]),
        torch_cpu=torch.get_rng_state().clone(), torch_cuda=None)
    # Do not query/initialize CUDA at all for a CPU job.
    if torch.device(device).type == 'cuda':
        result['torch_cuda'] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['kind'], np.asarray(numpy_state['keys'], dtype=np.uint32),
                        numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if torch.device(device).type == 'cuda':
        if state['torch_cuda'] is None:
            raise ValueError('CUDA restart requires the original CUDA RNG states')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    elif state['torch_cuda'] is not None:
        raise ValueError('CPU restart cannot consume CUDA RNG state')


def validate_rng(state, device):
    """Validate using isolated generators, without changing process RNG state."""
    try:
        random.Random().setstate(state['python'])
        numpy_state = state['numpy']
        np.random.RandomState().set_state((numpy_state['kind'],
            np.asarray(numpy_state['keys'], dtype=np.uint32), numpy_state['position'],
            numpy_state['has_gauss'], numpy_state['cached_gaussian']))
        torch.Generator(device='cpu').set_state(state['torch_cpu'])
        if torch.device(device).type == 'cpu' and state['torch_cuda'] is not None:
            raise ValueError('Unexpected CUDA state in CPU restart')
        if torch.device(device).type == 'cuda' and not isinstance(state['torch_cuda'], list):
            raise ValueError('Missing CUDA RNG states')
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError('Invalid RNG state in full-epoch restart') from exc


def restore_training_parameters(bundle, model, optimizer):
    """Check tensor structure and actual optimizer settings before mutation."""
    verify_optimizer_integrity(bundle)
    expected = model.state_dict()
    _check_model_state(bundle['model_state_dict'], expected)
    _check_model_state(bundle['selected']['state_dict'], expected)
    saved = bundle['optimizer_state_dict']
    current = optimizer.state_dict()
    progress = validate_control_state(bundle)
    expected_groups = deepcopy(current['param_groups'])
    rates = progress['learning_rates_after']
    if len(expected_groups) != len(rates):
        raise ValueError('Restart AdamW parameter-group count differs')
    for group, rate in zip(expected_groups, rates):
        group['lr'] = rate
    if saved.get('param_groups') != expected_groups:
        raise ValueError('Restart AdamW settings or replayed learning rates differ from the bound recipe')
    parameters = {index: parameter for serialized, group in zip(current['param_groups'], optimizer.param_groups)
                  for index, parameter in zip(serialized['params'], group['params'])}
    if not isinstance(saved.get('state'), dict) or not set(saved['state']) <= set(parameters):
        raise ValueError('Restart AdamW state has unknown parameter identities')
    for index, state in saved['state'].items():
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state):
            raise ValueError('Incomplete AdamW moment state')
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or
                not 0 <= step.item() <= bundle['updates'] or step.item() != int(step.item())):
            raise ValueError('Invalid AdamW update step')
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state:
                tensor = state[name]
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[index].shape or
                        tensor.dtype != parameters[index].dtype or not torch.isfinite(tensor).all()):
                    raise ValueError('Invalid AdamW moment tensor: ' + name)
                if name in ('exp_avg_sq', 'max_exp_avg_sq') and (tensor < 0).any():
                    raise ValueError('Negative AdamW second moment: ' + name)
    model.load_state_dict(bundle['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved)


def _check_model_state(state, expected):
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Restart model state keys differ from the bound architecture')
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or
                value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid restart model tensor: ' + name)


def save_training_state(path, *, binding, model, optimizer, device, completed_epochs,
                        history, selected, selected_validation, diagnostics, updates, best,
                        training_control_state):
    bundle = dict(schema=SCHEMA, binding=deepcopy(binding), completed_epochs=completed_epochs,
        model_state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        optimizer_state_dict=deepcopy(optimizer.state_dict()), rng_state=capture_rng(device),
        history=deepcopy(history), selected=deepcopy(selected), selected_validation=deepcopy(selected_validation),
        diagnostics=deepcopy(diagnostics), updates=updates, best_validation_ap=best,
        training_control_state=deepcopy(training_control_state))
    bundle['optimizer_integrity'] = optimizer_state_integrity(bundle['optimizer_state_dict'])
    atomic_checkpoint(path, bundle)


def validate_control_state(bundle):
    """Metadata replay guards adaptive restarts before touching model/optimizer."""
    provenance = bundle['binding']['provenance']
    requested = provenance['requested_epochs']
    policy = policy_for_epochs(requested)
    if provenance.get('training_control') != policy:
        raise ValueError('Restart training-control policy differs from the epoch recipe')
    rates = [provenance['learning_rate']]
    progress = replay_history(bundle['history'], requested, policy, rates)
    saved = bundle.get('training_control_state')
    if (not isinstance(saved, dict) or saved.get('schema') != 'fwt_training_control_state_v1' or
            saved.get('max_epochs') != requested or saved.get('policy') != policy or
            saved.get('initial_lrs') != rates or saved.get('progress') != progress or
            bundle['completed_epochs'] != progress['completed_epochs'] or
            bundle['diagnostics'].get('training_control_state') != progress):
        raise ValueError('Restart controller state differs from replayed validation history')
    if (progress['should_stop'] or progress['completed_epochs'] == requested or
            bundle['diagnostics'].get('stop_reason') is not None or
            bundle['diagnostics'].get('status') in ('finishing', 'completed') or
            bundle['diagnostics'].get('selection_frozen') is True):
        raise ValueError('Completed or early-stopped training cannot be resumed')
    return progress


def load_training_state(path, binding):
    """Reject changed recipe, exhausted budget, and inconsistent epoch history."""
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    required = {'schema', 'binding', 'completed_epochs', 'model_state_dict', 'optimizer_state_dict',
                'rng_state', 'history', 'selected', 'selected_validation', 'diagnostics', 'updates',
                'best_validation_ap', 'optimizer_integrity', 'training_control_state'}
    if not isinstance(bundle, dict) or bundle.get('schema') != SCHEMA or not required <= bundle.keys():
        raise ValueError('Expected a complete full-epoch FWT training state')
    verify_optimizer_integrity(bundle)
    if bundle['binding'] != binding:
        changed = sorted(key for key in set(bundle['binding']) | set(binding)
                         if bundle['binding'].get(key) != binding.get(key))
        raise ValueError('Strict resume binding mismatch: ' + ', '.join(changed))
    completed = bundle['completed_epochs']
    requested = binding['provenance']['requested_epochs']
    if type(completed) is not int or completed < 1 or completed > requested:
        raise ValueError('Invalid completed epoch count in restart state')
    if completed == requested:
        raise ValueError('Requested training budget already completed; resume cannot extend it')
    validate_control_state(bundle)
    history = bundle['history']
    diagnostics = bundle['diagnostics']
    if (not isinstance(history, list) or len(history) != completed or
            [row['epoch'] for row in history] != list(range(completed)) or
            len(diagnostics['epochs']) != completed or
            diagnostics['epochs'] != [row['training_checks'] for row in history]):
        raise ValueError('Restart history does not cover exactly the completed epochs')
    if (diagnostics.get('completed_epochs') != completed or
            diagnostics.get('remaining_epochs') != requested - completed):
        raise ValueError('Restart diagnostic epoch counts disagree with the saved boundary')
    updates = bundle['updates']
    expected_updates = 0
    for row in history:
        checks = row['training_checks']
        expected_updates += checks['updates']
        if checks['cumulative_updates'] != expected_updates:
            raise ValueError('Inconsistent cumulative updates in restart history')
    if type(updates) is not int or updates != expected_updates or diagnostics['total_updates'] != updates:
        raise ValueError('Restart update count disagrees with completed history')
    aps = [row['validation']['pooled_ap'] for row in history]
    if not all(type(ap) in (float, int) and np.isfinite(ap) and 0 <= ap <= 1 for ap in aps):
        raise ValueError('Invalid validation AP in restart history')
    best = max(aps)
    selected_epoch = aps.index(best)
    selected, report = bundle['selected'], bundle['selected_validation']
    if (bundle['best_validation_ap'] != best or selected['epoch'] != selected_epoch or
            selected['validation_ap'] != best or selected.get('selection_frozen') is not False or
            report['epoch'] != selected_epoch or report['validation'] != history[selected_epoch]['validation'] or
            report['operating_points'] != history[selected_epoch]['validation_operating_points']):
        raise ValueError('Restart selected checkpoint violates maximum AP / earliest epoch rule')
    for key, value in binding['provenance'].items():
        if selected.get(key) != value:
            # Config tuples are normalized before binding and torch save retains
            # the original representation, so compare JSON-compatible values.
            if json.loads(json.dumps(selected.get(key))) != value:
                raise ValueError('Restart selected checkpoint provenance differs: ' + key)
    validate_rng(bundle['rng_state'], binding['provenance']['device'])
    return bundle
