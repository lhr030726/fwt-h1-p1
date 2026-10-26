"""Audited performance-source continuation; originals are never rewritten.

The manifest binds reviewed old/new source bytes. Only explicitly listed
performance/resume code may differ; scientific modules and unchanged functions
are additionally checked structurally. This does not promise bitwise CUDA
trajectory equivalence when a different validation batch kernel is used.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from .portable import file_sha256

SCHEMA = 'fwt_performance_continuation_v1'
ALLOWED = {'train.py', 'fwt_v2/engine.py', 'fwt_v2/diffusion.py',
           'fwt_v2/gradient_diagnostics.py', 'fwt_v2/validation_sampling.py',
           'fwt_v2/source_resume.py'}


def source_inventory(root):
    root = Path(root).resolve()
    paths = [root / 'train.py', *sorted((root / 'fwt_v2').glob('*.py'))]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}


def package_digest(inventory):
    return hashlib.sha256(''.join(Path(p).name + inventory[p] for p in sorted(inventory)
        if p.startswith('fwt_v2/')).encode()).hexdigest()


def _protected_ast(path, kind):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module in (
                    'gradient_diagnostics', 'validation_sampling', 'source_resume'):
                continue
        if kind == 'engine' and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ('train', 'raw_probability', 'validation_records', 'gradient_l2_norm'):
                continue
        if kind == 'diffusion' and isinstance(node, ast.ClassDef) and node.name == 'ConditionalResidualDiffusion':
            node = deepcopy(node)
            node.body = [part for part in node.body if not
                (isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                 part.name in ('sample', 'sample_parallel', 'sample_batched'))]
        # Ignore only module documentation; imports and executable setup remain bound.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        key = getattr(node, 'name', str(len(result)))
        result[key] = ast.dump(node, include_attributes=False)
    return result


def verify_manifest(path, current_root=None):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != SCHEMA:
        raise ValueError('Expected an explicit performance continuation manifest')
    current_root = Path(current_root or Path(__file__).resolve().parents[1]).resolve()
    old_root = Path(document['original_source_tree']).resolve()
    if old_root == current_root:
        raise ValueError('Preserve original source separately from optimized source')
    old, new = source_inventory(old_root), source_inventory(current_root)
    if old != document.get('original_files') or new != document.get('current_files'):
        raise ValueError('Reviewed continuation source inventory changed')
    changed = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
    if set(old) - set(new) or set(changed) - ALLOWED or changed != document.get('changed_files'):
        raise ValueError('Continuation contains unreviewed or forbidden source changes')
    for filename, kind in [('engine.py', 'engine'), ('diffusion.py', 'diffusion')]:
        if _protected_ast(old_root / 'fwt_v2' / filename, kind) != _protected_ast(current_root / 'fwt_v2' / filename, kind):
            raise ValueError('Protected scientific code changed: ' + filename)
    original_binding = document['original_entry_binding']
    if (original_binding.get('source_sha256') != package_digest(old)
            or original_binding.get('entry_sha256') != old['train.py']):
        raise ValueError('Original entry binding does not match preserved source')
    checkpoint = document['checkpoint']
    parallel = document.get('approved_p1_validation_parallel_samples', 1)
    if type(parallel) is not int or not 1 <= parallel <= 128:
        raise ValueError('Reviewed validation parallelism must be an integer from 1 to 128')
    if (checkpoint.get('phase') not in ('det', 'diff') or type(checkpoint.get('completed_epochs')) is not int
            or checkpoint['completed_epochs'] < 1 or file_sha256(checkpoint['path']) != checkpoint.get('sha256')):
        raise ValueError('Preserved full-epoch checkpoint changed or has invalid metadata')
    evidence = dict(schema=SCHEMA, manifest_path=str(path), manifest_sha256=file_sha256(path),
        original_source_sha256=package_digest(old), current_source_sha256=package_digest(new),
        original_entry_sha256=old['train.py'], current_entry_sha256=new['train.py'],
        checkpoint=deepcopy(checkpoint), changed_files=changed,
        approved_p1_validation_parallel_samples=parallel,
        scope='Exact saved model/optimizer/RNG/history continuation with reviewed performance source changes')
    return dict(document=document, evidence=evidence)


def bind_entry(binding, verified, phase):
    """The scientific entry contract stays exactly equal across segments."""
    original = deepcopy(verified['document']['original_entry_binding'])
    expected = deepcopy(original)
    expected['epochs'] = original.get('stage_epochs', {}).get(phase, original['epochs'])
    expected['source_sha256'] = verified['evidence']['current_source_sha256']
    expected['entry_sha256'] = verified['evidence']['current_entry_sha256']
    if 'warm_start_h1' in original:
        binding['warm_start_h1'] = deepcopy(original['warm_start_h1'])
    if binding != expected:
        changed = sorted(k for k in set(binding) | set(expected) if binding.get(k) != expected.get(k))
        raise ValueError('Continuation scientific entry recipe changed: ' + ', '.join(changed))
    binding['continuation'] = deepcopy(verified['evidence'])
    return binding


def verify_runtime_parallelism(evidence):
    raw = os.environ.get('FWT_P1_VALIDATION_PARALLEL_SAMPLES', '1')
    if (not raw.isascii() or not raw.isdecimal()
            or int(raw) != evidence['approved_p1_validation_parallel_samples']):
        raise ValueError('Runtime P1 validation parallelism differs from reviewed continuation')


def read_snapshot(path, manifest_path, family):
    import torch
    from .training_state import load_training_state
    checked = verify_manifest(manifest_path)
    verify_runtime_parallelism(checked['evidence'])
    checkpoint = checked['document']['checkpoint']
    if Path(path).resolve() != Path(checkpoint['path']).resolve() or family != checkpoint['phase']:
        raise ValueError('Resume phase/path differs from the preserved checkpoint')
    preliminary = torch.load(path, map_location='cpu', weights_only=True)
    original = load_training_state(path, preliminary['binding'])
    if file_sha256(path) != checkpoint['sha256']:
        raise ValueError('Preserved checkpoint changed while being read')
    if (original['completed_epochs'] != checkpoint['completed_epochs']
            or original['binding']['provenance']['family'] != family
            or original['binding']['provenance']['source_sha256'] != checked['evidence']['original_source_sha256']):
        raise ValueError('Actual resume state differs from continuation evidence')
    return dict(original=original, evidence=checked['evidence'])


def inherit_initialization(provenance, snapshot):
    original = snapshot['original']['binding']['provenance']
    for key in ('initialization', 'warm_start'):
        if key in original:
            provenance[key] = deepcopy(original[key])
    provenance['source_migration'] = deepcopy(snapshot['evidence'])


def migrate_snapshot(snapshot, binding):
    """Create an in-memory continuation; tensor/optimizer/RNG bytes stay original."""
    original = snapshot['original']
    comparison = deepcopy(binding)
    comparison['provenance'].pop('source_migration', None)
    comparison['provenance']['source_sha256'] = original['binding']['provenance']['source_sha256']
    performance = comparison['provenance']['performance']
    approved = snapshot['evidence']['approved_p1_validation_parallel_samples']
    if performance.get('p1_validation_parallel_samples', 1) != approved:
        raise ValueError('Runtime P1 validation parallelism differs from reviewed continuation')
    old_performance = original['binding']['provenance']['performance']
    if 'p1_validation_parallel_samples' in old_performance:
        performance['p1_validation_parallel_samples'] = old_performance['p1_validation_parallel_samples']
    else:
        performance.pop('p1_validation_parallel_samples', None)
    if comparison != original['binding']:
        changed = sorted(k for k in set(comparison) | set(original['binding'])
                         if comparison.get(k) != original['binding'].get(k))
        raise ValueError('Resume scientific/runtime/input contract changed: ' + ', '.join(changed))
    migrated = deepcopy(original)
    migrated['binding'] = deepcopy(binding)
    selected = migrated['selected']
    selected['selection_origin'] = dict(
        original_checkpoint_sha256=snapshot['evidence']['checkpoint']['sha256'],
        original_metadata={k: deepcopy(v) for k, v in selected.items() if k != 'state_dict'},
        model_state_unchanged=True, historical_validation_unchanged=True)
    for key, value in binding['provenance'].items():
        selected[key] = deepcopy(value)
    migrated['diagnostics']['source_sha256'] = binding['provenance']['source_sha256']
    migrated['diagnostics']['source_migration'] = deepcopy(snapshot['evidence'])
    return migrated
