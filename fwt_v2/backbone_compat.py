"""Explicit, fail-closed source rebinding for an unchanged frozen FWT backbone.

This is an engineering migration, never a new fit or a way to migrate diffusion
semantics. The original checkpoint and its original source tree remain intact.
Only source files are read from the old tree; its Python code is never executed.
"""
import ast
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path

from .portable import file_sha256 as sha256


# Complete local dependency closure for deterministic architecture, input
# preparation, normalization, artifact loading and validation selection metrics.
# Diffusion is deliberately outside this closure; a diffusion checkpoint can
# never use this command. The engine's deterministic call path is checked below.
BACKBONE_MODULES = ('__init__.py', 'models.py', 'normalization.py', 'fusion.py', 'propagation.py',
                    'rcda.py', 'data.py', 'geography.py', 'hourly.py', 'manifest.py',
                    'artifacts.py', 'protocol.py', 'portable.py', 'sample_cache.py')
ENGINE_FUNCTIONS = ('audited_manifest', 'seed_everything', 'source_digest',
                    'model_state_digest', 'build_det', 'to_device',
                    'deterministic_loss', 'make_loader', 'load_det',
                    'load_predictor', 'records_from_batch', 'validation_records')


def package_source_digest(package):
    """Same algorithm as engine.source_digest, parameterized by source folder."""
    package = Path(package)
    files = sorted(package.glob('*.py'))
    if not files or not (package / '__init__.py').is_file():
        raise ValueError('Expected a source tree containing the fwt_v2 package')
    return hashlib.sha256(''.join(p.name + sha256(p) for p in files).encode()).hexdigest()


def _ast_digest(node):
    text = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _engine_contract(path, strict_training_only=False):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    functions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in functions:
                raise ValueError('Ambiguous engine function definition: ' + node.name)
            functions[node.name] = node
    if any(name not in functions for name in (*ENGINE_FUNCTIONS, 'raw_probability')):
        raise ValueError('Missing protected deterministic engine function')
    # Top-level imports, constants and any executable setup must also be equal.
    top_level = ast.Module(body=[node for node in tree.body
                                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
                           type_ignores=[])
    result = {'module_setup': _ast_digest(top_level)}
    if strict_training_only:
        for name, node in sorted(functions.items()):
            if name not in ('train', 'load_warm_start_h1'):
                result['strict_function:' + name] = _ast_digest(node)
    # Follow any engine-local helper referenced by a protected function. This
    # prevents an unchanged wrapper from concealing a changed implementation.
    pending = list(ENGINE_FUNCTIONS)
    while pending:
        name = pending.pop()
        if name in result:
            continue
        node = functions[name]
        result[name] = _ast_digest(node)
        pending.extend(child.id for child in ast.walk(node)
                       if isinstance(child, ast.Name) and child.id in functions
                       and child.id not in result and child.id != 'raw_probability')
    # The common predictor starts with backbone inference and the deterministic
    # early return. Diffusion-only sampling after this branch may change. Check
    # the branch shape explicitly so a future refactor fails closed for review.
    raw = functions['raw_probability']
    expected_test = ast.dump(ast.parse('head is None', mode='eval').body,
                             include_attributes=False)
    if (len(raw.body) < 2 or not isinstance(raw.body[0], ast.Assign)
            or not isinstance(raw.body[1], ast.If)
            or ast.dump(raw.body[1].test, include_attributes=False) != expected_test
            or raw.body[1].orelse
            or not isinstance(raw.body[1].body[-1], ast.Return)):
        raise ValueError('Deterministic raw_probability prefix needs explicit compatibility review')
    result['raw_probability_deterministic_prefix'] = _ast_digest(
        ast.FunctionDef(name=raw.name, args=raw.args, body=raw.body[:2],
                        decorator_list=raw.decorator_list, returns=raw.returns,
                        type_comment=raw.type_comment))
    return result


def verify_backbone_sources(source_tree, strict_training_only=False):
    old_package = Path(source_tree).resolve() / 'fwt_v2'
    current_package = Path(__file__).resolve().parent
    if old_package == current_package:
        raise ValueError('Provide the preserved original source tree, not the current implementation')
    old_digest, new_digest = package_source_digest(old_package), package_source_digest(current_package)
    modules = {}
    protected = BACKBONE_MODULES
    if strict_training_only:
        old_names = {path.name for path in old_package.glob('*.py')}
        new_names = {path.name for path in current_package.glob('*.py')}
        if old_names != new_names:
            raise ValueError('Strict warm-start requires the identical package module set')
        protected = sorted(new_names - {'engine.py', 'backbone_compat.py'})
    for name in protected:
        old_path, new_path = old_package / name, current_package / name
        if not old_path.is_file() or not new_path.is_file() or sha256(old_path) != sha256(new_path):
            raise ValueError('Backbone source changed: ' + name)
        modules[name] = sha256(new_path)
    old_engine = _engine_contract(old_package / 'engine.py', strict_training_only)
    new_engine = _engine_contract(current_package / 'engine.py', strict_training_only)
    if old_engine != new_engine:
        changed = sorted(name for name in set(old_engine) | set(new_engine)
                         if old_engine.get(name) != new_engine.get(name))
        raise ValueError('Deterministic engine contract changed: ' + ', '.join(changed))
    return dict(original_source_tree=str(Path(source_tree).resolve()),
                current_source_tree=str(current_package.parent),
                original_source_sha256=old_digest, current_source_sha256=new_digest,
                unchanged_module_sha256=modules, unchanged_engine_ast_sha256=new_engine,
                strict_training_only=strict_training_only,
                allowed_source_changes=('Training/diagnostic paths and compatibility validator only'
                    if strict_training_only else 'Diffusion implementation, CLI and non-inference training/diagnostic paths only'),
                scope='Frozen deterministic inference equivalence; no diffusion compatibility or new training result')


def _state_equal(left, right):
    import torch
    return (left.keys() == right.keys()
            and all(isinstance(left[name], torch.Tensor) and isinstance(right[name], torch.Tensor)
                    and left[name].dtype == right[name].dtype
                    and left[name].shape == right[name].shape
                    and torch.equal(left[name], right[name]) for name in left))


def rebind_backbone(args):
    """Validate metadata and unchanged deterministic code, then write a new copy.

    Manifest path/identity checks inspect filesystem metadata only. Constructing
    RCDADataset verifies normalization/georeferences but never reads its arrays.
    No checkpoint selection, model optimization or held-out evaluation occurs.
    """
    import torch
    from .artifacts import atomic_checkpoint, atomic_json, load_checkpoint, output_path
    from .data import DataConfig, NormalizationStats, RCDADataset
    from .manifest import manifest_digest
    from .engine import audited_manifest, build_det, model_state_digest, source_digest
    from .models import FWTConfig, validate_wind_prior_config

    source = Path(args.source).resolve()
    destination = output_path(args.output)
    report_path = output_path(getattr(args, 'report', None) or destination.with_suffix('.compatibility.json'))
    if source in (destination, report_path) or destination == report_path:
        raise ValueError('Checkpoint source, destination and report must be distinct paths')
    if destination.exists() or report_path.exists():
        raise FileExistsError('Rebinding requires new checkpoint and report paths; originals are immutable')
    source_hash = sha256(source)
    bundle = load_checkpoint(source)
    if bundle.get('family') != 'det' or bundle.get('architecture') != 'fwt':
        raise ValueError('Only a deterministic FWT backbone trained on Tb can be rebound; never diffusion')
    if bundle.get('selection_frozen') is not True:
        raise ValueError('Only completed frozen validation selections can be rebound')
    if (bundle.get('completed_epochs', 0) < 1
            or bundle.get('completed_epochs') != bundle.get('requested_epochs')):
        raise ValueError('Backbone training must have completed its exact requested epoch budget')
    config = DataConfig(**bundle['data_config'])
    model_config = FWTConfig(**bundle['model_config'])
    validate_wind_prior_config(config, model_config)
    evidence = verify_backbone_sources(args.source_tree)
    if bundle.get('source_sha256') != evidence['original_source_sha256']:
        raise ValueError('Original source tree digest does not match the checkpoint provenance')
    if source_digest() != evidence['current_source_sha256']:
        raise RuntimeError('Current source digest is inconsistent with the compatibility evidence')
    manifest = audited_manifest(args.manifest)
    normalizer = NormalizationStats.load(args.normalizer)
    if (bundle.get('manifest_sha256') != manifest_digest(manifest)
            or bundle.get('normalizer_sha256') != sha256(args.normalizer)):
        raise ValueError('Exact original manifest and normalization are required')
    training = RCDADataset(manifest, 'backbone_train', normalizer, config)
    validation = RCDADataset(manifest, 'validation', normalizer, config)
    for key, rows in (('training_sample_ids', training.samples), ('validation_sample_ids', validation.samples)):
        expected = [row['sample_id'] for row in rows]
        if bundle.get(key) != expected:
            raise ValueError('Checkpoint sample identities disagree with frozen role: ' + key)
    model = build_det(model_config, 'fwt', config).cpu().eval()
    model.load_state_dict(bundle['state_dict'], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Backbone state contains nonfinite parameters or buffers')
    if not _state_equal(bundle['state_dict'], model.state_dict()):
        raise ValueError('Loading backbone changed tensor values or dtypes')
    evidence.update(schema='fwt_backbone_compatibility_v1',
                    verified_at=datetime.now().astimezone().isoformat(),
                    original_checkpoint=str(source), original_checkpoint_sha256=source_hash,
                    manifest_sha256=bundle['manifest_sha256'], normalizer_sha256=bundle['normalizer_sha256'],
                    model_state_sha256=model_state_digest(model), training_performed=False,
                    original_selection_preserved=True, arrays_read=False,
                    data_config=asdict(config))
    rebound = dict(bundle, source_sha256=evidence['current_source_sha256'], compatibility_rebinding=evidence)
    if (sha256(source) != source_hash
            or package_source_digest(Path(args.source_tree) / 'fwt_v2') != evidence['original_source_sha256']
            or source_digest() != evidence['current_source_sha256']):
        raise RuntimeError('Checkpoint or source changed during compatibility verification')
    atomic_checkpoint(destination, rebound)
    restored = load_checkpoint(destination)
    if not _state_equal(bundle['state_dict'], restored['state_dict']) or sha256(source) != source_hash:
        raise RuntimeError('Checkpoint copy did not preserve the original backbone state')
    result = dict(evidence, output_checkpoint=str(destination), output_checkpoint_sha256=sha256(destination),
                  tensor_values_unchanged=True, original_checkpoint_unchanged=True, report=str(report_path))
    atomic_json(report_path, result)
    return result
