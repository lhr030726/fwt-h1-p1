"""Audited machine-relocation adapter for frozen history trainer 574322e.

Original source, metadata and checkpoints remain unchanged. Only filesystem
stat identities and machine-specific runtime descriptions are rebound in memory.
All file/content checks, model/AdamW/RNG state, orders and recipe remain enforced.
Different hardware may change floating point results; parent AP replay still
must pass the original tolerance. This is not a bitwise-equivalence claim.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

FROZEN_SOURCE_SHA = '9721eefc84c8cb11cbb0305a948d1de104a31c0d00d2199d57c9303ebb719b9f'
MACHINE_FIELDS = {'runtime', 'array_stat_identity'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stat_content_identity(rows):
    return [{k: row[k] for k in ('sample_id', 'field', 'path', 'size')} for row in rows]


def check_relocation(key, new, old):
    if key == 'array_stat_identity':
        require(stat_content_identity(new) == stat_content_identity(old), 'Array content identities differ')
    elif key == 'runtime':
        require(set(new) == set(old), 'Runtime field set changed')
        for field in new:
            if field == 'python':
                require(re.match(r'^\d+\.\d+', new[field]).group() == re.match(r'^\d+\.\d+', old[field]).group(),
                        'Python major/minor version changed')
            elif field != 'gpu':
                require(new[field] == old[field], 'Runtime setting/build changed: ' + field)
    else:
        require(new == old, 'Immutable data/runtime field changed: ' + key)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('training_arguments', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'tools'))
    sys.path.insert(0, str(repo))
    import train_history_refiner as h
    for package, expected in {'numpy': '1.22.3', 'scipy': '1.11.4',
                              'scikit-learn': '1.2.2', 'pyproj': '3.7.1'}.items():
        require(importlib.metadata.version(package) == expected, 'Scientific dependency differs: ' + package)
    source_path = repo / 'tools/train_history_refiner.py'
    require(h.sha(source_path) == FROZEN_SOURCE_SHA, 'Adapter only accepts frozen trainer source')
    training_args = args.training_arguments
    if training_args and training_args[0] == '--':
        training_args = training_args[1:]
    parsed = h.parse_args(training_args)
    require(parsed.resume_run is not None and not parsed.preflight, 'Relocation requires --resume-run; no fresh training')
    require(not parsed.output.exists(), 'Use a fresh output directory')
    old_common = h.read_json(parsed.resume_run / 'input-binding.json')
    plan = h.make_plan(parsed)  # Rechecks every historical artifact and exact source/recipe.
    require(old_common['source_files'] == plan['source_files'], 'Original checkpoint source differs')
    audit = dict(schema='fwt_history_machine_relocation_v1', started_at=time.time(),
                 adapter_sha256=h.sha(Path(__file__)), original_source_sha256=h.sha(source_path),
                 original_run=str(parsed.resume_run), output=str(parsed.output),
                 original_plan_sha256=h.sha(parsed.resume_run / 'plan.json'),
                 original_binding_sha256=h.sha(parsed.resume_run / 'input-binding.json'),
                 allowed_rebound_fields=sorted(MACHINE_FIELDS), parent_checks={}, transactions={},
                 gpu_resume_validation='pending', bitwise_equivalence_claim=False)
    audit_path = parsed.output.parent / (parsed.output.name + '.machine-relocation.json')

    def save_audit():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        temp = audit_path.with_suffix('.tmp')
        temp.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
        temp.replace(audit_path)

    def parent_check(key, value, original):
        # run() has already verified full array SHA256 before this loop.
        check_relocation(key, value, original)
        if key in MACHINE_FIELDS:
            check_relocation(key, value, old_common[key])
        audit['parent_checks'][key] = dict(equal=value == original,
            original_sha256=h.canonical_sha(original), relocated_sha256=h.canonical_sha(value))
        if key == 'runtime':
            audit['original_runtime'], audit['relocated_runtime'] = original, value
        save_audit()

    def rebase_transaction(transaction, bindings, resume_path):
        import torch
        decoded = h.decode_transaction(transaction)  # all original checksums + AdamW integrity
        result = deepcopy(transaction)
        report = dict(original_transaction_sha256=h.sha(resume_path),
                      completed_epochs=transaction['completed_epochs'], arms={})
        for arm, kinds in decoded.items():
            original_configuration = h.read_json(resume_path.parent / arm / 'configuration.json')
            target_binding = bindings[arm]
            require(h.canonical_sha({k: v for k, v in original_configuration.items() if k not in MACHINE_FIELDS}) ==
                    h.canonical_sha({k: v for k, v in target_binding.items() if k not in MACHINE_FIELDS}),
                    'Non-machine checkpoint binding changed: ' + arm)
            for key in MACHINE_FIELDS:
                check_relocation(key, target_binding[key], original_configuration[key])
            report['arms'][arm] = {}
            for kind, original in kinds.items():
                require(h.canonical_sha(original['binding']) == h.canonical_sha(original_configuration),
                        'Original configuration differs')
                state = deepcopy(original)
                state['binding'] = deepcopy(target_binding)
                h.check_state(state, target_binding)
                # Only binding metadata was replaced; every tensor, counter, history and RNG object came from deepcopy.
                unchanged_old, unchanged_new = deepcopy(original), deepcopy(state)
                unchanged_old.pop('binding'); unchanged_new.pop('binding')
                require(h.serialize_state(unchanged_old) == h.serialize_state(unchanged_new),
                        'Model/optimizer/RNG/history state changed during migration')
                data = h.serialize_state(state)
                result['arms'][arm][kind + '_data'] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                result['arms'][arm][kind + '_sha256'] = hashlib.sha256(data).hexdigest()
                report['arms'][arm][kind] = dict(original_sha256=transaction['arms'][arm][kind + '_sha256'],
                                                derived_sha256=result['arms'][arm][kind + '_sha256'])
        h.decode_transaction(result, bindings)
        audit['transactions'][str(transaction['seed'])] = report
        audit['gpu_resume_validation'] = 'state_rebased_after_full_parent_replay'
        save_audit()
        return result

    run_source = inspect.getsource(h.run)
    old = "require(canonical_sha(value) == canonical_sha(old_common[key]), 'Parent runtime/data/H1 differs: ' + key)"
    new = '_migration_parent_check(key, value, old_common[key])'
    require(run_source.count(old) == 1, 'Unknown parent validation source layout')
    run_source = run_source.replace(old, new)
    old = "transaction = torch.load(resume_path, map_location='cpu', weights_only=True)"
    new = old + '\n            transaction = _migration_rebase_transaction(transaction, bindings, resume_path)'
    require(run_source.count(old) == 1, 'Unknown resume source layout')
    run_source = run_source.replace(old, new)
    h.__dict__.update(_migration_parent_check=parent_check, _migration_rebase_transaction=rebase_transaction)
    exec(compile(run_source, str(Path(__file__)) + ':explicit_machine_overlay', 'exec'), h.__dict__)
    audit['execution_overlay_sha256'] = hashlib.sha256(run_source.encode()).hexdigest()
    save_audit()
    if not parsed.execute:
        print(json.dumps(dict(status='relocation_dry_run_passed', audit=str(audit_path),
                              remaining='CUDA parent replay and transaction restore are pending')))
        return 0
    try:
        h.run(parsed, plan)
        audit['status'] = 'completed'
    except BaseException as error:
        audit.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        audit['updated_at'] = time.time()
        save_audit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
