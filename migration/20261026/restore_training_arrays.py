"""Restore unchanged RCDA arrays from the user's original dataset, with SHA checks.

No array transformation and no overwrite. The default roles are the frozen
history branch's Tr/V; --all-development also restores H1 backbone training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/fwt-h1-p1/data'))
    p.add_argument('--all-development', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.index.read_text())['files']
    selected = [r for r in rows if a.all_development or r['role'] in ('residual_train', 'validation')]
    count, total = 0, 0
    for row in selected:
        assert row['path'].startswith('data/arrays/'), row['path']
        relative = Path(row['path'][len('data/arrays/'):])
        assert not relative.is_absolute() and '..' not in relative.parts
        source, target = a.dataset / relative, a.data_root / 'arrays' / relative
        assert source.is_file() and source.stat().st_size == row['size'], str(source)
        assert sha(source) == row['sha256'], 'Original dataset hash mismatch: ' + str(source)
        if not a.verify_only:
            if target.exists():
                assert target.is_file() and sha(target) == row['sha256'], 'Refusing to overwrite: ' + str(target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                assert sha(target) == row['sha256'], str(target)
        count += 1
        total += row['size']
    receipt = dict(status='verified' if a.verify_only else 'restored', files=count,
                   bytes=total, roles='all_development' if a.all_development else 'residual_train_validation',
                   dataset=str(a.dataset.resolve()), restoration_performed=not a.verify_only,
                   data_root=str(a.data_root.resolve()) if not a.verify_only else None,
                   arrays_index_sha256=sha(a.index), all_sha256_equal=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
