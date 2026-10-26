"""CPU checks for identity rejection and exact trained-state preservation."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = next((parent for parent in HERE.parents if (parent / 'tools/train_history_refiner.py').is_file()),
            HERE.parent / 'p1-redesign/repo')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'tests'))
import resume_on_new_server as adapter
import test_history_training as fixture


class RelocationTests(unittest.TestCase):
    def test_stat_relocation_accepts_machine_identity_only(self):
        old = [dict(sample_id='a', field='input_path', path='/same/a.npy', size=10, inode=1, device=2, mtime_ns=3)]
        new = deepcopy(old)
        new[0].update(inode=100, device=200, mtime_ns=300)
        adapter.check_relocation('array_stat_identity', new, old)
        for key, changed in [('sample_id', 'b'), ('path', '/other'), ('size', 11), ('field', 'label_path')]:
            invalid = deepcopy(new); invalid[0][key] = changed
            with self.assertRaises(ValueError):
                adapter.check_relocation('array_stat_identity', invalid, old)

    def test_runtime_preserves_torch_and_numerical_settings(self):
        old = dict(python='3.10.12 build-old', torch='2.7.1+cu128', gpu='old GPU', cpu_threads=2)
        new = dict(old, python='3.10.20 build-new', gpu='new GPU')
        adapter.check_relocation('runtime', new, old)
        for key, changed in [('python', '3.11.2'), ('torch', '2.8.0'), ('cpu_threads', 4)]:
            with self.assertRaises(ValueError):
                adapter.check_relocation('runtime', dict(new, **{key: changed}), old)

    def test_rebind_preserves_trained_model_optimizer_rng_and_rejects_recipe_change(self):
        import torch
        heads, optimizers, bindings, tx = fixture.fixture()
        tx = fixture.advance(heads, optimizers, bindings, tx, ap=.30)
        h = fixture.runner
        decoded = h.decode_transaction(tx)
        old_runtime = dict(python='3.10.12 build-old', torch='2.7.1+cu128', gpu='old GPU', cpu_threads=2)
        for arm in h.ARMS:
            for kind, state in decoded[arm].items():
                state['binding']['runtime'] = old_runtime
                state['binding']['array_stat_identity'] = []
                data = h.serialize_state(state)
                tx['arms'][arm][kind + '_data'] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                tx['arms'][arm][kind + '_sha256'] = hashlib.sha256(data).hexdigest()
        decoded = h.decode_transaction(tx)
        target = {arm: deepcopy(decoded[arm]['last']['binding']) for arm in h.ARMS}
        for binding in target.values():
            binding['runtime'] = dict(old_runtime, gpu='new GPU')
        tree = ast.parse((HERE / 'resume_on_new_server.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'rebase_transaction')
        namespace = dict(deepcopy=deepcopy, h=h, require=adapter.require, MACHINE_FIELDS=adapter.MACHINE_FIELDS,
                         check_relocation=adapter.check_relocation, hashlib=hashlib,
                         audit={'transactions': {}}, save_audit=lambda: None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<test-adapter>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in h.ARMS:
                (root / arm).mkdir()
                (root / arm / 'configuration.json').write_text(json.dumps(decoded[arm]['last']['binding']))
            resume = root / 'transaction.pth'; torch.save(tx, resume)
            result = namespace['rebase_transaction'](tx, target, resume)
            restored = h.decode_transaction(result, target)
            self.assertEqual(restored['true_delta']['last']['completed_epochs'], 1)
            self.assertEqual(restored['zero_delta']['last']['optimizer_integrity'], decoded['zero_delta']['last']['optimizer_integrity'])
            target['true_delta']['epochs'] = 51
            with self.assertRaises(ValueError):
                namespace['rebase_transaction'](tx, target, resume)


if __name__ == '__main__':
    unittest.main()
