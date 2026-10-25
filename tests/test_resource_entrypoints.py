"""Resource options must reach execution and bind the exact H1/P1 pair.

Uses the portable metadata fixture and a mocked engine; never reads real arrays,
loads weights, imports CUDA, or starts training.
"""
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from test_portable_entry import Fixture, ROOT, write_json
import train


class ResourceEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def execute_mock(self, args):
        args.execute = True
        checked = train.preflight(args)
        calls = []
        fake_torch = ModuleType('torch')
        fake_torch.set_num_threads = lambda count: None
        fake_engine = ModuleType('fwt_v2.engine')
        fake_artifacts = ModuleType('fwt_v2.artifacts')
        fake_artifacts.WORKSPACE = self.root
        fake_artifacts.atomic_json = write_json

        def run(arguments, configuration, family):
            calls.append((arguments, family))
            output = Path(arguments.output)
            output.mkdir(parents=True)
            for name in ('best.pth', 'last.pth'):
                (output / name).write_bytes(b'placeholder: never deserialize')
            write_json(output / 'diagnostics.json', dict(status='completed', completed_epochs=2))
            write_json(output / 'selection.json', dict(selection_frozen=True))
            for name in ('history.json', 'configuration.json', 'validation_selected.json'):
                write_json(output / name, {})
            return dict(family=family)

        fake_engine.train = run
        import fwt_v2
        with patch.dict(sys.modules, {'torch': fake_torch, 'fwt_v2.engine': fake_engine,
                                     'fwt_v2.artifacts': fake_artifacts}), \
                patch.object(fwt_v2, 'engine', fake_engine, create=True), \
                patch.object(fwt_v2, 'artifacts', fake_artifacts, create=True):
            receipt = train.execute(args, checked)
        self.assertEqual(len(calls), 1)
        return calls[0], checked, receipt

    def test_nondefault_resources_are_forwarded_and_bound_to_p1(self):
        args = self.fixture.args()
        args.batch_size, args.validation_batch_size, args.sample_cache_gib = 8, 2, 40.
        (forwarded, family), checked, receipt = self.execute_mock(args)
        self.assertEqual((family, forwarded.batch_size, forwarded.validation_batch_size,
                          forwarded.sample_cache_gib), ('det', 8, 2, 40.))
        self.assertEqual(receipt['binding'], checked['binding'])
        diff = self.fixture.args('diff')
        diff.batch_size, diff.validation_batch_size, diff.sample_cache_gib = 8, 2, 40.
        matched = train.preflight(diff)
        self.assertEqual(matched['binding'], checked['binding'])
        for attribute, value in [('batch_size', 4), ('validation_batch_size', 1),
                                 ('sample_cache_gib', 0.)]:
            changed = copy.copy(diff)
            setattr(changed, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(
                    ValueError, 'identical data/source/budget'):
                train.preflight(changed)

    def test_validation_batch_default_resolves_before_execution(self):
        args = self.fixture.args()
        args.batch_size = 4
        (forwarded, _), checked, _ = self.execute_mock(args)
        self.assertIsNone(args.validation_batch_size)
        self.assertEqual(forwarded.validation_batch_size, 4)
        self.assertEqual(checked['binding']['performance']['validation_batch_size'], 4)

    def test_resource_change_after_preflight_cannot_execute(self):
        args = self.fixture.args()
        args.execute = True
        checked = train.preflight(args)
        args.sample_cache_gib = 40.
        with self.assertRaisesRegex(ValueError, 'unchanged preflight'):
            train.execute(args, checked)
        self.assertFalse(Path(checked['output']).exists())


if __name__ == '__main__':
    unittest.main()
