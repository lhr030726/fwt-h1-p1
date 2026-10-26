"""Read-only original selection, phase budgets, and explicit optimizer recipe."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from test_portable_entry import Fixture, ROOT, write_json
import train
import fwt_v2
from fwt_v2.portable import file_sha256


class ExtendedEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)
        self.args = self.f.args()
        self.args.p1_zero_init = True
        checked = train.preflight(self.args)
        self.old_binding = copy.deepcopy(checked['binding'])
        self.source = self.root / 'original-source'
        self.source.mkdir()
        (self.source / 'train.py').write_text('original entry fixture', encoding='utf-8')
        self.old_binding['entry_sha256'] = file_sha256(self.source / 'train.py')
        self.compat = dict(original_source_tree=str(self.source),
            original_source_sha256=self.old_binding['source_sha256'],
            current_source_sha256=checked['binding']['source_sha256'], strict_training_only=True)
        self.old = Path(checked['output'])
        self.old.mkdir(parents=True)
        for name in ('best.pth', 'last.pth'):
            (self.old / name).write_bytes(b'synthetic placeholder, never deserialize')
        self.configuration = dict(schema='fwt_checkpoint_v2', family='det', architecture='fwt',
            requested_epochs=2, model_config=dict(train.MODEL, wind_scale=1., slope_scale=1.),
            data_config=train._expanded_data(self.f.config['data']),
            source_sha256=self.old_binding['source_sha256'],
            manifest_file_sha256=self.old_binding['files']['manifest'], manifest_sha256='a' * 64,
            normalizer_sha256=self.old_binding['files']['normalizer'],
            training_sample_ids=[self.f.rows[0]['sample_id']],
            validation_sample_ids=[self.f.rows[2]['sample_id']],
            batch_size=1, learning_rate=.001, seed=0)
        self.selection = dict(self.configuration, selection_frozen=True, completed_epochs=2,
            total_updates=2, epoch=1, validation_ap=.3,
            selection_metric='pooled_validation_new_pixel_day_AP_max', selection_tie_rule='earliest_epoch')
        self.history = [dict(epoch=i, validation=dict(pooled_ap=ap)) for i, ap in enumerate((.2, .3))]
        self.validation = dict(selection_frozen=True, checkpoint_sha256=file_sha256(self.old / 'best.pth'),
            source_sha256=self.old_binding['source_sha256'], manifest_sha256='a' * 64,
            normalizer_sha256=self.old_binding['files']['normalizer'], epoch=1,
            validation=self.history[1]['validation'], validation_sample_ids=self.configuration['validation_sample_ids'])
        self.diagnostic = dict(status='completed', family='det', completed_epochs=2,
            remaining_epochs=0, total_updates=2, source_sha256=self.old_binding['source_sha256'])
        self.refresh()
        self.args.run_dir = self.root / 'continued-run'
        self.args.warm_start_h1_run = self.old.parent.parent
        self.args.warm_start_h1_source_tree = self.source
        self.args.h1_learning_rate = self.args.p1_learning_rate = .0001
        self.args.h1_epochs = self.args.p1_epochs = 2
        self.mock_compat = patch('fwt_v2.backbone_compat.verify_backbone_sources', return_value=self.compat)
        self.mock_compat.start()

    def tearDown(self):
        self.mock_compat.stop()
        self.temp.cleanup()

    def refresh(self):
        for name, value in [('configuration.json', self.configuration), ('selection.json', self.selection),
                ('history.json', self.history), ('diagnostics.json', self.diagnostic),
                ('validation_selected.json', self.validation)]:
            write_json(self.old / name, value)
        self.receipt = dict(schema_version='fwt_portable_training_completed_v1', status='completed',
            phase='det', binding=self.old_binding, smoke_only=True, heldout_evaluation=False,
            files={p.name: file_sha256(p) for p in self.old.iterdir() if p.name != 'portable_completed.json'})
        write_json(self.old / 'portable_completed.json', self.receipt)

    def test_lineage_preflight_reads_without_modifying_original(self):
        before = {p.name: p.read_bytes() for p in self.old.iterdir()}
        checked = train.preflight(self.args)
        lineage = checked['binding']['warm_start_h1']
        self.assertEqual(lineage['checkpoint'], str(self.old / 'best.pth'))
        self.assertEqual(lineage['selected_epoch_display'], 2)
        self.assertEqual(lineage['source_binding'], self.old_binding)
        self.assertFalse(lineage['optimizer_state_reused'])
        self.assertEqual(checked['weights_deserialized'], 0)
        self.assertIsNone(checked['backbone'])
        self.assertFalse(self.args.run_dir.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.old.iterdir()})

    def test_every_original_artifact_is_hash_bound(self):
        for name in self.receipt['files']:
            with self.subTest(name=name):
                path = self.old / name
                previous = path.read_bytes()
                path.write_bytes(previous + b' ')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    train.preflight(self.args)
                path.write_bytes(previous)

    def test_invalid_completed_selection_rejected_even_after_rehash(self):
        for document, key, bad in [(self.selection, 'selection_frozen', False),
                (self.configuration, 'family', 'diff'), (self.configuration, 'training_sample_ids', []),
                (self.diagnostic, 'total_updates', 1), (self.selection, 'epoch', 0),
                (self.validation, 'checkpoint_sha256', 'b' * 64)]:
            with self.subTest(key=key):
                previous = document[key]
                document[key] = bad
                self.refresh()
                with self.assertRaises(ValueError):
                    train.preflight(self.args)
                document[key] = previous
                self.refresh()

    def test_entry_source_and_data_mismatch_rejected(self):
        for key in ('entry_sha256', 'bundle_sha256', 'source_sha256'):
            with self.subTest(key=key):
                previous = self.old_binding[key]
                self.old_binding[key] = '0' * 64
                self.refresh()
                with self.assertRaises(ValueError):
                    train.preflight(self.args)
                self.old_binding[key] = previous
                self.refresh()

    def test_overlap_and_partial_warm_start_arguments_rejected(self):
        self.args.run_dir = self.args.warm_start_h1_run
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            train.preflight(self.args)
        self.args.run_dir = self.root / 'continued-run'
        self.args.warm_start_h1_source_tree = None
        with self.assertRaisesRegex(ValueError, 'both'):
            train.preflight(self.args)

    def test_finite_positive_rates_and_consistent_stage_budgets(self):
        for name in ('h1_learning_rate', 'p1_learning_rate'):
            for value in (0., -1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value):
                    args = copy.copy(self.args)
                    setattr(args, name, value)
                    with self.assertRaises(ValueError):
                        train.preflight(args)
        self.args.p1_epochs = 60
        with self.assertRaisesRegex(ValueError, 'same profile'):
            train.preflight(self.args)

    def test_two_stages_forward_distinct_learning_rates_and_no_warm_start_to_p1(self):
        self.args.h1_learning_rate, self.args.p1_learning_rate = .0001, .0002
        self.args.execute = True
        fake_torch, fake_engine, fake_artifacts = (ModuleType(name) for name in
            ('torch', 'fwt_v2.engine', 'fwt_v2.artifacts'))
        fake_torch.set_num_threads = lambda count: None
        fake_artifacts.WORKSPACE = self.root
        fake_artifacts.atomic_json = write_json
        calls = []

        def run(arguments, configuration, family):
            calls.append((family, arguments))
            output = Path(arguments.output)
            output.mkdir(parents=True)
            for name in ('best.pth', 'last.pth'):
                (output / name).write_bytes(b'mocked new checkpoint')
            write_json(output / 'diagnostics.json', dict(status='completed', completed_epochs=2))
            write_json(output / 'selection.json', dict(selection_frozen=True))
            for name in ('history.json', 'configuration.json', 'validation_selected.json'):
                write_json(output / name, {})
            return dict(family=family)

        fake_engine.train = run
        with patch.dict(sys.modules, {'torch': fake_torch, 'fwt_v2.engine': fake_engine,
                                     'fwt_v2.artifacts': fake_artifacts}), \
                patch.object(fwt_v2, 'engine', fake_engine, create=True), \
                patch.object(fwt_v2, 'artifacts', fake_artifacts, create=True):
            det = train.execute(self.args, train.preflight(self.args))
            self.args.phase = 'diff'
            diff = train.execute(self.args, train.preflight(self.args))
        self.assertEqual(det['binding'], diff['binding'])
        self.assertEqual([call[1].learning_rate for call in calls], [.0001, .0002])
        self.assertEqual(calls[0][1].warm_start_h1, det['binding']['warm_start_h1'])
        self.assertIsNone(calls[1][1].warm_start_h1)
        self.assertEqual(Path(calls[1][1].backbone), self.args.run_dir / 'smoke-2ep/det/best.pth')
        self.assertIsNone(calls[1][1].resume)

    def test_own_det_budget_normalization_only_allows_declared_asymmetry(self):
        binding = dict(self.old_binding, epochs=60, stage_epochs=dict(det=30, diff=60))
        receipt = dict(self.receipt, binding=dict(binding, epochs=30))
        write_json(self.old / 'portable_completed.json', receipt)
        write_json(self.old / 'diagnostics.json', dict(status='completed', completed_epochs=30))
        receipt['files']['diagnostics.json'] = file_sha256(self.old / 'diagnostics.json')
        write_json(self.old / 'portable_completed.json', receipt)
        self.assertEqual(train._check_own_det(self.old, binding), self.old / 'best.pth')
        binding['stage_epochs'] = dict(det=60, diff=60)
        with self.assertRaisesRegex(ValueError, 'identical data/source/budget'):
            train._check_own_det(self.old, binding)


if __name__ == '__main__':
    unittest.main()
