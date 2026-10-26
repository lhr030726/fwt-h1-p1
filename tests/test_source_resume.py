"""CPU full-loop continuation equality plus fail-closed migration evidence."""
import copy
from dataclasses import asdict
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2 import artifacts, engine, source_resume as migration
from fwt_v2.data import DataConfig
from fwt_v2.models import FWTConfig
from test_h1_weight_warm_start import TinyBackbone, TinyDataset


class DropoutBackbone(TinyBackbone):
    def forward(self, x, physics):
        x = torch.nn.functional.dropout(x, .2, self.training)
        return dict(logits_det=self.logit(x))


class ResumeMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.repo = Path(__file__).resolve().parents[1]
        cls.archive = subprocess.check_output(['git', 'archive',
            '5da517f51b8d6b6ca06ff008b7b42a1b170baa64', 'train.py', 'fwt_v2'], cwd=cls.repo)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / 'original-source'
        with tarfile.open(fileobj=io.BytesIO(self.archive)) as archive:
            for member in archive.getmembers():
                target = (self.old / member.name).resolve()
                self.assertIn(self.old.resolve(), target.parents)
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        self.old_inventory = migration.source_inventory(self.old)
        self.old_digest = migration.package_digest(self.old_inventory)
        spec = importlib.util.spec_from_file_location('fwt_v2._source_resume_test_original_engine',
                                                     self.old / 'fwt_v2/engine.py')
        self.original_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.original_engine)
        self.config = dict(data=asdict(DataConfig(wind_convention='grid_towards_ccw_from_east',
            wind_reference='projected_grid', wind_metadata_review='synthetic CPU test')),
            model=asdict(FWTConfig()))
        self.manifest = dict(samples=[])
        for i, role in enumerate(('backbone_train', 'backbone_train', 'backbone_train',
                                  'residual_train', 'validation')):
            path = self.root / ('input-' + str(i))
            path.write_bytes(b'synthetic metadata fixture')
            self.manifest['samples'].append(dict(role=role, sample_id=str(i), event_id=str(i),
                input_path=str(path), label_path=str(path)))
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf8')
        self.normalizer = self.root / 'normalizer.json'
        self.normalizer.write_text('{}', encoding='utf8')

    def tearDown(self):
        self.temp.cleanup()

    def run_segment(self, name, *, old=False, resume=None, continuation=None, stop=None):
        args = SimpleNamespace(epochs=3, batch_size=1, learning_rate=.0001, seed=0,
            device='cpu', output=str(self.root / name), manifest=str(self.manifest_path),
            normalizer=str(self.normalizer), samples=8, sampling_steps=4, sampler='ddim',
            architecture='fwt', loss='focal', warm_start_h1=None, resume=resume,
            stop_after_epoch=stop, continuation_manifest=continuation,
            sample_cache_gib=0, validation_batch_size=1)
        def loader(manifest, role, normalizer, config, batch_size, shuffle=False, sample_cache=None):
            return DataLoader(TinyDataset([s for s in manifest['samples'] if s['role'] == role]),
                              batch_size=batch_size, shuffle=shuffle)
        runner = self.original_engine if old else engine
        true_metrics = runner.probability_metrics
        aps = iter([.8, .7] if resume else [.9, .8, .7])
        def metrics(records):
            value = true_metrics(records)
            value['pooled_ap'] = next(aps)
            return value
        with patch.object(artifacts, 'WORKSPACE', self.root), \
                patch.object(runner, 'audited_manifest', return_value=self.manifest), \
                patch.object(runner.NormalizationStats, 'load', return_value=object()), \
                patch.object(runner, 'make_loader', side_effect=loader), \
                patch.object(runner, 'build_det', side_effect=lambda *a: DropoutBackbone()), \
                patch.object(runner, 'probability_metrics', side_effect=metrics):
            runner.train(args, self.config, 'det')
        return self.root / name

    def manifest_for(self, checkpoint):
        current = migration.source_inventory(self.repo)
        doc = dict(schema=migration.SCHEMA, original_source_tree=str(self.old),
            original_files=self.old_inventory, current_files=current,
            changed_files=sorted(p for p in set(self.old_inventory) | set(current)
                                 if self.old_inventory.get(p) != current.get(p)),
            original_entry_binding=dict(source_sha256=self.old_digest,
                                        entry_sha256=self.old_inventory['train.py']),
            checkpoint=dict(path=str(checkpoint), sha256=artifacts.sha256(checkpoint),
                            phase='det', completed_epochs=1),
            approved_p1_validation_parallel_samples=1)
        path = self.root / 'continuation.json'
        path.write_text(json.dumps(doc), encoding='utf8')
        return path

    def assert_recursive_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_recursive_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_recursive_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_interrupted_resume_matches_uninterrupted_optimizer_rng_weights_and_old_best(self):
        full = self.run_segment('full', old=True)
        stopped = self.run_segment('stopped', old=True, stop=1)
        source = stopped / 'last.pth'
        original_bytes = source.read_bytes()
        manifest = self.manifest_for(source)
        resumed = self.run_segment('resumed', resume=str(source), continuation=str(manifest))
        left = torch.load(full / 'last.pth', weights_only=True)
        right = torch.load(resumed / 'last.pth', weights_only=True)
        for field in ('model_state_dict', 'optimizer_state_dict', 'optimizer_integrity', 'rng_state',
                      'completed_epochs', 'updates', 'best_validation_ap', 'selected_validation'):
            self.assert_recursive_equal(left[field], right[field])
        self.assertEqual(right['completed_epochs'], 3)
        self.assertEqual(right['updates'], 9)
        self.assertEqual([row['epoch'] for row in right['history']], [0, 1, 2])
        self.assertEqual([row['training_loss'] for row in left['history']],
                         [row['training_loss'] for row in right['history']])
        self.assert_recursive_equal(left['selected']['state_dict'], right['selected']['state_dict'])
        self.assertEqual(right['selected']['epoch'], 0)
        report = json.loads((resumed / 'validation_selected.json').read_text())
        self.assertEqual(report['selection_origin']['original_metadata']['source_sha256'], self.old_digest)
        self.assertEqual(report['source_migration']['checkpoint']['sha256'], artifacts.sha256(source))
        self.assertEqual(original_bytes, source.read_bytes())

    def test_source_checkpoint_and_recipe_changes_rejected(self):
        stopped = self.run_segment('stopped', old=True, stop=1)
        source = stopped / 'last.pth'
        manifest = self.manifest_for(source)
        snapshot = migration.read_snapshot(source, manifest, 'det')
        binding = copy.deepcopy(snapshot['original']['binding'])
        migration.inherit_initialization(binding['provenance'], snapshot)
        binding['provenance']['source_sha256'] = snapshot['evidence']['current_source_sha256']
        for key, value in [('batch_size', 2), ('requested_epochs', 4), ('learning_rate', .1)]:
            changed = copy.deepcopy(binding)
            changed['provenance'][key] = value
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                migration.migrate_snapshot(snapshot, changed)
        (self.old / 'fwt_v2/models.py').write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            migration.verify_manifest(manifest)
        source.write_bytes(source.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            migration.read_snapshot(source, manifest, 'det')

    def test_initialization_lineage_and_only_approved_parallelism_are_preserved(self):
        stopped = self.run_segment('stopped', old=True, stop=1)
        manifest = self.manifest_for(stopped / 'last.pth')
        snapshot = migration.read_snapshot(stopped / 'last.pth', manifest, 'det')
        old = snapshot['original']['binding']['provenance']
        old.update(initialization='warm_start_selected_h1_new_optimizer', warm_start=dict(
            original_selected_epoch=28, optimizer_state_reused=False, rng_state_reused=False))
        snapshot['original']['selected'].update(copy.deepcopy(old))
        binding = copy.deepcopy(snapshot['original']['binding'])
        migration.inherit_initialization(binding['provenance'], snapshot)
        binding['provenance']['source_sha256'] = snapshot['evidence']['current_source_sha256']
        snapshot['evidence']['approved_p1_validation_parallel_samples'] = 4
        binding['provenance']['performance']['p1_validation_parallel_samples'] = 4
        migrated = migration.migrate_snapshot(snapshot, binding)
        self.assertEqual(migrated['binding']['provenance']['warm_start'], old['warm_start'])
        binding['provenance']['performance']['p1_validation_parallel_samples'] = 8
        with self.assertRaisesRegex(ValueError, 'parallelism differs'):
            migration.migrate_snapshot(snapshot, binding)

    def test_reviewed_hashes_do_not_allow_changed_scientific_function(self):
        stopped = self.run_segment('stopped', old=True, stop=1)
        manifest = self.manifest_for(stopped / 'last.pth')
        candidate = self.root / 'candidate'
        shutil.copytree(self.repo / 'fwt_v2', candidate / 'fwt_v2')
        shutil.copyfile(self.repo / 'train.py', candidate / 'train.py')
        source = candidate / 'fwt_v2/engine.py'
        code = source.read_text(encoding='utf8')
        code = code.replace('def deterministic_loss(', 'def changed_scientific_loss(', 1)
        source.write_text(code, encoding='utf8')
        document = json.loads(manifest.read_text())
        document['current_files'] = migration.source_inventory(candidate)
        manifest.write_text(json.dumps(document), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Protected scientific code changed'):
            migration.verify_manifest(manifest, candidate)

    def test_entry_contract_only_changes_source_and_preserves_warm_lineage(self):
        old = dict(source_sha256='old', entry_sha256='old-entry', epochs=30,
            stage_epochs=dict(det=30, diff=60), recipe=dict(batch_size=16, learning_rate=.0001),
            performance=dict(validation_batch_size=8), warm_start_h1=dict(selected_epoch=28))
        checked = dict(document=dict(original_entry_binding=old), evidence=dict(
            current_source_sha256='new', current_entry_sha256='new-entry'))
        new = copy.deepcopy(old)
        new.pop('warm_start_h1')
        new.update(source_sha256='new', entry_sha256='new-entry', epochs=60)
        result = migration.bind_entry(new, checked, 'diff')
        self.assertEqual(result['warm_start_h1'], old['warm_start_h1'])
        changed = copy.deepcopy(old)
        changed.update(source_sha256='new', entry_sha256='new-entry')
        changed['recipe']['batch_size'] = 32
        with self.assertRaisesRegex(ValueError, 'scientific entry recipe changed'):
            migration.bind_entry(changed, checked, 'det')

    def test_parallelism_guard_applies_without_a_resume_checkpoint(self):
        checked = dict(evidence=dict(approved_p1_validation_parallel_samples=4))
        args = SimpleNamespace(epochs=60, batch_size=16, learning_rate=.0001,
            resume=None, continuation_manifest='synthetic-manifest')
        with patch.object(migration, 'verify_manifest', return_value=checked), \
                patch.dict(os.environ, {'FWT_P1_VALIDATION_PARALLEL_SAMPLES': '8'}):
            with self.assertRaisesRegex(ValueError, 'parallelism differs'):
                engine.train(args, {}, 'diff')


if __name__ == '__main__':
    unittest.main()
