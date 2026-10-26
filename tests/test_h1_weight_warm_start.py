"""Actual CPU tensor loading/optimization; synthetic data, no quality claim."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fwt_v2 import artifacts, engine
from fwt_v2.data import DataConfig
from fwt_v2.models import FWTConfig
from fwt_v2.manifest import manifest_digest


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Conv2d(18, 1, 1)

    def forward(self, x, physics):
        return dict(logits_det=self.logit(x))


class TinyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return dict(sample_id=sample['sample_id'], event_id=sample['event_id'],
            x_norm=torch.arange(72, dtype=torch.float32).reshape(18, 2, 2) / 72,
            target=torch.tensor([[[1., 1.], [0., 1.]]]), valid_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            domain_mask=torch.ones(1, 2, 2, dtype=torch.bool), physics={})


class WeightWarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_config = DataConfig(wind_convention='grid_towards_ccw_from_east',
            wind_reference='projected_grid', wind_metadata_review='synthetic fixture')
        self.model_config = FWTConfig()
        self.manifest = dict(samples=[])
        for i, role in enumerate(('backbone_train', 'residual_train', 'validation')):
            array = self.root / f'{i}.bin'
            array.write_bytes(b'synthetic metadata file, not an array')
            self.manifest['samples'].append(dict(role=role, sample_id=str(i), event_id=str(i),
                input_path=str(array), label_path=str(array)))
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.normalizer_path = self.root / 'normalizer.json'
        self.normalizer_path.write_text('{}', encoding='utf-8')
        self.original = TinyBackbone()
        with torch.no_grad():
            self.original.logit.weight.fill_(.25)
            self.original.logit.bias.fill_(.5)
        self.checkpoint = self.root / 'original.pth'
        self.bundle = dict(schema='fwt_checkpoint_v2', family='det', architecture='fwt', selection_frozen=True,
            source_sha256=engine.source_digest(), model_config=asdict(self.model_config),
            data_config=asdict(self.data_config), manifest_sha256=manifest_digest(self.manifest),
            normalizer_sha256=artifacts.sha256(self.normalizer_path), requested_epochs=30, completed_epochs=30,
            epoch=28, validation_ap=.229, training_sample_ids=['0'], validation_sample_ids=['2'],
            state_dict=copy.deepcopy(self.original.state_dict()))
        torch.save(self.bundle, self.checkpoint)
        self.compat = dict(original_source_tree='synthetic-preserved-source',
            original_source_sha256=engine.source_digest(), current_source_sha256=engine.source_digest())
        self.lineage = dict(checkpoint=str(self.checkpoint), checkpoint_sha256=artifacts.sha256(self.checkpoint),
            original_completed_epochs=30, selected_epoch=28, validation_ap=.229, source_compatibility=self.compat)
        self.compat_patch = patch('fwt_v2.backbone_compat.verify_backbone_sources', return_value=self.compat)
        self.compat_patch.start()

    def tearDown(self):
        self.compat_patch.stop()
        self.temp.cleanup()

    def load(self, model):
        return engine.load_warm_start_h1(model, self.lineage, self.model_config,
            self.data_config, self.manifest, artifacts.sha256(self.normalizer_path))

    def test_exact_weights_and_rng_preserved_without_optimizer_restore(self):
        target = TinyBackbone()
        before_rng = torch.get_rng_state().clone()
        report = self.load(target)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(engine.model_state_digest(target), engine.model_state_digest(self.original))
        self.assertEqual(report['initial_state_sha256'], engine.model_state_digest(self.original))
        self.assertFalse(report['optimizer_state_reused'])
        self.assertFalse(report['rng_state_reused'])
        self.assertTrue(all(parameter.requires_grad for parameter in target.parameters()))

    def test_rejects_invalid_tensor_and_original_checkpoint_mutation(self):
        original_bytes = self.checkpoint.read_bytes()
        self.checkpoint.write_bytes(original_bytes + b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.load(TinyBackbone())
        self.bundle['state_dict']['logit.weight'][0, 0, 0, 0] = float('nan')
        torch.save(self.bundle, self.checkpoint)
        self.lineage['checkpoint_sha256'] = artifacts.sha256(self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'Invalid warm-start tensor'):
            self.load(TinyBackbone())

    def test_actual_additional_segment_uses_fresh_adamw_and_preserves_original(self):
        original_bytes = self.checkpoint.read_bytes()
        args = SimpleNamespace(epochs=2, batch_size=1, learning_rate=.0001, seed=0, device='cpu',
            output=str(self.root / 'new-segment'), manifest=str(self.manifest_path),
            normalizer=str(self.normalizer_path), samples=8, sampling_steps=4, sampler='ddim',
            architecture='fwt', loss='focal', warm_start_h1=self.lineage, resume=None,
            sample_cache_gib=0, validation_batch_size=1)

        def loader(manifest, role, normalizer, config, batch_size, shuffle=False, sample_cache=None):
            return DataLoader(TinyDataset([s for s in manifest['samples'] if s['role'] == role]), batch_size=1)

        with patch.object(artifacts, 'WORKSPACE', self.root), \
                patch.object(engine, 'audited_manifest', return_value=self.manifest), \
                patch.object(engine.NormalizationStats, 'load', return_value=object()), \
                patch.object(engine, 'make_loader', side_effect=loader), \
                patch.object(engine, 'build_det', side_effect=lambda *args: TinyBackbone()):
            engine.train(args, dict(data=asdict(self.data_config), model=asdict(self.model_config)), 'det')
        configuration = json.loads((Path(args.output) / 'configuration.json').read_text())
        self.assertEqual(configuration['initialization'], 'warm_start_selected_h1_new_optimizer')
        self.assertTrue(configuration['warm_start']['optimizer_initial_state_empty'])
        self.assertEqual(configuration['warm_start']['initial_state_sha256'], engine.model_state_digest(self.original))
        last = torch.load(Path(args.output) / 'last.pth', weights_only=True)
        self.assertEqual(last['updates'], 2)
        self.assertEqual(last['completed_epochs'], 2)
        self.assertTrue(all(int(state['step']) == 2 for state in last['optimizer_state_dict']['state'].values()))
        self.assertTrue(any(not torch.equal(value, self.original.state_dict()[name])
                            for name, value in last['model_state_dict'].items()))
        self.assertEqual(original_bytes, self.checkpoint.read_bytes())

    def test_h1_weight_warm_start_cannot_be_passed_to_p1_or_resume(self):
        args = SimpleNamespace(epochs=2, batch_size=1, learning_rate=.0001,
            warm_start_h1=self.lineage, resume='old-optimizer.pth')
        for family in ('det', 'diff'):
            with self.assertRaisesRegex(ValueError, 'det without optimizer resume'):
                engine.train(args, {}, family)


if __name__ == '__main__':
    unittest.main()
