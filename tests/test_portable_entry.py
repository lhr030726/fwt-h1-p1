"""Synthetic control/relocation tests: no real arrays, weights, model or GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from fwt_v2.portable import PortableSources, contained_path, file_sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return dict(path=path.name, sha256=file_sha256(path))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.data = root / 'data'
        self.data.mkdir()
        self.rows = []
        for i, role in enumerate(train.DEVELOPMENT_ROLES):
            row = dict(sample_id=f'2015_UID_{i}/2015-06-01', event_id=f'2015_UID_{i}',
                date='2015-06-01', target_date='2015-06-02', role=role,
                input_path=f'arrays/inputs/{i}.npy', label_path=f'arrays/labels/{i}.npy',
                georeference=dict(crs='EPSG:32606', shape=[2, 2], affine_transform=[0, 50, 0, 0, 0, -50]))
            self.rows.append(row)
            for key in ('input_path', 'label_path'):
                path = self.data / row[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic placeholder; NEVER deserialize')
        self.manifest = dict(schema_version='fwt_rcda_manifest_v2', data_root='.',
            samples=self.rows, audit=dict(label_semantics='next_day_cumulative_burn',
            dates_verified=True, roles_frozen=True, exposure_review='synthetic test'))
        original_ref = write_json(self.data / 'original_full.json', self.manifest)
        weather_original = self.data / 'evidence' / 'weather_original.json'
        original = write_json(weather_original, dict(schema_version='fwt_power_hourly_manifest_v1'))
        original.update(path='evidence/weather_original.json', original_path='E:/old/full/hourly.json')
        self.wrapper = dict(schema_version='fwt_power_portable_manifest_v1',
            original_manifest=original, files=[original])
        weather_ref = write_json(self.data / 'weather_manifest.json', self.wrapper)
        data = dict(wind_convention='geographic_towards_ccw_from_east_covariates',
            wind_reference='geographic_east_north', wind_metadata_review='synthetic source',
            pixel_size_m=50., distance_scale_m=1600.,
            input_schema_version='fwt_rcda_inputs_v5_power_wind_covariates22',
            external_wind_manifest='weather_manifest.json',
            external_wind_manifest_sha256=weather_ref['sha256'], external_wind_variant='hourly')
        self.config = dict(data=data, model=train.MODEL, diffusion=train.DIFFUSION)
        geo = [[self.rows[0]['sample_id'], self.rows[0]['georeference']]]
        self.normalizer = dict(role='backbone_train', fit_sample_ids=[self.rows[0]['sample_id']],
            mean=[0.] * 22, std=[1.] * 22, data_config=train._expanded_data(data),
            georeference_sha256=hashlib.sha256(json.dumps(geo, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
        original_normal = copy.deepcopy(self.normalizer)
        original_normal['data_config']['external_wind_manifest'] = 'E:/old/hourly.json'
        original_normal['data_config']['external_wind_manifest_sha256'] = 'a' * 64
        self.bundle = dict(schema_version='fwt_portable_data_bundle_v1',
            data_scope='development_only', heldout_arrays_included=False, purpose='smoke',
            role_counts={role: 1 for role in train.DEVELOPMENT_ROLES},
            original_full_manifest=original_ref,
            original_normalizer=write_json(self.data / 'original_normalizer.json', original_normal),
            files=dict(weather=weather_ref))
        self.array_index = dict(schema_version='fwt_portable_array_index_v1', files=[
            dict(sample_id=row['sample_id'], field=field, role=row['role'],
                 path='data/' + row[field], size=(self.data / row[field]).stat().st_size,
                 sha256=file_sha256(self.data / row[field]))
            for row in self.rows for field in ('input_path', 'label_path')])
        self.bundle['files']['arrays_index'] = write_json(self.data / 'arrays_index.json', self.array_index)
        self.refresh()

    def refresh(self):
        for key, name, value in [('manifest', 'manifest.json', self.manifest),
                ('normalizer', 'normalizer.json', self.normalizer), ('config', 'config.h1.json', self.config)]:
            self.bundle['files'][key] = write_json(self.data / name, value)
        write_json(self.data / 'bundle.json', self.bundle)

    def args(self, phase='det'):
        return train.parser().parse_args(['--phase', phase, '--epochs', '2',
            '--data-root', str(self.data), '--run-dir', str(self.root / 'runs')])


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_preflight_zero_writes_imports_or_array_reads(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        code = ('import sys; import train; '
                f'train.main({["--phase", "det", "--epochs", "2", "--data-root", str(self.f.data), "--run-dir", str(self.root / "runs")]!r}); '
                'assert "torch" not in sys.modules; assert "numpy" not in sys.modules')
        process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result['arrays_read'], 0)
        self.assertFalse(result['training_started'])
        self.assertFalse((self.root / 'runs').exists())
        after = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_formal_rejects_small_bundle(self):
        args = self.f.args()
        args.epochs = 30
        with self.assertRaisesRegex(ValueError, '3197/811/568'):
            train.preflight(args)

    def test_smoke_output_is_separate(self):
        checked = train.preflight(self.f.args())
        self.assertTrue(checked['output'].replace('\\', '/').endswith('/smoke-2ep/det'))

    def test_heldout_role_rejected(self):
        self.f.rows[0]['role'] = 'final'
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'held-out role'):
            train.preflight(self.f.args())

    def test_original_identity_change_rejected(self):
        self.f.rows[0]['georeference']['shape'] = [3, 3]
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'original full manifest'):
            train.preflight(self.f.args())

    def test_changed_statistics_rejected_even_when_bundle_hash_updated(self):
        self.f.normalizer['mean'][0] = 1
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'statistics changed'):
            train.preflight(self.f.args())

    def test_recipe_change_rejected(self):
        self.f.config['model'] = dict(train.MODEL, normalization='batch')
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'frozen H1/P1'):
            train.preflight(self.f.args())

    def test_swapped_input_label_pairing_rejected_by_published_index(self):
        first, second = self.f.rows[:2]
        first['label_path'], second['label_path'] = second['label_path'], first['label_path']
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'runtime sample pairing'):
            train.preflight(self.f.args())

    def test_array_size_change_rejected_without_reading_bytes(self):
        (self.f.data / self.f.rows[0]['input_path']).write_bytes(b'different size')
        with self.assertRaisesRegex(ValueError, 'size differs'):
            train.preflight(self.f.args())

    def test_missing_array_index_record_rejected(self):
        self.f.array_index['files'].pop()
        self.f.bundle['files']['arrays_index'] = write_json(self.f.data / 'arrays_index.json', self.f.array_index)
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            train.preflight(self.f.args())

    def test_default_diff_waits_for_own_det(self):
        with self.assertRaisesRegex(ValueError, 'SAME run-dir/profile'):
            train.preflight(self.f.args('diff'))

    def test_data_output_overlap_rejected(self):
        args = self.f.args()
        args.run_dir = self.f.data / 'runs'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            train.preflight(args)

    def test_existing_output_rejected(self):
        output = Path(train.preflight(self.f.args())['output'])
        output.mkdir(parents=True)
        (output / 'anything').write_text('no overwrite')
        with self.assertRaises(FileExistsError):
            train.preflight(self.f.args())

    def test_unsafe_relative_paths(self):
        for path in ('../escape.npy', 'E:/other/a.npy', '/tmp/a.npy', 'x\\a.npy'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                contained_path(self.f.data, path)

    def test_mapping_windows_paths_case_slash_and_no_fallback(self):
        mapping = PortableSources(self.f.data / 'weather_manifest.json')
        self.assertEqual(mapping.resolve('e:\\OLD\\full\\hourly.json', self.f.data),
                         self.f.data / 'evidence/weather_original.json')
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            mapping.resolve(str(self.f.data / 'manifest.json'), self.f.data)

    def test_mapping_relative_original_base(self):
        mapping = PortableSources(self.f.data / 'weather_manifest.json')
        self.assertEqual(mapping.resolve('hourly.json', self.f.data),
                         self.f.data / 'evidence/weather_original.json')
        self.assertEqual(mapping.logical_parent('E:/old/full/hourly.json', self.f.data),
                         PureWindowsPath('E:/old/full'))

    def test_mapping_changed_evidence_rejected(self):
        (self.f.data / 'evidence/weather_original.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            PortableSources(self.f.data / 'weather_manifest.json')

    def test_mapping_missing_original_or_duplicate_rejected(self):
        for files in ([], self.f.wrapper['files'] * 2):
            document = dict(self.f.wrapper, files=files)
            with self.assertRaises(ValueError):
                PortableSources(self.f.data / 'weather_manifest.json', document)

    def test_execute_needs_explicit_flag(self):
        args = self.f.args()
        with self.assertRaisesRegex(ValueError, 'Explicit execution'):
            train.execute(args, train.preflight(args))

    def test_mock_engine_single_phase_cwd_fixed_recipe_and_restore(self):
        args = self.f.args()
        args.execute = True
        checked = train.preflight(args)
        calls = []
        fake_torch = ModuleType('torch')
        fake_torch.set_num_threads = lambda count: calls.append(('threads', count))
        fake_engine = ModuleType('fwt_v2.engine')
        fake_artifacts = ModuleType('fwt_v2.artifacts')
        fake_artifacts.WORKSPACE = self.root
        fake_artifacts.atomic_json = write_json

        def run(arguments, configuration, family):
            calls.append(('train', family, arguments, Path.cwd()))
            output = Path(arguments.output)
            output.mkdir(parents=True)
            for name in ('best.pth', 'last.pth'):
                (output / name).write_bytes(b'synthetic checkpoint placeholder')
            write_json(output / 'diagnostics.json', dict(status='completed', completed_epochs=2))
            write_json(output / 'selection.json', dict(selection_frozen=True))
            for name in ('history.json', 'configuration.json', 'validation_selected.json'):
                write_json(output / name, {})
            return dict(family=family)

        fake_engine.train = run
        old_cwd = Path.cwd()
        with patch.dict(sys.modules, {'torch': fake_torch, 'fwt_v2.engine': fake_engine,
                                     'fwt_v2.artifacts': fake_artifacts}):
            receipt = train.execute(args, checked)
        self.assertEqual(Path.cwd(), old_cwd)
        self.assertEqual(fake_artifacts.WORKSPACE, self.root)
        self.assertEqual(len([c for c in calls if c[0] == 'train']), 1)
        _, family, forwarded, cwd = calls[-1]
        self.assertEqual((family, cwd), ('det', self.f.data))
        self.assertEqual((forwarded.epochs, forwarded.batch_size, forwarded.seed,
            forwarded.samples, forwarded.sampling_steps, forwarded.loss), (2, 1, 0, 8, 4, 'focal'))
        self.assertIsNone(forwarded.resume)
        self.assertTrue(receipt['smoke_only'])
        diff = train.preflight(self.f.args('diff'))
        self.assertEqual(Path(diff['backbone']), Path(checked['output']) / 'best.pth')
        changed = copy.deepcopy(diff['binding'])
        changed['epochs'] = 30
        with self.assertRaisesRegex(ValueError, 'identical data/source/budget'):
            train._check_own_det(Path(checked['output']), changed)
        (Path(checked['output']) / 'best.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            train._check_own_det(Path(checked['output']), diff['binding'])


if __name__ == '__main__':
    unittest.main()
