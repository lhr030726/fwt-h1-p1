"""Mock-only setup/runtime tests. Never run pip, shell setup or real CUDA."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fwt_environment_check', ROOT / 'scripts/check_environment.py')
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


class FakeTensor:
    def __init__(self, total, shape=None):
        self.total, self.shape = total, shape

    def __matmul__(self, other):
        return FakeTensor(self.shape[0] * self.shape[1] * other.shape[1])

    def sum(self):
        return self

    def item(self):
        return self.total


def fake_torch(*, available=True, capability=(12, 0), version='2.7.1+cu128', cuda='12.8', conv=324.):
    calls = []

    def ones(shape, **kwargs):
        calls.append(('ones', shape, kwargs))
        return FakeTensor(1., shape)

    fake = SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: 1,
            get_device_capability=lambda device: capability,
            get_device_name=lambda device: 'NVIDIA RTX PRO 6000 Blackwell',
            get_arch_list=lambda: ['sm_120'], synchronize=lambda device: calls.append(('synchronize', device))),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90701)),
        device=lambda value: value, float32='float32', inference_mode=nullcontext, ones=ones,
        nn=SimpleNamespace(functional=SimpleNamespace(conv2d=lambda image, kernel: FakeTensor(conv))))
    return fake, calls


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name).resolve()
        self.venv = self.root / '.venv-fwt'
        self.venv.mkdir()
        (self.venv / environment.MARKER).write_text(environment.MARKER_PREFIX + self.root.as_posix() + '\n', encoding='utf-8')
        for name in ('requirements.txt', 'requirements-torch.txt'):
            (self.root / name).write_bytes((ROOT / name).read_bytes())

    def tearDown(self):
        self.temp.cleanup()

    def collect(self, **kwargs):
        with patch.object(environment.sys, 'prefix', str(self.venv)), \
                patch.object(environment.sys, 'base_prefix', str(self.root / 'system')), \
                patch.object(environment.sys, 'version_info', (3, 10, 12)):
            return environment.collect(self.root,
                version=lambda name: environment.EXPECTED[name] + ('+cu128' if name == 'torch' else ''),
                distributions=lambda: [], **kwargs)

    def test_metadata_default_never_loads_torch_or_queries_gpu(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout='No broken requirements found.', stderr='')

        result = self.collect(run=run, torch_loader=lambda: self.fail('Unexpected torch import'))
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['gpu_check_completed'])
        self.assertFalse(result['training_executed'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], ['-I', '-m', 'pip', '--isolated', 'check'])

    def test_unknown_environment_marker_rejected(self):
        (self.venv / environment.MARKER).write_text('unowned', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'dedicated environment'):
            environment.check_managed_environment(self.root, str(self.venv), 'other', (3, 10))

    def test_other_interpreter_and_python_version_rejected(self):
        for prefix, base, version in [('outside', 'base', (3, 10)),
                (str(self.venv), str(self.venv), (3, 10)), (str(self.venv), 'base', (3, 11))]:
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                environment.check_managed_environment(self.root, prefix, base, version)

    def test_wrong_dependency_version_rejected(self):
        for package in environment.EXPECTED:
            with self.subTest(package=package), self.assertRaisesRegex(RuntimeError, 'versions differ'):
                environment.check_versions(lambda name: '0.0' if name == package else environment.EXPECTED[name])

    def test_science_custom_build_not_silently_accepted(self):
        with self.assertRaises(RuntimeError):
            environment.check_versions(lambda name: environment.EXPECTED[name] + ('+custom' if name == 'numpy' else ''))

    def test_failed_pip_check_stops_before_gpu(self):
        result = self.collect(gpu_test=True,
            run=lambda *a, **k: SimpleNamespace(returncode=1, stdout='Broken dependency.', stderr=''),
            torch_loader=lambda: self.fail('Must not reach GPU'))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('pip check', result['error']['message'])

    def test_driver_failure_is_recorded_without_torch_import(self):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=int(command[0] == 'nvidia-smi'), stdout='', stderr='unavailable')
        result = self.collect(gpu_test=True, run=run, torch_loader=lambda: self.fail('Must not reach torch'))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('platform', result['error']['message'])

    def test_explicit_gpu_probe_uses_only_three_small_allocations(self):
        torch, calls = fake_torch()
        result = environment.gpu_probe(torch, 0)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual([call[1] for call in calls if call[0] == 'ones'], [(16, 16), (1, 1, 8, 8), (1, 1, 3, 3)])
        self.assertEqual((result['matmul_sum'], result['convolution_sum']), (4096., 324.))
        self.assertEqual((result['backward_calls'], result['optimizer_updates']), (0, 0))

    def test_cpu_or_old_cuda_wheel_rejected(self):
        for options in (dict(version='2.0.0'), dict(cuda=None), dict(cuda='11.8')):
            with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, 'cu128'):
                environment.gpu_probe(fake_torch(**options)[0], 0)

    def test_wrong_device_unavailable_or_wrong_capability_rejected(self):
        for options, index in [(dict(available=False), 0), ({}, 2), (dict(capability=(8, 9)), 0)]:
            with self.subTest(options=options, index=index), self.assertRaises(RuntimeError):
                environment.gpu_probe(fake_torch(**options)[0], index)

    def test_nonfinite_or_wrong_arithmetic_rejected(self):
        for result in (float('nan'), float('inf'), 0.):
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, 'arithmetic'):
                environment.gpu_probe(fake_torch(conv=result)[0], 0)

    def test_mock_full_probe_records_runtime_not_training_equivalence(self):
        torch, _ = fake_torch()
        result = self.collect(gpu_test=True, torch_loader=lambda: torch,
            run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='mock', stderr=''))
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['gpu_check_completed'])
        self.assertEqual(result['gpu']['capability'], [12, 0])
        self.assertIn('equivalence is untested', result['migration'])
        self.assertEqual(result['array_or_checkpoint_reads'], 0)

    def test_snapshot_is_unique_and_under_runs_environment(self):
        value = dict(status='failed', error={'message': 'synthetic failure'})
        first = environment.write_snapshot(self.root, value)
        second = environment.write_snapshot(self.root, value)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.root / 'runs' / 'environment')
        self.assertEqual(json.loads(first.read_text(encoding='utf-8')), value)

    def test_setup_contract_has_no_global_install_driver_or_training(self):
        source = (ROOT / 'scripts/setup_ubuntu22.sh').read_text(encoding='utf-8')
        commands = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('#'))
        self.assertEqual(commands.count('"$python" -I -m pip --isolated --require-virtualenv install --no-cache-dir'), 2)
        self.assertIn('https://download.pytorch.org/whl/cu128', commands)
        self.assertIn('system_python=/usr/bin/python3.10', commands)
        self.assertIn('"$system_python" -I -m venv "$environment"', commands)
        self.assertIn('"$(cat -- "$marker")" != "$marker_text"', commands)
        self.assertNotIn('train.py', commands)
        self.assertNotIn('download_data.py', commands)
        self.assertNotIn('apt-get', commands)
        self.assertNotIn('--clear', commands)
        self.assertNotIn('rm -', commands)


if __name__ == '__main__':
    unittest.main()
