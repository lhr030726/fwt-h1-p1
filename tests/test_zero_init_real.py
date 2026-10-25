"""Real-witness invocation gates only; never initializes GPU or reads datasets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_zero_init_real', ROOT / 'scripts/check_zero_init_real.py')
witness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(witness)


class WitnessGateTests(unittest.TestCase):
    def test_default_plan_without_data_gpu_or_torch(self):
        code = ('import sys; sys.path.insert(0, "scripts"); import check_zero_init_real; '
                'check_zero_init_real.main(["--data-root", "absent-development-data"]); '
                'assert "torch" not in sys.modules; assert "numpy" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['status'], 'plan_only')
        self.assertEqual((plan['samples'], plan['batch_size'], plan['updates_requested']), (2, 1, 2))
        self.assertEqual(plan['checkpoint_reads'], 0)
        self.assertEqual(plan['validation_arrays_read'], 0)

    def test_plan_ignores_existing_or_missing_output_without_inspecting_it(self):
        with patch.object(witness, 'require_no_jobs', side_effect=AssertionError('GPU queried')), \
                patch.object(witness, 'execute', side_effect=AssertionError('executed')), \
                patch.object(witness.Path, 'read_bytes', side_effect=AssertionError('file read')), \
                patch.object(witness.Path, 'read_text', side_effect=AssertionError('file read')), \
                patch('builtins.print'):
            self.assertEqual(witness.main(['--output', 'absent.json']), 0)

    def test_execution_requires_linux_before_any_output_write(self):
        with patch.object(witness.sys, 'platform', 'win32'), \
                patch.object(witness.Path, 'open', side_effect=AssertionError('file opened')):
            with self.assertRaisesRegex(RuntimeError, 'Linux GPU environment'):
                witness.main(['--execute'])

    def test_execution_requires_idle_host_before_any_output_write(self):
        with patch.object(witness.sys, 'platform', 'linux'), \
                patch.object(witness, 'require_no_jobs', side_effect=RuntimeError('busy')), \
                patch.object(witness.Path, 'open', side_effect=AssertionError('file opened')):
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                witness.main(['--execute'])


if __name__ == '__main__':
    unittest.main()
