"""No GPU calls: fail-closed fixtures for the future diagnostic benchmark."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).parents[1] / 'scripts/benchmark_gradient_diagnostics.py'
spec = importlib.util.spec_from_file_location('gradient_benchmark', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IdleGateTests(unittest.TestCase):
    def test_cpu_preflight_training_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process = root / '123'
            process.mkdir()
            for script in ['train.py', 'check_zero_init_real.py']:
                (process / 'cmdline').write_bytes(b'python\0/repo/' + script.encode() + b'\0--execute\0')
                with self.subTest(script=script), patch.object(module.sys, 'platform', 'linux'), \
                        patch.object(module.subprocess, 'check_output', return_value=''):
                    with self.assertRaisesRegex(RuntimeError, '123'):
                        module.idle_gate(root)

    def test_other_compute_process_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module.sys, 'platform', 'linux'), \
                    patch.object(module.subprocess, 'check_output', return_value='456\n'):
                with self.assertRaisesRegex(RuntimeError, '456'):
                    module.idle_gate(Path(temp))

    def test_own_compute_pid_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module.sys, 'platform', 'linux'), \
                    patch.object(module.os, 'getpid', return_value=123), \
                    patch.object(module.subprocess, 'check_output', return_value='123\n'):
                self.assertEqual(module.idle_gate(Path(temp))['other_training_or_compute_pids'], [])

    def test_failed_query_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module.sys, 'platform', 'linux'), \
                    patch.object(module.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, 'query')):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.idle_gate(Path(temp))

    def test_malformed_query_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module.sys, 'platform', 'linux'), \
                    patch.object(module.subprocess, 'check_output', return_value='N/A\n'):
                with self.assertRaisesRegex(RuntimeError, 'Unexpected nvidia-smi'):
                    module.idle_gate(Path(temp))


if __name__ == '__main__':
    unittest.main()
