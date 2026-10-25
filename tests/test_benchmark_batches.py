"""CPU-only checks for probe safety gates and throughput arithmetic."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('benchmark_batches', ROOT / 'scripts/benchmark_batches.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class BatchProbeTests(unittest.TestCase):
    def test_refuses_active_gpu_training(self):
        result = SimpleNamespace(returncode=0, stdout='11392, python3.10\n', stderr='')
        self.assertEqual(probe.active_gpu_processes(lambda *a, **k: result),
                         [{'pid': 11392, 'name': 'python3.10'}])
        with patch.object(probe, 'active_gpu_processes', return_value=[{'pid': 11392}]):
            with self.assertRaisesRegex(RuntimeError, 'active'):
                probe.require_idle_gpu()

    def test_unknown_gpu_accounting_fails_closed(self):
        for code, stdout in [(1, ''), (0, '[Not Supported]\n')]:
            result = SimpleNamespace(returncode=code, stdout=stdout, stderr='unavailable')
            with self.assertRaises(RuntimeError):
                probe.active_gpu_processes(lambda *a, **k: result)

    def test_throughput_counts_samples_not_batches(self):
        result = probe.summarize_timings([.1, .3], 8)
        self.assertEqual(result['samples_per_second'], 40.)
        self.assertEqual(result['mean_seconds'], .2)
        for times in [[], [0], [-1], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                probe.summarize_timings(times, 1)

    def test_plan_does_not_touch_gpu_or_start_child(self):
        with patch.object(probe, 'require_idle_gpu', side_effect=AssertionError('GPU touched')), \
                patch.object(probe.subprocess, 'run', side_effect=AssertionError('child started')), \
                patch('builtins.print') as printed:
            self.assertEqual(probe.main(['--phases', 'det', '--batches', '1,4']), 0)
        plan = json.loads(printed.call_args.args[0])
        self.assertEqual(plan['status'], 'plan_only')
        self.assertEqual(plan['batches'], [1, 4])
        self.assertIn('not training evidence', plan['purpose'])

    def test_child_retains_shape_recipe_and_limits(self):
        args = probe.parse_args(['--height', '128', '--width', '256', '--cpu-threads', '4',
                                 '--max-memory-fraction', '.8', '--iterations', '7'])
        command = probe.child_command(args, 'diff', 8)
        parsed = probe.parse_args(command[2:])
        self.assertTrue(parsed.execute)
        self.assertEqual((parsed._trial, parsed._batch), ('diff', 8))
        self.assertEqual((parsed.height, parsed.width, parsed.cpu_threads), (128, 256, 4))
        self.assertEqual((parsed.max_memory_fraction, parsed.iterations), (.8, 7))


if __name__ == '__main__':
    unittest.main()
