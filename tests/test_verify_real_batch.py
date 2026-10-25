"""Safety gates and comparison rules; no real data, GPU or training launch."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('verify_real_batch', ROOT / 'scripts/verify_real_batch.py')
witness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(witness)


class RealBatchWitnessTests(unittest.TestCase):
    def test_default_is_plan_without_data_gpu_or_torch(self):
        with patch.object(witness, 'require_no_jobs', side_effect=AssertionError('GPU touched')), \
                patch.object(witness, 'execute', side_effect=AssertionError('execution called')), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('file opened')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('file opened')), \
                patch('builtins.print') as printed:
            self.assertEqual(witness.main(['--data-root', 'absent-development-data']), 0)
        plan = json.loads(printed.call_args.args[0])
        self.assertEqual(plan['status'], 'plan_only')
        self.assertEqual(plan['batch_size'], 1)
        self.assertEqual(plan['gpu_tolerance'], dict(atol=0., rtol=0.))
        self.assertEqual(plan['checkpoint_reads'], 0)

    def test_blocked_pid_refuses_before_gpu_query(self):
        args = witness.parse_args(['--blocked-pid', '11392'])
        for state in ('R', 'S', 'D', 'T', 't', 'I'):
            with patch.object(witness, 'blocked_process_state', return_value=state), \
                    patch.object(witness, 'active_gpu_processes') as gpu:
                with self.assertRaisesRegex(RuntimeError, 'is live'):
                    witness.require_no_jobs(args)
                gpu.assert_not_called()

    def test_zombie_dead_and_missing_blocked_pid_allow_idle_gpu(self):
        args = witness.parse_args(['--blocked-pid', '11392'])
        for state in ('Z', 'X', 'x', 'missing'):
            with patch.object(witness, 'blocked_process_state', return_value=state), \
                    patch.object(witness, 'active_gpu_processes', return_value=[]) as gpu:
                witness.require_no_jobs(args)
                gpu.assert_called_once_with()

    def test_proc_state_handles_zombie_missing_and_unreadable(self):
        with patch.object(Path, 'is_file', return_value=True):
            for state in ('Z', 'T', 'S'):
                with patch.object(Path, 'read_text', return_value=f'11392 (python (worker)) {state} 1 2 3'):
                    self.assertEqual(witness.blocked_process_state(11392), state)
            with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
                self.assertEqual(witness.blocked_process_state(11392), 'missing')
            for error in (PermissionError, OSError):
                with patch.object(Path, 'read_text', side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, 'Cannot read'):
                        witness.blocked_process_state(11392)
            for invalid in ('not a proc stat', '11392 (python) ? 1', '999 (python) Z 1'):
                with patch.object(Path, 'read_text', return_value=invalid):
                    with self.assertRaisesRegex(RuntimeError, 'Cannot parse'):
                        witness.blocked_process_state(11392)
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'procfs is unavailable'):
                witness.blocked_process_state(11392)

    def test_own_gpu_context_is_allowed_but_other_process_always_refused(self):
        args = witness.parse_args([])
        own = dict(pid=24680, name='this witness')
        other = dict(pid=13579, name='other GPU job')
        with patch.object(witness.os, 'getpid', return_value=own['pid']):
            for rows in ([], [own]):
                with patch.object(witness, 'active_gpu_processes', return_value=rows):
                    witness.require_no_jobs(args)
            for rows in ([other], [own, other]):
                with patch.object(witness, 'active_gpu_processes', return_value=rows):
                    with self.assertRaisesRegex(RuntimeError, 'Other GPU compute processes'):
                        witness.require_no_jobs(args)
            # Also exercise the actual nvidia-smi parser used by the witness.
            result = SimpleNamespace(returncode=0, stdout='24680, python\n13579, python\n', stderr='')
            self.assertEqual(witness.active_gpu_processes(lambda *a, **kw: result),
                             [dict(pid=13579, name='python')])

    def test_report_cannot_replace_data_run_or_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing.json'
            existing.write_text('{}')
            for output in [existing, root / 'data/report.json', root / 'runs/job/report.json',
                           ROOT / 'scripts/report.json', root / 'not-json.pth']:
                with self.assertRaises(ValueError):
                    witness.checked_output(output, root / 'data')
            output = root / 'reports/new.json'
            self.assertEqual(witness.checked_output(output, root / 'data'), output.resolve())

    def test_exact_comparison_checks_identity_and_container_shape(self):
        no_tensor = SimpleNamespace(Tensor=())
        value = dict(sample_ids=['sample1'], rng=(1, 2, [3, 4]))
        self.assertTrue(witness.compare(value, value, no_tensor)['bitwise_equal'])
        for wrong in [dict(sample_ids=['sample2'], rng=(1, 2, [3, 4])),
                      dict(sample_ids=['sample1']),
                      dict(sample_ids=('sample1',), rng=(1, 2, [3, 4]))]:
            with self.assertRaises(AssertionError):
                witness.compare(value, wrong, no_tensor, atol=1., rtol=1.)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch in the target environment')
    def test_tensor_comparison_requires_explicit_tolerance_and_finite_values(self):
        import torch
        reference = dict(value=torch.tensor([1., 2.]))
        changed = dict(value=torch.tensor([1.000001, 2.]))
        self.assertTrue(witness.compare(reference, reference, torch)['bitwise_equal'])
        with self.assertRaises(AssertionError):
            witness.compare(reference, changed, torch)
        result = witness.compare(reference, changed, torch, atol=2e-6)
        self.assertFalse(result['bitwise_equal'])
        self.assertGreater(result['max_absolute_difference'], 0)
        for invalid in [torch.tensor([1., float('nan')]), torch.tensor([1, 2])]:
            with self.assertRaises(AssertionError):
                witness.compare(reference, dict(value=invalid), torch, atol=1.)


if __name__ == '__main__':
    unittest.main()
