"""Small control/safety tests; no real data, subprocess GPU query or torch required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('benchmark_real_batches', ROOT / 'scripts/benchmark_real_batches.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RealBatchProbeTests(unittest.TestCase):
    def test_default_plan_does_not_open_data_query_gpu_or_execute(self):
        with patch.object(probe, 'require_no_jobs', side_effect=AssertionError('queried GPU')), \
                patch.object(probe, 'execute', side_effect=AssertionError('executed')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('read data')), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('read data')), \
                patch('builtins.print') as printed:
            self.assertEqual(probe.main(['--data-root', 'absent-real-data']), 0)
        plan = json.loads(printed.call_args.args[0])
        self.assertEqual(plan['status'], 'plan_only')
        self.assertEqual(plan['checkpoint_reads'], 0)
        self.assertEqual(plan['checkpoint_writes'], 0)
        self.assertEqual([r['timed_sample_exposures'] for r in plan['configurations']], [288] * 3)
        self.assertEqual([r['timed_updates'] for r in plan['configurations']], [288, 18, 9])
        self.assertEqual([r['warmup_updates'] for r in plan['configurations']], [96, 6, 3])

    def test_bounds_prevent_unbounded_exposure_or_duplicate_trials(self):
        for argv in [['--samples', '97'], ['--passes', '4'], ['--batches', '2'],
                     ['--batches', '1', '1'], ['--phases', 'det', 'det'],
                     ['--cache-mib', '4096'], ['--blocked-pid', '0']]:
            with self.subTest(argv=argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                probe.parse_args(argv)
        plan = probe.make_plan(probe.parse_args(['--samples', '32', '--passes', '1']))
        self.assertEqual([c['timed_sample_exposures'] for c in plan['configurations']], [32] * 3)

    def test_training_and_formal_queue_detection(self):
        for command in [['/usr/bin/python3', '/repo/train.py', '--phase', 'diff'],
                        ['python', '-m', 'fwt_v2', 'train'],
                        ['python', 'scripts/benchmark_real_batches.py', '--execute'],
                        ['bash', '/repo/fwt_cache_formal_pipeline.sh'],
                        ['bash', '/repo/launch_fwt_cache_formal.sh'],
                        ['bash', '/repo/fwt_pipeline.sh'],
                        ['bash', '/repo/fwt_wait_and_probe_gpu.sh']]:
            self.assertTrue(probe.training_command(command), command)
        for command in [[], ['python', 'serve_monitor.py'], ['bash', 'read_status.sh'],
                        ['bash', '-c', 'echo train.py'], ['python', '-m', 'unittest']]:
            self.assertFalse(probe.training_command(command), command)

    def test_proc_scan_ignores_zombie_missing_and_self_but_refuses_live_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'self').mkdir()
            (root / 'self' / 'stat').write_text('available')
            own_pid = os.getpid()
            pids = [own_pid, own_pid + 100, own_pid + 101, own_pid + 102]
            for pid, state in zip(pids, ('S', 'Z', 'T', None)):
                target = root / str(pid)
                target.mkdir()
                (target / 'cmdline').write_bytes(b'python\0/repo/train.py\0')
                if state:
                    (target / 'stat').write_text(f'{pid} (python (worker)) {state} 0')
            found = probe.active_training_processes(root)
        self.assertEqual([row['pid'] for row in found], [pids[2]])
        self.assertEqual(found[0]['state'], 'T')

    def test_guard_allows_own_context_rejects_others_and_query_failures(self):
        args = probe.parse_args([])
        with patch.object(probe, 'active_training_processes', return_value=[]), \
                patch.object(probe, 'active_gpu_processes', return_value=[dict(pid=os.getpid())]):
            probe.require_no_jobs(args)
        with patch.object(probe, 'active_training_processes', return_value=[]), \
                patch.object(probe, 'active_gpu_processes', return_value=[dict(pid=os.getpid() + 1)]), \
                self.assertRaisesRegex(RuntimeError, 'Other GPU'):
            probe.require_no_jobs(args)
        with patch.object(probe, 'active_training_processes', return_value=[]), \
                patch.object(probe, 'active_gpu_processes', side_effect=RuntimeError('query unavailable')), \
                self.assertRaisesRegex(RuntimeError, 'query unavailable'):
            probe.require_no_jobs(args)

    def test_guard_blocks_cpu_training_and_live_blocked_pid_before_gpu_query(self):
        args = probe.parse_args(['--blocked-pid', '11392'])
        for state in ('S', 'T', 'D'):
            with patch.object(probe, 'blocked_process_state', return_value=state), \
                    patch.object(probe, 'active_gpu_processes') as gpu, \
                    self.assertRaisesRegex(RuntimeError, 'Blocked process'):
                probe.require_no_jobs(args)
            gpu.assert_not_called()
        for state in ('missing', 'Z'):
            with patch.object(probe, 'blocked_process_state', return_value=state), \
                    patch.object(probe, 'active_training_processes', return_value=[]), \
                    patch.object(probe, 'active_gpu_processes', return_value=[]):
                probe.require_no_jobs(args)
        with patch.object(probe, 'active_training_processes', return_value=[dict(pid=321)]), \
                patch.object(probe, 'active_gpu_processes') as gpu, \
                self.assertRaisesRegex(RuntimeError, 'Other training'):
            probe.require_no_jobs(probe.parse_args([]))
        gpu.assert_not_called()

    def test_observer_error_is_retained_until_next_batch_check(self):
        observer = probe.Observer(probe.parse_args([]))
        with patch.object(probe, 'require_no_jobs', side_effect=RuntimeError('other job')), \
                patch.object(probe, 'memory_snapshot') as snapshot:
            observer.poll()
        snapshot.assert_not_called()
        self.assertTrue(observer.stop.is_set())
        with self.assertRaisesRegex(RuntimeError, 'other job'):
            observer.check()
        observer.close()

    def test_observer_reports_only_samples_in_window_and_unknown_peaks_as_null(self):
        observer = probe.Observer(probe.parse_args([]))
        observer.records = [dict(at=1., poll_seconds=.1, memory=dict(rss_bytes=100)),
                            dict(at=2., poll_seconds=.2, memory=dict(rss_bytes=200))]
        summary = observer.summary(1.5, 2.5)
        self.assertEqual(summary['polls'], 1)
        self.assertEqual(summary['poll_elapsed_seconds'], .2)
        self.assertEqual(summary['sampled_maximum']['rss_bytes'], 200)
        self.assertIsNone(summary['sampled_maximum']['cgroup_current_bytes'])

    def test_warm_cache_requires_retention_and_exact_hit_exposure(self):
        before = dict(entries=96, hits=0, misses=96, evictions=0, oversize_skips=0, disabled_bypasses=0)
        after = dict(before, hits=96)
        self.assertEqual(probe.verify_warm_cache(before, after, 96, 96)['misses'], 0)
        for field, value in [('entries', 95), ('misses', 97), ('evictions', 1),
                             ('hits', 95), ('oversize_skips', 1), ('disabled_bypasses', 1)]:
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'warm-cache'):
                probe.verify_warm_cache(before, dict(after, **{field: value}), 96, 96)

    def test_nonzero_gradient_acceptance_and_initialization_matching(self):
        probe.require_positive_gradient_norms([1e-15, 2.])
        for norms in ([], [0.], [1., 0.], [float('nan')], [float('inf')]):
            with self.subTest(norms=norms), self.assertRaisesRegex(RuntimeError, 'nonzero gradient'):
                probe.require_positive_gradient_norms(norms)
        trials = [dict(initial_model_sha256='head', frozen_backbone=dict(state_sha256_before='backbone'))
                  for _ in range(3)]
        self.assertTrue(probe.verify_initializations(trials, 'diff')['all_batch_initializations_equal'])
        trials[1]['frozen_backbone']['state_sha256_before'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'frozen backbone'):
            probe.verify_initializations(trials, 'diff')
        trials[1]['initial_model_sha256'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'optimized model'):
            probe.verify_initializations(trials, 'det')

    def test_atomic_report_does_not_touch_existing_partial_or_leave_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.json'
            output.write_text('owned report')
            partial = root / 'report.json.partial'
            partial.write_text('unrelated existing partial')
            probe.write_report(output, dict(status='completed'))
            self.assertEqual(json.loads(output.read_text())['status'], 'completed')
            self.assertEqual(partial.read_text(), 'unrelated existing partial')
            self.assertEqual(set(root.iterdir()), {output, partial})
            # Invalid report serialization must leave the last good report and
            # unrelated partial intact, and remove only the new owned tempfile.
            with self.assertRaises(ValueError):
                probe.write_report(output, dict(invalid=float('nan')))
            self.assertEqual(json.loads(output.read_text())['status'], 'completed')
            self.assertEqual(partial.read_text(), 'unrelated existing partial')
            self.assertEqual(set(root.iterdir()), {output, partial})

    def test_output_must_be_new_and_outside_data_training_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing.json'
            existing.write_text('original')
            for output in [existing, root / 'data' / 'report.json', root / 'runs' / 'report.json',
                           ROOT / 'scripts' / 'report.json']:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    probe.checked_output(output, root / 'data')
            self.assertEqual(existing.read_text(), 'original')
            self.assertEqual(probe.checked_output(root / 'new.json', root / 'data'), root / 'new.json')

    def test_failure_report_is_retained_and_no_trial_retried(self):
        def fail(args, report, save):
            report['phases'].append(dict(phase='det', status='running', trials=[dict(status='running')]))
            raise RuntimeError('OOM-like failure')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed.json'
            with patch.object(probe.sys, 'platform', 'linux'), \
                    patch.object(probe, 'require_no_jobs'), patch.object(probe, 'memory_snapshot', return_value={}), \
                    patch.object(probe, 'execute', side_effect=fail) as execute, patch('builtins.print'), \
                    self.assertRaisesRegex(RuntimeError, 'OOM-like failure'):
                probe.main(['--execute', '--data-root', str(Path(directory) / 'data'), '--output', str(output)])
            execute.assert_called_once()
            report = json.loads(output.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['phases'][0]['trials'][0]['status'], 'failed')
        self.assertEqual(report['error_type'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
