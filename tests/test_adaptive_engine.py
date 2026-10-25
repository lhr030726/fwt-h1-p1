"""Epoch-boundary state checks with a scalar CPU parameter, never a FWT model."""
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fwt_v2 import training_state
from fwt_v2.training_control import TrainingController, policy_for_epochs
from test_training_control import history_for


def scalar_module():
    model = torch.nn.Module()
    model.register_parameter('value', torch.nn.Parameter(torch.tensor(1.0)))
    return model


def saved_boundary(count=8, budget=100):
    """Produce a real optimizer/control state from synthetic scalar updates."""
    model = scalar_module()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    control = TrainingController(optimizer, budget, policy_for_epochs(budget))
    history = history_for([.2] * count, budget)
    for _ in history:
        optimizer.zero_grad()
        model.value.grad = torch.tensor(.1)
        optimizer.step()
        control.step(.2)
    provenance = dict(requested_epochs=budget, training_control=policy_for_epochs(budget),
                      learning_rate=.001, device='cpu', selection_frozen=False)
    binding = dict(provenance=provenance)
    selected = dict(provenance, epoch=0, validation_ap=.2,
                    state_dict=deepcopy(model.state_dict()))
    for row in history:
        row['validation_operating_points'] = {'threshold': .5}
    terminal = count == budget or control.state['should_stop']
    diagnostics = dict(epochs=[row['training_checks'] for row in history],
        completed_epochs=count, remaining_epochs=0 if terminal else budget-count,
        total_updates=count, training_control_state=deepcopy(control.state),
        status='finishing' if terminal else 'stopped_after_epoch', selection_frozen=False,
        stop_reason=('early_stopping' if control.state['should_stop'] else
                     'max_epochs' if terminal else None))
    selected_validation = dict(epoch=0, validation=history[0]['validation'],
                               operating_points=history[0]['validation_operating_points'])
    with patch.object(training_state, 'atomic_checkpoint') as save:
        training_state.save_training_state('unused.pth', binding=binding, model=model,
            optimizer=optimizer, device='cpu', completed_epochs=count, history=history,
            selected=selected, selected_validation=selected_validation, diagnostics=diagnostics,
            updates=count, best=.2, training_control_state=control.state_dict())
    return save.call_args.args[1]


class AdaptiveEpochStateTests(unittest.TestCase):
    def test_det_engine_loop_early_freezes_scalar_mock_at_epoch16(self):
        self._run_engine_loop('det')

    def test_diff_engine_loop_early_freezes_with_scalar_frozen_backbone(self):
        self._run_engine_loop('diff')

    def _run_engine_loop(self, family):
        from fwt_v2 import engine, reporting

        class ScalarPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(.5))

            def forward(self, x, physics):
                return {'logits_det': self.value}

        class SyntheticLoader(list):
            def __init__(self, role):
                super().__init__([dict(x_norm=None, physics={}, target=None, valid_mask=None,
                                      sample_id=['synthetic-' + role])])
                self.dataset = SimpleNamespace(samples=[dict(sample_id='synthetic-' + role)])

        class ScalarHead(ScalarPredictor):
            def training_loss(self, target, outputs, physics, mask):
                self.last_training_diagnostics = dict(timestep_groups=[[0, 1]],
                    denoiser_evaluations=2, noise_draws=1)
                return self.value.square()

        documents, checkpoints = {}, {}
        def save_json(path, value):
            documents[Path(path).name] = deepcopy(value)

        def save_checkpoint(path, value):
            checkpoints[Path(path).name] = deepcopy(value)

        # The real control loop runs, but every FWT model/data/metric and disk
        # serialization boundary is replaced. Only one scalar CPU parameter is
        # updated per synthetic epoch; no fire data or FWT forward occurs.
        workspace = Path(__file__).resolve().parents[1] / 'runs'
        workspace.mkdir(exist_ok=True)
        data = dict(wind_convention='grid_towards_ccw_from_east', wind_reference='projected_grid',
                    wind_metadata_review='Synthetic control-flow fixture only; no wind data')
        backbone = ScalarPredictor()
        backbone_bundle = dict(family='det', architecture='fwt', source_sha256='a' * 64,
            selection_frozen=True, manifest_sha256='synthetic-manifest-sha',
            normalizer_sha256='b' * 64, model_config=asdict(engine.FWTConfig()), data_config=data)
        with tempfile.TemporaryDirectory(dir=workspace, prefix='scalar-control-test-') as temporary:
            args = SimpleNamespace(epochs=100, batch_size=1, learning_rate=.001,
                training_control=policy_for_epochs(100), resume=None, stop_after_epoch=None,
                output=str(Path(temporary) / 'phase'), manifest='synthetic-manifest',
                normalizer='synthetic-normalizer', seed=0, samples=8, sampling_steps=4,
                sampler='ddim', architecture='fwt', device='cpu', loss='focal', backbone='synthetic-h1')
            patches = [
                patch.object(engine, 'output_path', side_effect=lambda path: Path(path)),
                patch.object(engine, 'audited_manifest', return_value={}),
                patch.object(engine, 'manifest_digest', return_value='synthetic-manifest-sha'),
                patch.object(engine.NormalizationStats, 'load', return_value=None),
                patch.object(engine, 'seed_everything'),
                patch.object(engine, 'validate_wind_prior_config'),
                patch.object(engine, 'build_det', return_value=backbone),
                patch.object(engine, 'load_det', return_value=(backbone, backbone_bundle)),
                patch.object(engine, 'ConditionalResidualDiffusion', return_value=ScalarHead()),
                patch.object(engine, 'make_loader', side_effect=lambda manifest, role, *a: SyntheticLoader(role)),
                patch.object(engine, 'to_device', side_effect=lambda batch, device: batch),
                patch.object(engine, 'batch_diagnostics', return_value={'synthetic': True}),
                patch.object(engine, 'deterministic_loss', side_effect=lambda logits, *a: logits.square()),
                patch.object(engine, 'validation_records', return_value=[{'synthetic': True}]),
                patch.object(engine, 'prediction_diagnostics', return_value={'synthetic': True}),
                patch.object(engine, 'probability_metrics', return_value={'pooled_ap': .2}),
                patch.object(engine, 'select_raw_threshold', return_value=.5),
                patch.object(reporting, 'operating_point_report', return_value={'threshold_rule': 'synthetic'}),
                patch.object(engine, 'source_digest', return_value='a' * 64),
                patch.object(engine, 'sha256', return_value='b' * 64),
                patch.object(training_state, 'sample_file_identity', return_value=[]),
                patch.object(engine, 'atomic_json', side_effect=save_json),
                patch.object(engine, 'atomic_checkpoint', side_effect=save_checkpoint),
                patch.object(training_state, 'atomic_checkpoint', side_effect=save_checkpoint)]
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                for item in patches:
                    stack.enter_context(item)
                result = engine.train(args, {'data': data}, family)
        self.assertEqual(result['actual_epochs'], 16)
        self.assertEqual(result['max_epochs'], 100)
        self.assertEqual(result['stop_reason'], 'early_stopping')
        self.assertEqual(documents['selection.json']['epoch'], 0)
        self.assertEqual(documents['diagnostics.json']['selected_epoch'], 0)
        self.assertEqual(documents['diagnostics.json']['budget_remaining_epochs'], 84)
        self.assertTrue(checkpoints['best.pth']['selection_frozen'])
        if family == 'diff':
            self.assertTrue(documents['diagnostics.json']['frozen_backbone']['unchanged'])
            self.assertTrue(documents['diagnostics.json']['frozen_backbone']['gradients_absent'])
            self.assertFalse(backbone.value.requires_grad)
        rates = [row['training_control']['learning_rates_after'][0] for row in documents['history.json']]
        self.assertEqual(rates, [.001] * 6 + [.0005] * 6 + [.00025] * 4)
        last = checkpoints['last.pth']
        self.assertEqual(last['completed_epochs'], 16)
        self.assertEqual(last['diagnostics']['status'], 'finishing')
        with patch.object(training_state.torch, 'load', return_value=last):
            with self.assertRaisesRegex(ValueError, 'cannot be resumed'):
                training_state.load_training_state('unused.pth', last['binding'])

    def test_full_state_roundtrip_at_reduced_lr(self):
        saved = saved_boundary()
        with patch.object(training_state.torch, 'load', return_value=saved) as load:
            loaded = training_state.load_training_state('unused.pth', saved['binding'])
        self.assertTrue(load.call_args.kwargs['weights_only'])
        model = scalar_module()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        control = TrainingController(optimizer, 100, policy_for_epochs(100))
        training_state.restore_training_parameters(loaded, model, optimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'], .0005)
        control.load_state_dict(loaded['training_control_state'], loaded['history'])
        self.assertEqual(control.step(.2)['bad_epochs'], 8)
        self.assertEqual(optimizer.state[model.value]['step'].item(), 8)

    def test_completed_budget_and_early_boundary_cannot_resume(self):
        for count, budget in ((16, 100), (2, 2), (30, 30)):
            saved = saved_boundary(count, budget)
            with patch.object(training_state.torch, 'load', return_value=saved):
                with self.assertRaises(ValueError):
                    training_state.load_training_state('unused.pth', saved['binding'])

    def test_same_budget_interruption_remains_resumable(self):
        for count, budget in ((1, 2), (16, 30), (15, 100)):
            saved = saved_boundary(count, budget)
            with patch.object(training_state.torch, 'load', return_value=saved):
                actual = training_state.load_training_state('unused.pth', saved['binding'])
            self.assertEqual(actual['completed_epochs'], count)

    def test_changed_total_budget_or_recipe_rejected(self):
        saved = saved_boundary()
        for key, value in (('requested_epochs', 30), ('learning_rate', .002)):
            binding = deepcopy(saved['binding'])
            binding['provenance'][key] = value
            with patch.object(training_state.torch, 'load', return_value=saved):
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    training_state.load_training_state('unused.pth', binding)

    def test_restore_rejects_wrong_lr_even_with_recomputed_optimizer_digest(self):
        for key, value in (('lr', .001), ('weight_decay', .2)):
            saved = saved_boundary()
            saved['optimizer_state_dict']['param_groups'][0][key] = value
            saved['optimizer_integrity'] = training_state.optimizer_state_integrity(
                saved['optimizer_state_dict'])
            model = scalar_module()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            before = model.value.detach().clone()
            with self.assertRaises(ValueError):
                training_state.restore_training_parameters(saved, model, optimizer)
            self.assertTrue(torch.equal(model.value, before))

    def test_history_control_tamper_rejected_before_parameters_restore(self):
        saved = saved_boundary()
        saved['training_control_state']['progress']['bad_epochs'] = 0
        with patch.object(training_state.torch, 'load', return_value=saved):
            with self.assertRaises(ValueError):
                training_state.load_training_state('unused.pth', saved['binding'])

    def test_terminal_marker_blocks_resume_even_with_unfinished_counter(self):
        saved = saved_boundary()
        saved['diagnostics']['status'] = 'finishing'
        with patch.object(training_state.torch, 'load', return_value=saved):
            with self.assertRaises(ValueError):
                training_state.load_training_state('unused.pth', saved['binding'])


if __name__ == '__main__':
    unittest.main()
