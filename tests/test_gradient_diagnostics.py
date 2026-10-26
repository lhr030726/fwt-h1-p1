"""CPU equivalence plus optional CUDA branch equivalence; no dataset required."""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from fwt_v2.gradient_diagnostics import gradient_l2_norm, gradients_finite


def legacy_finite(gradients):
    return all(torch.isfinite(g).all() for g in gradients)


def legacy_norm(gradients):
    return math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                        for g in gradients))


class GradientDiagnosticsTests(unittest.TestCase):
    def assert_unchanged(self, values, device='cpu'):
        gradients = [g.to(device) for g in values]
        copies = [g.clone() for g in gradients]
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
        expected_finite = legacy_finite(gradients)
        self.assertEqual(gradients_finite(gradients), expected_finite)
        expected_norm = legacy_norm(gradients)
        actual_norm = gradient_l2_norm(gradients)
        if math.isnan(expected_norm):
            self.assertTrue(math.isnan(actual_norm))
        else:
            self.assertEqual(actual_norm, expected_norm)
        for original, unchanged in zip(gradients, copies):
            torch.testing.assert_close(original, unchanged, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before is not None:
            self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state()))

    def cases(self):
        return [[], [torch.empty(0)], [torch.zeros(3)],
                [torch.tensor([3., 4.]), torch.tensor([-12.], dtype=torch.float64)],
                [torch.tensor([1e30, -1e30]), torch.tensor([1e150], dtype=torch.float64)],
                [torch.arange(24., dtype=torch.float32).reshape(4, 6).T,
                 torch.tensor([.125], dtype=torch.float16)],
                [torch.tensor([float('nan')]), torch.ones(2)],
                [torch.tensor([float('inf'), -float('inf')])]]

    def test_cpu_values_exact_rng_and_inputs_unchanged(self):
        for gradients in self.cases():
            with self.subTest(shapes=[list(g.shape) for g in gradients]):
                self.assert_unchanged(gradients)

    def test_batched_operations_on_cpu_exact(self):
        # Exercises stack/list algorithm without claiming to test CUDA scheduling.
        for gradients in self.cases()[1:]:
            with patch('fwt_v2.gradient_diagnostics._same_dense_cuda', return_value=True):
                self.assert_unchanged(gradients)

    def test_cpu_finite_short_circuits(self):
        with patch('fwt_v2.gradient_diagnostics.torch.isfinite',
                   side_effect=[torch.tensor(False), AssertionError('must not evaluate')]) as finite:
            self.assertFalse(gradients_finite([torch.ones(1), torch.ones(2)]))
        self.assertEqual(finite.call_count, 1)

    def test_nonfinite_rejected_before_norm_or_optimizer(self):
        # Read and execute the actual engine statements at the gradient boundary.
        tree = ast.parse((Path(__file__).parents[1] / 'fwt_v2' / 'engine.py').read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'gradients' for t in node.targets))
        owners = [node for node in ast.walk(tree) if hasattr(node, 'body')
                  and isinstance(node.body, list) and loop in node.body]
        self.assertEqual(len(owners), 1)
        body = owners[0].body
        start = body.index(loop)
        boundary = compile(ast.fix_missing_locations(ast.Module(
            body=body[start:start + 4], type_ignores=[])), '<engine gradient boundary>', 'exec')
        for value in [None, float('nan'), float('inf'), -float('inf')]:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            for param in model.parameters():
                param.grad = None if value is None else torch.full_like(param, value)
            before_parameters = [p.detach().clone() for p in model.parameters()]
            before_optimizer = deepcopy(optimizer.state_dict())
            with patch('fwt_v2.gradient_diagnostics.gradient_l2_norm',
                       side_effect=AssertionError('norm must not run')) as norm:
                environment = dict(model=model, optimizer=optimizer, gradient_norms=[],
                                   gradients_finite=gradients_finite, gradient_l2_norm=norm)
                with self.assertRaisesRegex(RuntimeError, 'Missing or nonfinite gradients'):
                    exec(boundary, environment)
                norm.assert_not_called()
            self.assertEqual(optimizer.state_dict(), before_optimizer)
            for before, after in zip(before_parameters, model.parameters()):
                self.assertTrue(torch.equal(before, after.detach()))

    def test_empty_and_iterable_helpers(self):
        self.assertTrue(gradients_finite([]))
        self.assertEqual(gradient_l2_norm([]), 0.)
        self.assertEqual(gradient_l2_norm(iter([torch.tensor([3., 4.])])), 5.)

    def test_adamw_update_identical(self):
        initial = torch.nn.Linear(3, 2)
        models = [deepcopy(initial), deepcopy(initial)]
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.0001) for model in models]
        diagnostics = [(legacy_finite, legacy_norm), (gradients_finite, gradient_l2_norm)]
        for model, optimizer, (finite, norm) in zip(models, optimizers, diagnostics):
            for parameter in model.parameters():
                parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) / 8
            gradients = [p.grad for p in model.parameters()]
            self.assertTrue(finite(gradients))
            self.assertGreater(norm(gradients), 0)
            optimizer.step()
        for first, second in zip(models[0].parameters(), models[1].parameters()):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]))

    @unittest.skipUnless(os.environ.get('FWT_RUN_CUDA_EQUIVALENCE') == '1'
                         and torch.cuda.is_available(), 'Explicit idle GPU opt-in required')
    def test_cuda_exact_values_rng_and_inputs_unchanged(self):
        for gradients in self.cases():
            self.assert_unchanged(gradients, 'cuda')

    @unittest.skipUnless(os.environ.get('FWT_RUN_CUDA_EQUIVALENCE') == '1'
                         and torch.cuda.is_available(), 'Explicit idle GPU opt-in required')
    def test_mixed_devices_fallback(self):
        gradients = [torch.tensor([3., 4.]), torch.tensor([12.], device='cuda')]
        self.assertEqual(gradients_finite(gradients), legacy_finite(gradients))
        self.assertEqual(gradient_l2_norm(gradients), legacy_norm(gradients))


if __name__ == '__main__':
    unittest.main()
