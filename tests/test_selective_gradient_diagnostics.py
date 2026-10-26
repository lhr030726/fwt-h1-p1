"""Exercise the real engine gradient boundary for both measured phase policies."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch

from fwt_v2.gradient_diagnostics import gradient_l2_norm, gradients_finite, legacy_gradient_l2_norm


def boundary():
    tree = ast.parse((Path(__file__).parents[1] / 'fwt_v2' / 'engine.py').read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Name) and node.iter.id == 'train_loader')
    start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'gradients' for t in node.targets))
    end = next(i + 1 for i, node in enumerate(loop.body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == 'optimizer' and node.value.func.attr == 'step')
    return compile(ast.fix_missing_locations(ast.Module(body=loop.body[start:end], type_ignores=[])),
                   '<actual selective gradient boundary>', 'exec')


class SelectiveGradientTests(unittest.TestCase):
    def execute(self, model, optimizer, h1):
        optimized_norm = Mock(wraps=gradient_l2_norm)
        legacy_norm = Mock(wraps=legacy_gradient_l2_norm)
        finite = Mock(wraps=gradients_finite)
        norms = []
        before_rng = torch.get_rng_state().clone()
        exec(boundary(), dict(model=model, optimizer=optimizer, head=None if h1 else object(),
                             gradient_norms=norms, torch=torch, gradients_finite=finite,
                             gradient_l2_norm=optimized_norm, legacy_gradient_l2_norm=legacy_norm))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(optimized_norm.call_count, int(h1))
        self.assertEqual(finite.call_count, int(h1))
        self.assertEqual(legacy_norm.call_count, int(not h1))
        return norms

    def test_phase_selection_norms_and_zero_gradients(self):
        for h1 in (True, False):
            for value in (0., 1.25):
                with self.subTest(h1=h1, value=value):
                    model = torch.nn.Linear(3, 2)
                    for parameter in model.parameters():
                        parameter.grad = torch.full_like(parameter, value)
                    expected = legacy_gradient_l2_norm([p.grad for p in model.parameters()])
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
                    self.assertEqual(self.execute(model, optimizer, h1), [expected])

    def test_missing_nonfinite_rejected_for_each_phase_before_norm_and_step(self):
        for h1 in (True, False):
            for value in (None, float('nan'), float('inf'), -float('inf')):
                with self.subTest(h1=h1, value=value):
                    model = torch.nn.Linear(3, 2)
                    before = deepcopy(model.state_dict())
                    for parameter in model.parameters():
                        parameter.grad = None if value is None else torch.full_like(parameter, value)
                    optimizer = Mock()
                    norms = []
                    norm = Mock(side_effect=AssertionError('norm must not run'))
                    with self.assertRaisesRegex(RuntimeError, 'Missing or nonfinite gradients'):
                        exec(boundary(), dict(model=model, optimizer=optimizer,
                             head=None if h1 else object(), gradient_norms=norms, torch=torch,
                             gradients_finite=gradients_finite, gradient_l2_norm=norm,
                             legacy_gradient_l2_norm=norm))
                    optimizer.step.assert_not_called()
                    norm.assert_not_called()
                    self.assertEqual(norms, [])
                    for name, value_before in before.items():
                        self.assertTrue(torch.equal(value_before, model.state_dict()[name]))

    def test_resumed_adamw_parameters_and_moments_exact_for_each_phase(self):
        initial = torch.nn.Linear(3, 2)
        initial_optimizer = torch.optim.AdamW(initial.parameters(), lr=.0001, weight_decay=.01)
        for parameter in initial.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        initial_optimizer.step()
        results = []
        for h1 in (True, False):
            model = deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
            optimizer.load_state_dict(deepcopy(initial_optimizer.state_dict()))
            for parameter in model.parameters():
                parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) / 8
            norms = self.execute(model, optimizer, h1)
            results.append((deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), norms))
        self.assertEqual(results[0][2], results[1][2])
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
        self.assertEqual(results[0][1]['param_groups'], results[1][1]['param_groups'])
        for parameter in results[0][1]['state']:
            for key in results[0][1]['state'][parameter]:
                self.assertTrue(torch.equal(results[0][1]['state'][parameter][key], results[1][1]['state'][parameter][key]))


if __name__ == '__main__':
    unittest.main()
