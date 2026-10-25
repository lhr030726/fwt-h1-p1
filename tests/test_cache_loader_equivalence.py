"""Exercise real loader/cache integration and its interaction with model RNG."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from fwt_v2.engine import make_loader, seed_everything
from fwt_v2.sample_cache import SharedTensorCache


class FixedSamples(Dataset):
    def __init__(self, *args):
        self.samples = [dict(sample_id=str(i)) for i in range(9)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(x=torch.tensor([index / 10., (index + 1) / 10.]),
                    target=torch.tensor([index % 2], dtype=torch.float32), sample_id=str(index))


class LoaderEquivalenceTests(unittest.TestCase):
    def test_cache_preserves_shuffling_rng_loss_and_optimizer_updates(self):
        def run(enabled):
            seed_everything(123)
            model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.3),
                                        torch.nn.Linear(4, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            cache = SharedTensorCache(4096) if enabled else None
            with patch('fwt_v2.engine.RCDADataset', FixedSamples):
                loader = make_loader({}, 'backbone_train', None, None, 1, True,
                                     sample_cache=cache)
            order, losses = [], []
            for _ in range(3):
                for batch in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(batch['x']), batch['target'])
                    loss.backward()
                    optimizer.step()
                    order.extend(batch['sample_id'])
                    losses.append(loss.item())
            return order, losses, model.state_dict(), torch.get_rng_state(), cache
        plain, cached = run(False), run(True)
        self.assertEqual(plain[:2], cached[:2])
        self.assertTrue(torch.equal(plain[3], cached[3]))
        for name, expected in plain[2].items():
            self.assertTrue(torch.equal(expected, cached[2][name]), name)
        self.assertEqual(cached[4].stats()['hits'], 18)
        self.assertEqual(cached[4].stats()['misses'], 9)


if __name__ == '__main__':
    unittest.main()
