"""CPU-only sample-cache invariants; no files, GPU, or training data needed."""
import pickle
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fwt_v2.sample_cache import CachedDataset, SharedTensorCache


class FakeDataset(Dataset):
    def __init__(self, count=3, offset=0):
        self.samples = [dict(sample_id=f'id-{offset + index}') for index in range(count)]
        self.role, self.config = 'validation', {'deterministic': True}
        self.calls = []
        self.values = [dict(
            x=torch.arange(6, dtype=torch.float32).reshape(2, 3).t() + offset + index,
            physics={'mask': torch.tensor([True, False])},
            mixed=[None, ('tag', index, 1.5, True)], sample_id=row['sample_id'])
            for index, row in enumerate(self.samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.calls.append(index)
        return self.values[index]


class SampleCacheTests(unittest.TestCase):
    ENTRY_BYTES = 6 * 4 + 2

    def assertTreeEqual(self, first, second):
        self.assertEqual(type(first), type(second))
        if type(first) is torch.Tensor:
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertTrue(torch.equal(first, second))
        elif type(first) is dict:
            self.assertEqual(list(first), list(second))
            for key in first:
                self.assertTreeEqual(first[key], second[key])
        elif type(first) in (list, tuple):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(first, second)

    def test_values_metadata_order_and_dataloader(self):
        source = FakeDataset()
        cache = SharedTensorCache(1024)
        wrapped = CachedDataset(source, cache)
        self.assertIs(wrapped.samples, source.samples)
        self.assertIs(wrapped.config, source.config)
        self.assertEqual(wrapped.role, source.role)
        for index in (2, 0, 2, 1):
            self.assertTreeEqual(wrapped[index], source.values[index])
        self.assertEqual(source.calls, [2, 0, 1])
        self.assertEqual(cache.stats()['hits'], 1)
        # A custom collator preserves this deliberately heterogeneous tree.
        observed = list(DataLoader(wrapped, batch_size=None, shuffle=False,
                                   num_workers=0, collate_fn=lambda sample: sample))
        self.assertEqual([value['sample_id'] for value in observed], ['id-0', 'id-1', 'id-2'])
        self.assertEqual(source.calls, [2, 0, 1])

    def test_cache_and_underlying_are_isolated_from_returned_mutations(self):
        source = FakeDataset(1)
        wrapped = CachedDataset(source, SharedTensorCache(1024))
        first = wrapped[0]
        first['x'].zero_()
        first['physics']['mask'][0] = False
        first['physics']['extra'] = torch.zeros(1)
        first['mixed'].append('mutated')
        first['sample_id'] = 'wrong'
        second = wrapped[0]
        self.assertTreeEqual(second, source.values[0])
        second['x'].fill_(999)
        self.assertTreeEqual(wrapped[0], source.values[0])
        # Even mutation of an underlying producer's reusable storage cannot
        # alter an already cached sample.
        expected = wrapped[0]
        source.values[0]['x'].fill_(-123)
        self.assertTreeEqual(wrapped[0], expected)

    def test_lru_eviction_and_exact_combined_byte_bound(self):
        source = FakeDataset()
        cache = SharedTensorCache(2 * self.ENTRY_BYTES)
        wrapped = CachedDataset(source, cache)
        for index in (0, 1, 0, 2):
            wrapped[index]
            self.assertLessEqual(cache.stats()['bytes'], cache.max_bytes)
        self.assertEqual(cache.stats()['bytes'], 2 * self.ENTRY_BYTES)
        self.assertEqual(cache.stats()['evictions'], 1)
        wrapped[0]  # MRU hit survived; index 1 was least recently used.
        self.assertEqual(source.calls, [0, 1, 2])
        wrapped[1]
        self.assertEqual(source.calls, [0, 1, 2, 1])
        self.assertEqual(cache.stats()['peak_bytes'], cache.max_bytes)

    def test_two_datasets_share_budget_without_key_collisions(self):
        cache = SharedTensorCache(self.ENTRY_BYTES)
        train_source, val_source = FakeDataset(1), FakeDataset(1, offset=100)
        train, val = CachedDataset(train_source, cache), CachedDataset(val_source, cache)
        self.assertEqual(train[0]['sample_id'], 'id-0')
        self.assertEqual(val[0]['sample_id'], 'id-100')
        self.assertEqual(cache.stats()['entries'], 1)
        self.assertEqual(cache.stats()['bytes'], self.ENTRY_BYTES)
        self.assertEqual(train[0]['sample_id'], 'id-0')
        self.assertEqual(train_source.calls, [0, 0])
        self.assertEqual(val_source.calls, [0])
        self.assertEqual(cache.stats()['evictions'], 2)

    def test_zero_budget_bypass_and_oversized_entries(self):
        for budget, counter in ((0, 'disabled_bypasses'), (self.ENTRY_BYTES - 1, 'oversize_skips')):
            with self.subTest(budget=budget):
                source, cache = FakeDataset(1), SharedTensorCache(budget)
                wrapped = CachedDataset(source, cache)
                first, second = wrapped[0], wrapped[0]
                self.assertTreeEqual(first, second)
                self.assertEqual(source.calls, [0, 0])
                self.assertEqual(cache.stats()['entries'], 0)
                self.assertEqual(cache.stats()['bytes'], 0)
                self.assertEqual(cache.stats()[counter], 2)

    def test_storage_views_are_conservatively_accounted(self):
        source = FakeDataset(1)
        source.values[0] = {'small_view': torch.zeros(1000)[::100]}
        cache = SharedTensorCache(100)
        wrapped = CachedDataset(source, cache)
        self.assertTreeEqual(wrapped[0], source.values[0])
        self.assertEqual(cache.stats()['oversize_skips'], 1)
        self.assertEqual(cache.stats()['bytes'], 0)

    def test_duplicate_references_are_independent_and_fully_charged(self):
        source = FakeDataset(1)
        tensor = torch.arange(4, dtype=torch.float32)
        source.values[0] = {'a': tensor, 'b': tensor}
        cache = SharedTensorCache(32)
        value = CachedDataset(source, cache)[0]
        self.assertEqual(cache.stats()['bytes'], 32)
        value['a'].zero_()
        self.assertTrue(torch.equal(value['b'], tensor))

    def test_cache_does_not_consume_python_numpy_or_torch_rng(self):
        source = FakeDataset()
        cache = SharedTensorCache(2 * self.ENTRY_BYTES)
        wrapped = CachedDataset(source, cache)
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        for index in (0, 1, 0, 2, 1):
            wrapped[index]
        cache.stats()
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_rejects_unsupported_values_and_non_cpu_tensor_without_gpu(self):
        cyclic = []
        cyclic.append(cyclic)
        values = [np.zeros(2), object(), {1: torch.zeros(1)}, cyclic,
                  torch.empty(2, device='meta'), torch.zeros(2, requires_grad=True),
                  torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.]), (2,))]
        for value in values:
            with self.subTest(kind=type(value)):
                source, cache = FakeDataset(1), SharedTensorCache(1024)
                source.values[0] = value
                with self.assertRaises((ValueError, TypeError)):
                    CachedDataset(source, cache)[0]
                self.assertEqual(cache.stats()['entries'], 0)

    def test_workers_and_pickling_cannot_duplicate_shared_cache(self):
        cache = SharedTensorCache(1024)
        wrapped = CachedDataset(FakeDataset(), cache)
        with patch('fwt_v2.sample_cache.get_worker_info', return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'num_workers=0'):
                wrapped[0]
        with patch('fwt_v2.sample_cache.os.getpid', return_value=-1):
            with self.assertRaisesRegex(RuntimeError, 'owning process'):
                wrapped[0]
        with self.assertRaisesRegex(TypeError, 'num_workers=0'):
            pickle.dumps(wrapped)

    def test_no_batched_fetch_delegation_negative_indices_and_clear(self):
        source = FakeDataset()
        source.__getitems__ = lambda indices: [source.values[i] for i in indices]
        cache, wrapped = SharedTensorCache(1024), None
        wrapped = CachedDataset(source, cache)
        self.assertFalse(hasattr(wrapped, '__getitems__'))
        self.assertTreeEqual(wrapped[-1], wrapped[2])
        self.assertEqual(source.calls, [2])
        for invalid in (3, -4):
            with self.assertRaises(IndexError):
                wrapped[invalid]
        cache.clear()
        self.assertEqual(cache.stats()['bytes'], 0)
        self.assertEqual(cache.stats()['entries'], 0)
        wrapped[2]
        self.assertEqual(source.calls, [2, 2])

    def test_invalid_budget_is_rejected(self):
        for value in (-1, True, 1.5, '1024'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SharedTensorCache(value)


if __name__ == '__main__':
    unittest.main()
