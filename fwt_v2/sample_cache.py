"""Optional, single-process RAM cache for deterministic dataset samples.

The cache is shared by wrapper datasets, so train and validation consume one
LRU budget. The budget covers retained CPU tensor storage, conservatively
counting repeated references separately. It is not a process-RSS limit: Python
metadata, original samples, returned copies, and allocator overhead are extra.
No labels, masks, order, random state, or provenance bindings are changed here.
Only use with deterministic, immutable underlying datasets and num_workers=0.
Caching is a performance hypothesis, not evidence of a speed improvement.
"""
from collections import OrderedDict
import operator
import os

import torch
from torch.utils.data import Dataset, get_worker_info


def _inspect(value, active=None):
    """Validate the supported tree and conservatively count tensor storage."""
    if type(value) is torch.Tensor:
        if (value.device.type != 'cpu' or value.layout != torch.strided
                or value.is_quantized or value.requires_grad):
            raise ValueError('Sample cache requires ordinary CPU tensors without gradients')
        return max(value.numel() * value.element_size(), value.untyped_storage().nbytes())
    if value is None or type(value) in (str, bool, int, float):
        return 0
    if type(value) not in (dict, list, tuple):
        raise TypeError('Unsupported sample-cache value: ' + type(value).__name__)
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError('Cyclic sample trees cannot be cached')
    active.add(identity)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('Sample-cache dictionary keys must be strings')
            return sum(_inspect(item, active) for item in value.values())
        return sum(_inspect(item, active) for item in value)
    finally:
        active.remove(identity)


def _clone(value):
    """Copy every mutable object; tensor clones do not consume RNG draws."""
    if type(value) is torch.Tensor:
        return value.clone(memory_format=torch.preserve_format)
    if type(value) is dict:
        return {key: _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    return value


class SharedTensorCache:
    """One byte-bounded LRU, shared explicitly by CachedDataset instances.

    This object is deliberately not transferable to DataLoader worker
    processes: spawn pickling is rejected, and forked use is detected by PID.
    Calls must remain sequential in the owning process (normal num_workers=0
    DataLoader behavior). Cached tensors are never returned directly.
    """

    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('max_bytes must be a nonnegative integer')
        self._max_bytes = max_bytes
        self._owner_pid = os.getpid()
        self._entries = OrderedDict()
        self._next_namespace = 0
        self._bytes = self._peak_bytes = 0
        self._hits = self._misses = self._evictions = 0
        self._oversize_skips = self._disabled_bypasses = 0

    @property
    def max_bytes(self):
        return self._max_bytes

    def _check_process(self):
        if os.getpid() != self._owner_pid or get_worker_info() is not None:
            raise RuntimeError('SharedTensorCache requires the owning process and num_workers=0')

    def __getstate__(self):
        raise TypeError('SharedTensorCache cannot be copied to workers; use num_workers=0')

    def _namespace(self):
        self._check_process()
        namespace = self._next_namespace
        self._next_namespace += 1
        return namespace

    def _get(self, key, load):
        self._check_process()
        if key in self._entries:
            self._hits += 1
            value, _ = self._entries.pop(key)
            self._entries[key] = (value, _)
            return _clone(value)
        self._misses += 1
        value = load()
        estimated_bytes = _inspect(value)
        if not self.max_bytes:
            self._disabled_bypasses += 1
            return _clone(value)
        if estimated_bytes > self.max_bytes:
            self._oversize_skips += 1
            return _clone(value)
        owned = _clone(value)
        size = _inspect(owned)
        if size > self.max_bytes:
            self._oversize_skips += 1
            return owned
        # Evict before admission. Returned copies and load/clone temporaries
        # are separate from the retained-storage budget documented above.
        while self._entries and self._bytes + size > self.max_bytes:
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self._bytes -= removed_bytes
            self._evictions += 1
        self._entries[key] = (owned, size)
        self._bytes += size
        self._peak_bytes = max(self._peak_bytes, self._bytes)
        return _clone(owned)

    def stats(self):
        """JSON-safe combined snapshot; reading it does not change LRU order."""
        self._check_process()
        return dict(schema='fwt_shared_cpu_sample_cache_v1', max_bytes=self.max_bytes,
                    bytes=self._bytes, peak_bytes=self._peak_bytes,
                    entries=len(self._entries), hits=self._hits, misses=self._misses,
                    evictions=self._evictions, oversize_skips=self._oversize_skips,
                    disabled_bypasses=self._disabled_bypasses,
                    budget_scope='retained_cpu_tensor_storage_only', num_workers=0)

    def clear(self):
        """Release retained samples; preserve cumulative diagnostic counters."""
        self._check_process()
        self._entries.clear()
        self._bytes = 0


class CachedDataset(Dataset):
    """Preserve scalar indexing and metadata while lazily caching samples.

    A private namespace prevents same-index collisions across datasets sharing
    one cache. The wrapped dataset must return deterministic samples; stochastic
    transforms would require a separate design and must not use this wrapper.
    A zero-byte cache never retains samples, but still validates and clones
    return values. For a zero-overhead disabled baseline, omit this wrapper.
    """

    _METADATA = frozenset(('samples', 'role', 'normalization', 'config', 'external_wind'))

    def __init__(self, dataset, cache):
        if not isinstance(cache, SharedTensorCache):
            raise TypeError('cache must be a SharedTensorCache')
        self.dataset = dataset
        self.cache = cache
        self._namespace = cache._namespace()

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # In particular, never forward __getitems__: DataLoader could then
        # bypass this wrapper using an underlying batched-fetch implementation.
        if name in self._METADATA:
            return getattr(self.dataset, name)
        raise AttributeError(name)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Dataset index out of range')
        return self.cache._get((self._namespace, index), lambda: self.dataset[index])
