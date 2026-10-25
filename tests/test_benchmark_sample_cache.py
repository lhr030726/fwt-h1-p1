"""No Torch import or data access: verify index selection and identity hash framing."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sample_probe', ROOT / 'scripts/benchmark_sample_cache.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeHelpersTests(unittest.TestCase):
    def test_spread_is_unique_bounded_and_includes_endpoints(self):
        indices = probe.spread_indices(3197, 96)
        self.assertEqual((len(indices), len(set(indices)), indices[0], indices[-1]), (96, 96, 0, 3196))
        self.assertEqual(probe.spread_indices(3, 96), [0, 1, 2])
        self.assertEqual(probe.spread_indices(10, 1), [0])

    def test_hash_preserves_identity_type_and_structure(self):
        values = ['12', 12, 12., True, [1, 2], (1, 2), {'a': 'bc'}, {'ab': 'c'}]
        self.assertEqual(len({probe.fingerprint(value) for value in values}), len(values))
        self.assertEqual(probe.fingerprint({'a': 1, 'b': 2}), probe.fingerprint({'b': 2, 'a': 1}))
        self.assertNotEqual(probe.fingerprint({'sample_id': 'a'}), probe.fingerprint({'sample_id': 'b'}))

    def test_tensor_hash_covers_dtype_shape_and_bytes(self):
        class TensorStub:
            def __init__(self, dtype, shape, raw):
                self.dtype, self.shape, self.raw = dtype, shape, raw

            def detach(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return self

            def tobytes(self, order):
                return self.raw

        variants = [('float32', (1, 2), b'12345678'), ('int32', (1, 2), b'12345678'),
                    ('float32', (2, 1), b'12345678'), ('float32', (1, 2), b'12345679')]
        hashes = {probe.fingerprint({'tensor': TensorStub(*args)}, TensorStub) for args in variants}
        self.assertEqual(len(hashes), len(variants))


if __name__ == '__main__':
    unittest.main()
