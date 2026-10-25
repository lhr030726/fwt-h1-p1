"""Cache capacity changes must preserve geography bits and immutable outputs."""
import importlib.util
import unittest

import numpy as np

from fwt_v2 import geography


@unittest.skipUnless(importlib.util.find_spec('pyproj'), 'requires pyproj')
class GeographyCacheTests(unittest.TestCase):
    def tearDown(self):
        geography.configure_image_basis_cache()

    def test_repeated_geometry_is_exact_across_eviction(self):
        geos = [dict(crs='EPSG:32611', shape=[8, 8],
                     affine_transform=[500000 + 1000 * i, 50, 0, 5000000, 0, -50])
                for i in range(40)]
        speed = np.arange(64, dtype=np.float32).reshape(8, 8) / 10
        angle = np.linspace(-np.pi, np.pi, 64, dtype=np.float32).reshape(8, 8)
        baseline = []
        geography.configure_image_basis_cache(32)
        for geo in geos:
            baseline.append(geography.geographic_wind(speed, angle, geo, 50))
        self.assertEqual(geography._image_basis.cache_info().currsize, 32)
        geography.configure_image_basis_cache(1024)
        for index in list(range(40)) + list(reversed(range(40))):
            actual = geography.geographic_wind(speed, angle, geos[index], 50)
            for wanted, found in zip(baseline[index], actual):
                self.assertEqual(wanted.tobytes(), found.tobytes())
        info = geography._image_basis.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (40, 40, 40))
        affine, crs = geography.validate_georeference(geos[0], (8, 8), 50)
        basis = geography._image_basis(affine, crs, (8, 8))
        self.assertFalse(basis.flags.writeable)
        with self.assertRaises(ValueError):
            basis[0, 0, 0, 0] = 0

    def test_configuration_rejects_unbounded_or_invalid_size(self):
        for value in (None, True, 0, -1, 1025, 2.5):
            with self.assertRaises(ValueError):
                geography.configure_image_basis_cache(value)


if __name__ == '__main__':
    unittest.main()
