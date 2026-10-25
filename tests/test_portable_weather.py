"""Synthetic source semantics survive relocation; no arrays or models."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fwt_v2.hourly import (ExternalWindStore, canonical_digest,
    hourly_statistics, sample_identity_digest)
from fwt_v2.portable import file_sha256


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        geo = dict(crs='EPSG:4326', shape=[2, 2], affine_transform=[-120, .01, 0, 40, 0, -.01])
        self.row = dict(sample_id='2015_UID_1/2015-06-01', event_id='2015_UID_1',
            date='2015-06-01', target_date='2015-06-02', role='backbone_train', georeference=geo)
        hours = ['20150601' + f'{i:02d}' for i in range(24)]
        speed, direction = [2.] * 24, [90.] * 24
        stats = hourly_statistics(speed, direction)
        self.response = dict(header=dict(time_standard='UTC'),
            properties=dict(parameter=dict(WS50M=dict(zip(hours, speed)), WD50M=dict(zip(hours, direction)))),
            parameters=dict(WS50M=dict(units='m/s'), WD50M=dict(units='Degrees')),
            geometry=dict(coordinates=[-119.99, 39.99]))
        self.record = dict(self.row, identity_sha256=sample_identity_digest(self.row),
            georeference_sha256=canonical_digest(geo), hour_keys=hours, complete_hours=24,
            missing_values=0, longitude=-119.99, latitude=39.99,
            speed_m_s=speed, direction_from_north_cw_degrees=direction,
            u_east_m_s=stats['u'].tolist(), v_north_m_s=stats['v'].tolist(),
            source_response_path='E:/immutable/weather/day.json')
        self.document = dict(schema_version='fwt_power_hourly_manifest_v1',
            time_standard='UTC', height_above_surface_m=50, hours_per_sample=24,
            samples=[self.record])

    def tearDown(self):
        self.temp.cleanup()

    def package(self):
        response_path = self.root / 'day.json'
        response_path.write_text(json.dumps(self.response), encoding='utf-8')
        self.record['source_response_sha256'] = file_sha256(response_path)
        original_path = self.root / 'original.json'
        original_path.write_text(json.dumps(self.document), encoding='utf-8')
        original = dict(original_path='E:/immutable/weather/hourly.json',
                        path='original.json', sha256=file_sha256(original_path))
        wrapper = dict(schema_version='fwt_power_portable_manifest_v1', original_manifest=original,
            files=[original, dict(original_path=self.record['source_response_path'],
                                  path='day.json', sha256=file_sha256(response_path))])
        wrapper_path = self.root / 'weather_manifest.json'
        wrapper_path.write_text(json.dumps(wrapper), encoding='utf-8')
        return wrapper_path

    def test_valid_original_source_values_preserved(self):
        wrapper = self.package()
        wind = ExternalWindStore(wrapper, file_sha256(wrapper), {'samples': [self.row]}).for_sample(self.row)
        self.assertEqual(wind.manifest_sha256, file_sha256(wrapper))
        self.assertAlmostEqual(wind.mean_speed, 2.)
        self.assertEqual(wind.extras, (0., 2., 1., 0.))

    def test_rehashed_response_with_wrong_values_still_rejected(self):
        self.response['properties']['parameter']['WS50M']['2015060100'] = 3.
        wrapper = self.package()
        with self.assertRaisesRegex(ValueError, 'values differ from source response'):
            ExternalWindStore(wrapper, file_sha256(wrapper), {'samples': [self.row]})

    def test_derived_response_cannot_be_laundered_as_raw(self):
        self.response['derivation'] = dict(kind='DERIVED')
        wrapper = self.package()
        with self.assertRaisesRegex(ValueError, 'explicit v2 provenance'):
            ExternalWindStore(wrapper, file_sha256(wrapper), {'samples': [self.row]})

    def test_v2_missing_ledger_not_bypassed(self):
        self.document['schema_version'] = 'fwt_power_hourly_manifest_v2_derived_provenance'
        wrapper = self.package()
        with self.assertRaises(KeyError):
            ExternalWindStore(wrapper, file_sha256(wrapper), {'samples': [self.row]})


if __name__ == '__main__':
    unittest.main()
