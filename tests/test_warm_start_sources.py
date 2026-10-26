"""Strict compatibility checks metadata only and covers the full package."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fwt_v2.backbone_compat import verify_backbone_sources


class WarmStartSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        package = self.root / 'fwt_v2'
        package.mkdir()
        for path in (ROOT / 'fwt_v2').glob('*.py'):
            shutil.copyfile(path, package / path.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_check_requires_no_ml_imports(self):
        code = ('import sys; from fwt_v2.backbone_compat import verify_backbone_sources; '
                f'v=verify_backbone_sources({str(self.root)!r}, strict_training_only=True); '
                'assert v["strict_training_only"]; assert "torch" not in sys.modules; '
                'assert "numpy" not in sys.modules; assert "sample_cache.py" in v["unchanged_module_sha256"]')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_portable_and_diffusion_changes_rejected(self):
        for name in ('sample_cache.py', 'portable.py', 'diffusion.py'):
            with self.subTest(name=name):
                path = self.root / 'fwt_v2' / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n# changed\n')
                with self.assertRaisesRegex(ValueError, 'Backbone source changed'):
                    verify_backbone_sources(self.root, strict_training_only=True)
                path.write_bytes(original)

    def test_full_engine_sampling_and_unapproved_helpers_rejected(self):
        path = self.root / 'fwt_v2/engine.py'
        original = path.read_text(encoding='utf-8')
        for text in (original.replace('probability.clamp(1e-6, 1 - 1e-6)',
                                      'probability.clamp(1e-5, 1 - 1e-5)'),
                     original + '\ndef unapproved_helper():\n    return 1\n'):
            with self.subTest(kind='engine'):
                path.write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'engine contract changed'):
                    verify_backbone_sources(self.root, strict_training_only=True)


if __name__ == '__main__':
    unittest.main()
