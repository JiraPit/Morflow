import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_action import package_action, host_platform
from package_plugin import package_plugin


class PluginPackagingTests(unittest.TestCase):
    def test_exact_latest_and_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'Cargo.toml'
            manifest.write_text('[package]\nname="plugin-fixture"\nversion="0.1.0"\n[lib]\nname="fixture_bridge"\n[package.metadata.morflow]\nplugin="fixture-bridge"\n')
            binary = root / 'plugin.so'
            binary.write_bytes(b'first')
            cache = root / 'cache'
            exact = package_plugin(manifest, binary, cache)
            target, ext = host_platform()
            latest = cache / 'fixture-bridge' / f'fixture-bridge_plugin-latest-{target}.{ext}'
            self.assertFalse(latest.exists())
            manifest.write_text(manifest.read_text().replace('0.1.0', '0.2.0'))
            binary.write_bytes(b'second')
            package_plugin(manifest, binary, cache, latest=True)
            self.assertEqual(exact.read_bytes(), b'first')
            self.assertEqual(latest.read_bytes(), b'second')
            receipt = json.loads(Path(str(latest) + '.json').read_text())
            self.assertEqual(receipt['version'], 'latest')
            self.assertEqual(receipt['concrete_version'], '0.2.0')
            self.assertEqual(receipt['sha256'], hashlib.sha256(b'second').hexdigest())

    def test_action_receipt_and_checksummed_metadata_have_plugin_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'actions' / 'image_opencv' / 'resize' / 'Cargo.toml'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[package]\nname="image_opencv_resize"\nversion="0.1.0"\n[package.metadata.morflow]\naction="resize"\n[[package.metadata.morflow.plugins]]\nname="opencv-bridge"\nversion="^0.1.0"\n')
            binary = root / 'action.so'
            binary.write_bytes(b'action')
            cache = root / 'cache'
            path = package_action(manifest, binary, cache)
            receipt = json.loads(Path(str(path) + '.json').read_text())
            metadata = cache / 'image_opencv' / 'resize_METADATA.json'
            self.assertEqual(receipt['plugins'], json.loads(metadata.read_text())['plugins'])
            catalog = json.loads((cache / 'image_opencv' / '.catalogs' / '0.1.0.json').read_text())
            self.assertEqual(catalog['checksums'][metadata.name], hashlib.sha256(metadata.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
