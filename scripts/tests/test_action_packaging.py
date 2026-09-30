import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_action import package_action, host_platform
from merge_release_assets import merge
import detect_changed_actions


class PackagingTests(unittest.TestCase):
    def test_versioned_files_receipts_catalogs_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action = root / 'actions' / 'custom' / 'transform'
            action.mkdir(parents=True)
            manifest = action / 'Cargo.toml'
            manifest.write_text('[package]\nname="transform"\nversion="0.1.0"\n')
            (action / 'SPEC.md').write_text('version one')
            binary = root / 'build.so'
            binary.write_bytes(b'one')
            cache = root / 'cache'
            exact = package_action(manifest, binary, cache)
            target, ext = host_platform()
            latest = cache / 'custom' / f'transform_action-latest-{target}.{ext}'
            self.assertFalse(latest.exists())
            package_action(manifest, binary, cache, latest=True)
            manifest.write_text('[package]\nname="transform"\nversion="0.2.0"\n')
            binary.write_bytes(b'two')
            package_action(manifest, binary, cache, latest=True)
            self.assertEqual(exact.read_bytes(), b'one')
            self.assertEqual(latest.read_bytes(), b'two')
            receipt = json.loads(Path(str(latest) + '.json').read_text())
            self.assertEqual(receipt['version'], 'latest')
            self.assertEqual(receipt['concrete_version'], '0.2.0')
            self.assertEqual(receipt['sha256'], hashlib.sha256(b'two').hexdigest())
            catalog = json.loads((cache / 'custom' / '.catalogs' / 'latest.json').read_text())
            self.assertEqual(catalog['concrete_version'], '0.2.0')

    def test_batches_keep_existing_assets_and_reject_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            old.mkdir(); new.mkdir()
            (old / 'first.so').write_bytes(b'first')
            (old / 'checksums.txt').write_text('old manifest')
            (new / 'second.so').write_bytes(b'second')
            merge(old, new)
            self.assertEqual((new / 'first.so').read_bytes(), b'first')
            self.assertEqual((new / 'second.so').read_bytes(), b'second')
            self.assertFalse((new / 'checksums.txt').exists())
            (new / 'first.so').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                merge(old, new)

    def test_release_groups_include_pack_and_version(self):
        actions = [{'pack': 'base', 'name': 'identity', 'version': '0.1.0'},
                   {'pack': 'base', 'name': 'to_tensor', 'version': '0.2.0'}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'github-output'
            with patch.object(sys, 'argv', ['detect', '--event', 'workflow_dispatch']), \
                 patch.object(detect_changed_actions, 'discover_all_actions', return_value=actions), \
                 patch.object(detect_changed_actions, 'get_release_assets', return_value=set()), \
                 patch.dict('os.environ', {'GITHUB_OUTPUT': str(output)}):
                detect_changed_actions.main()
            outputs = dict(line.split('=', 1) for line in output.read_text().splitlines())
            self.assertEqual(json.loads(outputs['packs']), [{'name': 'base', 'version': '0.1.0'}, {'name': 'base', 'version': '0.2.0'}])


if __name__ == '__main__':
    unittest.main()
