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
    def test_interface_detection_covers_every_commit_in_a_push(self):
        from subprocess import CompletedProcess
        with patch.dict("os.environ", {"GITHUB_BASE_SHA": "abc123"}), \
             patch.object(detect_changed_actions.subprocess, "run", side_effect=[
                 CompletedProcess([], 0, "abc123", ""),
                 CompletedProcess([], 0, "plugins/openblas/src/interface.rs\n", ""),
             ]) as run:
            self.assertEqual(detect_changed_actions.get_git_changed_files(),
                             ["plugins/openblas/src/interface.rs"])
        self.assertEqual(run.call_args.args[0],
                         ["git", "diff", "--name-only", "abc123", "HEAD"])

    def test_published_consumer_cannot_silently_skip_interface_change(self):
        action = {"pack": "linalg_blas", "name": "matmul", "version": "0.1.1",
                  "path": "actions/linalg_blas/matmul"}
        assets = {f"matmul_action-0.1.1-{platform}.{extension}"
                  for platform, extension in detect_changed_actions.TARGET_PLATFORMS}
        with patch.object(sys, "argv", ["detect"]), \
             patch.object(detect_changed_actions, "discover_all_actions", return_value=[action]), \
             patch.object(detect_changed_actions, "get_release_assets", return_value=assets), \
             patch.object(detect_changed_actions, "get_git_changed_files", return_value=["plugins/openblas/src/interface.rs"]):
            with self.assertRaisesRegex(RuntimeError, "Bump the action version"):
                detect_changed_actions.main()

    def test_plugin_interface_changes_identify_consumers(self):
        changed = detect_changed_actions.changed_plugin_interfaces([
            "plugins/openblas/src/interface.rs",
            "plugins/opencv-bridge/src/interface.rs",
            "plugins/openblas/src/numerical.rs",
            "actions/image_opencv/resize/src/lib.rs",
        ])
        self.assertEqual(changed, {"openblas", "opencv-bridge"})
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            detect_changed_actions.interface_dependencies(
                root / "actions/linalg_blas/matmul/Cargo.toml"
            ),
            {"openblas"},
        )
        self.assertEqual(
            detect_changed_actions.interface_dependencies(
                root / "actions/image_opencv/resize/Cargo.toml"
            ),
            {"opencv-bridge"},
        )

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

    def test_namespaced_cargo_packages_keep_public_action_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action = root / 'actions' / 'tensor_blas' / 'concat'
            action.mkdir(parents=True)
            manifest = action / 'Cargo.toml'
            manifest.write_text('[package]\nname="tensor_blas_concat"\nversion="0.1.0"\n[package.metadata.morflow]\naction="concat"\n')
            binary = root / 'build.so'
            binary.write_bytes(b'blas action')
            cache = root / 'cache'
            artifact = package_action(manifest, binary, cache)
            self.assertTrue(artifact.name.startswith('concat_action-'))
            receipt = json.loads(Path(str(artifact) + '.json').read_text())
            self.assertEqual(receipt['action'], 'concat')
            discovered = detect_changed_actions.discover_all_actions(root)
            self.assertEqual(discovered[0]['name'], 'concat')
            self.assertEqual(discovered[0]['package'], 'tensor_blas_concat')
            self.assertEqual(discovered[0]['library'], 'tensor_blas_concat')

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
