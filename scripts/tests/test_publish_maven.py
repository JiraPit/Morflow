import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_maven


def fake_sign(path):
    Path(f"{path}.asc").write_text("signature")


class MavenBundleTests(unittest.TestCase):
    def make_artifacts(self, directory):
        for suffix in (".jar", "-sources.jar", "-javadoc.jar", ".pom"):
            (directory / f"morflow-0.3.2{suffix}").write_bytes(suffix.encode())
        return publish_maven.artifact_files(directory, "morflow", "0.3.2")

    def test_bundle_has_repository_layout_signatures_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            files = self.make_artifacts(tmp / "in")
            bundle = publish_maven.build_bundle(files, "org.morflow", "morflow", "0.3.2", tmp / "out", sign=fake_sign)
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                jar = archive.read("org/morflow/morflow/0.3.2/morflow-0.3.2.jar")
                sha1 = archive.read("org/morflow/morflow/0.3.2/morflow-0.3.2.jar.sha1").decode()
        base = "org/morflow/morflow/0.3.2/morflow-0.3.2"
        for suffix in (".jar", "-sources.jar", "-javadoc.jar", ".pom"):
            for extra in ("", ".asc", ".md5", ".sha1", ".asc.md5", ".asc.sha1"):
                self.assertIn(base + suffix + extra, names)
        self.assertEqual(sha1, hashlib.sha1(jar).hexdigest())

    def test_missing_artifact_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "morflow-0.3.2.jar").write_bytes(b"jar")
            with self.assertRaisesRegex(FileNotFoundError, "sources"):
                publish_maven.artifact_files(Path(tmp), "morflow", "0.3.2")

    def test_published_version_is_skipped_without_uploading(self):
        with patch.object(sys, "argv", ["publish", "unused", "--version", "0.3.1"]), \
             patch.object(publish_maven, "api", return_value={"published": True}) as api, \
             patch.object(publish_maven, "upload") as upload:
            self.assertEqual(publish_maven.main(), 0)
        api.assert_called_once_with("GET", "published", {"namespace": "org.morflow", "name": "morflow", "version": "0.3.1"})
        upload.assert_not_called()

    def test_failed_deployment_raises(self):
        with patch.object(publish_maven, "api", return_value={"deploymentState": "FAILED", "errors": {"x": ["bad"]}}):
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                publish_maven.wait_until_validated("id", timeout=1)


if __name__ == "__main__":
    unittest.main()
