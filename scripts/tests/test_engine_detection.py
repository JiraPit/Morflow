from pathlib import Path
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_engine_version


class EngineDetectionTests(unittest.TestCase):
    def run_detector(self, released):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch.object(sys, "argv", ["detect"]), \
                 patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}), \
                 patch.object(detect_engine_version, "release_exists", return_value=released):
                detect_engine_version.main()
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_unreleased_version_builds_and_releases(self):
        out = self.run_detector(released=False)
        self.assertEqual(out["should_build"], "true")
        self.assertEqual(out["tag_name"], f"v{out['version']}")

    def test_released_version_is_skipped(self):
        self.assertEqual(self.run_detector(released=True)["should_build"], "false")

    def test_unreachable_api_is_an_error_not_unreleased(self):
        error = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
        with patch.object(detect_engine_version.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(urllib.error.HTTPError):
                detect_engine_version.release_exists("JiraPit/Morflow", "v0.0.0")

    def test_missing_release_reads_as_unreleased(self):
        error = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        with patch.object(detect_engine_version.urllib.request, "urlopen", side_effect=error):
            self.assertFalse(detect_engine_version.release_exists("JiraPit/Morflow", "v0.0.0"))


if __name__ == "__main__":
    unittest.main()
