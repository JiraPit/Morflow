from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_engine_version


class EngineDetectionTests(unittest.TestCase):
    def run_detector(self, event, ref, changed=None):
        version = detect_engine_version.parse_cargo_version(
            Path(__file__).resolve().parents[2] / "pipeline" / "Cargo.toml")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch.object(sys, "argv", ["detect", "--event", event, "--ref", ref.format(version=version)]), \
                 patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}), \
                 patch.object(detect_engine_version, "get_git_changed_files", return_value=changed or []):
                detect_engine_version.main()
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_matching_version_tag_releases(self):
        out = self.run_detector("push", "refs/tags/v{version}")
        self.assertEqual((out["should_build"], out["should_release"]), ("true", "true"))

    def test_tag_that_does_not_match_the_version_fails(self):
        with self.assertRaises(SystemExit):
            self.run_detector("push", "refs/tags/v99.0.0")

    def test_branch_push_with_engine_changes_builds_without_release(self):
        out = self.run_detector("push", "refs/heads/main", ["actions/base/identity/Cargo.toml", "Cargo.lock"])
        self.assertEqual((out["should_build"], out["should_release"]), ("true", "false"))

    def test_branch_push_without_engine_changes_skips(self):
        out = self.run_detector("push", "refs/heads/main", ["actions/base/identity/SPEC.md"])
        self.assertEqual((out["should_build"], out["should_release"]), ("false", "false"))

    def test_manual_run_builds_without_release(self):
        out = self.run_detector("workflow_dispatch", "refs/heads/main")
        self.assertEqual((out["should_build"], out["should_release"]), ("true", "false"))


if __name__ == "__main__":
    unittest.main()
