import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_changed_plugins


class PluginDetectionTests(unittest.TestCase):
    PLUGIN = {"name": "openblas", "version": "0.1.1", "library": "morflow_openblas_plugin"}

    def run_detector(self, assets, force="false"):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch.object(sys, "argv", ["detect", "--force", force]), \
                 patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}), \
                 patch.object(detect_changed_plugins, "discover_plugins", return_value=[self.PLUGIN]), \
                 patch.object(detect_changed_plugins, "get_release_assets", return_value=assets):
                detect_changed_plugins.main()
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def released(self, platforms):
        return {detect_changed_plugins.asset_name(self.PLUGIN, platform) for platform in platforms}

    def test_released_plugin_is_not_rebuilt(self):
        out = self.run_detector(self.released(detect_changed_plugins.PLUGIN_TARGETS))
        self.assertEqual(out["has_builds"], "false")
        self.assertEqual(json.loads(out["matrix"]), [])
        self.assertEqual(json.loads(out["plugins"]), [])

    def test_only_missing_platforms_are_built(self):
        platforms = list(detect_changed_plugins.PLUGIN_TARGETS)
        out = self.run_detector(self.released(platforms[1:]))
        jobs = json.loads(out["matrix"])
        self.assertEqual([job["target"]["platform"] for job in jobs], platforms[:1])
        self.assertEqual(jobs[0]["plugin"]["name"], "openblas")
        self.assertEqual(json.loads(out["plugins"]), [self.PLUGIN])

    def test_force_builds_every_platform(self):
        out = self.run_detector(self.released(detect_changed_plugins.PLUGIN_TARGETS), force="true")
        self.assertEqual(len(json.loads(out["matrix"])), len(detect_changed_plugins.PLUGIN_TARGETS))


if __name__ == "__main__":
    unittest.main()
