"""The check before tagging the Unity package."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_unity_release
import native_targets

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    "package": "0.1.0",
    "package_constant": "0.1.0",
    "engine_constant": "0.3.5",
    "engine": "0.3.5",
    "changelog": ["0.1.0"],
}
EVERY_LIBRARY = {native_targets.capi_asset(p) for p in native_targets.PLATFORMS} | {"checksums.txt"}


class UnityReleaseTests(unittest.TestCase):
    def test_a_ready_package_passes(self):
        self.assertEqual(check_unity_release.problems("unity-v0.1.0", VERSIONS, EVERY_LIBRARY), [])

    def test_the_tag_must_name_a_version(self):
        for tag in ["v0.1.0", "unity-0.1.0", "unity-v0.1", "unity-v0.1.0+1"]:
            self.assertIn("is not a unity-v", check_unity_release.problems(tag, VERSIONS, EVERY_LIBRARY)[0])

    def test_the_package_versions_and_changelog_must_match_the_tag(self):
        [problem] = check_unity_release.problems("unity-v0.2.0", VERSIONS, EVERY_LIBRARY)
        self.assertIn("package.json has 0.1.0", problem)
        self.assertIn("PackageVersion is 0.1.0", problem)
        self.assertIn("CHANGELOG.md has no entry", problem)

    def test_the_engine_must_match_and_be_released(self):
        versions = dict(VERSIONS, engine_constant="0.3.4")
        [problem] = check_unity_release.problems("unity-v0.1.0", versions, EVERY_LIBRARY)
        self.assertIn("EngineVersion is 0.3.4", problem)
        [problem] = check_unity_release.problems("unity-v0.1.0", VERSIONS, None)
        self.assertIn("does not exist yet", problem)
        partial = EVERY_LIBRARY - {native_targets.capi_asset("ios-arm64")}
        [problem] = check_unity_release.problems("unity-v0.1.0", VERSIONS, partial)
        self.assertIn("libmorflow_capi-ios-arm64.dylib", problem)

    def test_the_repository_is_consistent(self):
        versions = check_unity_release.unity_versions(ROOT)
        self.assertEqual(versions["package"], versions["package_constant"])
        self.assertEqual(versions["engine"], versions["engine_constant"])
        self.assertIn(versions["package"], versions["changelog"])


if __name__ == "__main__":
    unittest.main()
