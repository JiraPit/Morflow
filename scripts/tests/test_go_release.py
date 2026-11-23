"""The check before tagging the Go package."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_go_release
import native_targets

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"engine_constant": "0.3.5", "engine": "0.3.5", "changelog": ["0.3.5"]}
DESKTOP = {native_targets.capi_asset(p) for p in check_go_release.GO_PLATFORMS} | {"checksums.txt"}


class GoReleaseTests(unittest.TestCase):
    def test_a_ready_package_passes(self):
        self.assertEqual(check_go_release.problems("bindings/go/v0.3.5", VERSIONS, DESKTOP), [])

    def test_the_tag_must_be_a_go_module_tag(self):
        for tag in ["v0.3.5", "go-v0.3.5", "bindings/go/0.3.5", "bindings/go/v0.3", "bindings/go/v0.3.5-rc1"]:
            self.assertIn("is not a bindings/go/v", check_go_release.problems(tag, VERSIONS, DESKTOP)[0])

    def test_the_changelog_must_describe_the_version(self):
        [problem] = check_go_release.problems("bindings/go/v0.3.6", VERSIONS, DESKTOP)
        self.assertIn("CHANGELOG.md has no entry for 0.3.6", problem)

    def test_the_engine_must_match_and_be_released(self):
        [problem] = check_go_release.problems("bindings/go/v0.3.5", dict(VERSIONS, engine_constant="0.3.4"), DESKTOP)
        self.assertIn("EngineVersion in bindings/go/version.go is 0.3.4", problem)
        [problem] = check_go_release.problems("bindings/go/v0.3.5", VERSIONS, None)
        self.assertIn("does not exist yet", problem)
        partial = DESKTOP - {native_targets.capi_asset("darwin-x86_64")}
        [problem] = check_go_release.problems("bindings/go/v0.3.5", VERSIONS, partial)
        self.assertIn("libmorflow_capi-darwin-x86_64.dylib", problem)

    def test_mobile_libraries_are_not_needed(self):
        self.assertTrue(all(not p.startswith(("android", "ios")) for p in check_go_release.GO_PLATFORMS))

    def test_the_repository_is_consistent(self):
        versions = check_go_release.go_versions(ROOT)
        self.assertEqual(versions["engine"], versions["engine_constant"])
        self.assertTrue(versions["changelog"], "CHANGELOG.md lists no versions")


if __name__ == "__main__":
    unittest.main()
