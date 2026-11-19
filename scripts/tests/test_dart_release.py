"""The check before publishing the Dart package to pub.dev."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_dart_release
import native_targets

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    "bindings/dart/pubspec.yaml": "0.3.3",
    "bindings/dart/lib/src/version.dart": "0.3.3",
    "pipeline/Cargo.toml": "0.3.3",
}
EVERY_LIBRARY = {native_targets.capi_asset(p) for p in native_targets.PLATFORMS} | {"checksums.txt"}


class DartReleaseTests(unittest.TestCase):
    def test_a_complete_engine_release_can_be_published(self):
        self.assertEqual(check_dart_release.problems("dart-v0.3.3", VERSIONS, EVERY_LIBRARY), [])

    def test_a_dart_only_release_adds_a_build_number(self):
        versions = dict(VERSIONS, **{"bindings/dart/pubspec.yaml": "0.3.3+1"})
        self.assertEqual(check_dart_release.problems("dart-v0.3.3+1", versions, EVERY_LIBRARY), [])
        [problem] = check_dart_release.problems("dart-v0.3.3", versions, EVERY_LIBRARY)
        self.assertIn("pubspec.yaml has 0.3.3+1", problem)
        [problem] = check_dart_release.problems("dart-v0.3.3+2", versions, EVERY_LIBRARY)
        self.assertIn("pubspec.yaml has 0.3.3+1", problem)

    def test_the_tag_must_name_a_version(self):
        for tag in ["v0.3.3", "dart-0.3.3", "dart-v0.3", "dart-v0.3.3-beta", "dart-v0.3.3+"]:
            self.assertIn("is not a dart-v", check_dart_release.problems(tag, VERSIONS, EVERY_LIBRARY)[0])

    def test_every_version_must_match_the_tag(self):
        versions = dict(VERSIONS, **{"bindings/dart/lib/src/version.dart": "0.3.2"})
        [problem] = check_dart_release.problems("dart-v0.3.3", versions, EVERY_LIBRARY)
        self.assertIn("lib/src/version.dart has 0.3.2", problem)
        [problem] = check_dart_release.problems("dart-v0.3.4", VERSIONS, EVERY_LIBRARY)
        self.assertIn("pipeline/Cargo.toml has 0.3.3", problem)

    def test_the_engine_release_must_exist_and_hold_every_library(self):
        [problem] = check_dart_release.problems("dart-v0.3.3", VERSIONS, None)
        self.assertIn("v0.3.3 does not exist yet", problem)
        partial = EVERY_LIBRARY - {"libmorflow_capi-ios-arm64.dylib"}
        [problem] = check_dart_release.problems("dart-v0.3.3", VERSIONS, partial)
        self.assertIn("has no libmorflow_capi-ios-arm64.dylib", problem)

    def test_the_repository_versions_agree(self):
        versions = check_dart_release.dart_versions(ROOT)
        package = versions["bindings/dart/pubspec.yaml"]
        tag = f"dart-v{package}"
        self.assertEqual(check_dart_release.problems(tag, versions, EVERY_LIBRARY), [], versions)


if __name__ == "__main__":
    unittest.main()
