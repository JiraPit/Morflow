"""Pack libraries and the C interface library: release naming, build targets
and which ones the action workflow builds."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_changed_actions
import native_targets

ROOT = Path(__file__).resolve().parents[2]


def action(pack, name, version, plugins=()):
    return {"pack": pack, "name": name, "version": version, "path": f"actions/{pack}/{name}",
            "package": name, "library": name, "plugins": list(plugins)}


def released_actions(*actions):
    return {detect_changed_actions.asset_name(a, platform, ext)
            for a in actions for platform, ext in detect_changed_actions.TARGET_PLATFORMS}


class DetectionTests(unittest.TestCase):
    def detect(self, actions, assets, *args):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch.object(sys, "argv", ["detect", *args]), \
                 patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}), \
                 patch.object(detect_changed_actions, "discover_all_actions", return_value=actions), \
                 patch.object(detect_changed_actions, "get_release_assets", side_effect=lambda repo, tag, token: assets.get(tag, set())), \
                 patch.object(detect_changed_actions, "get_git_changed_files", return_value=[]):
                detect_changed_actions.main()
            lines = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return {"matrix": json.loads(lines["matrix"]), "packs": json.loads(lines["packs"]),
                "has_actions": lines["has_actions"], "has_more": lines["has_more"],
                "remaining": int(lines["remaining_count"])}

    @staticmethod
    def libraries(result):
        return sorted((j["pack_library"]["pack"], j["target"]["platform"]) for j in result["matrix"] if "pack_library" in j)

    def test_unreleased_pack_builds_its_library_for_every_platform(self):
        actions = [action("base", "identity", "0.3.4"), action("base", "to_tensor", "0.3.4")]
        result = self.detect(actions, {})
        self.assertEqual(self.libraries(result), [("base", p) for p in sorted(native_targets.PLATFORMS)])
        job = next(j for j in result["matrix"] if j.get("target", {}).get("platform") == "ios-arm64-sim" and "pack_library" in j)
        self.assertEqual(job["pack_library"], {"pack": "base", "version": "0.3.4"})
        self.assertEqual(job["target"], {"platform": "ios-arm64-sim", "ext": "dylib",
                                         "triple": "aarch64-apple-ios-sim", "os": "macos-latest"})
        self.assertEqual(result["packs"], [{"name": "base", "version": "0.3.4"}])

    def test_released_pack_libraries_are_never_rebuilt(self):
        actions = [action("base", "identity", "0.3.4")]
        tag = "action_packs/base/v0.3.4"
        assets = released_actions(*actions) | {native_targets.pack_asset("base", "0.3.4", p) for p in native_targets.PLATFORMS}
        result = self.detect(actions, {tag: assets})
        self.assertEqual(result["matrix"], [])
        self.assertEqual(result["has_actions"], "false")

    def test_only_missing_platforms_are_built(self):
        actions = [action("base", "identity", "0.3.4")]
        assets = released_actions(*actions) | {native_targets.pack_asset("base", "0.3.4", "android-aarch64")}
        result = self.detect(actions, {"action_packs/base/v0.3.4": assets})
        self.assertEqual(self.libraries(result), [("base", p) for p in sorted(native_targets.PLATFORMS) if p != "android-aarch64"])
        # The pack's release is updated even though every action is released.
        self.assertEqual(result["packs"], [{"name": "base", "version": "0.3.4"}])

    def test_plugin_packs_and_mixed_versions_have_no_library(self):
        actions = [action("linalg_blas", "matmul", "0.1.1", ["openblas"]),
                   action("tensor_stats", "sum", "0.4.1"), action("tensor_stats", "mean", "0.4.0")]
        self.assertEqual(self.libraries(self.detect(actions, {})), [])

    def test_action_filter_skips_pack_libraries_and_pack_filter_selects_one(self):
        actions = [action("base", "identity", "0.3.4"), action("audio_basics", "gain", "0.2.4")]
        self.assertEqual(self.libraries(self.detect(actions, {}, "--action", "gain")), [])
        result = self.detect(actions, {}, "--pack", "audio_basics")
        self.assertEqual({pack for pack, _ in self.libraries(result)}, {"audio_basics"})

    def test_force_rebuilds_released_libraries(self):
        actions = [action("base", "identity", "0.3.4")]
        assets = released_actions(*actions) | {native_targets.pack_asset("base", "0.3.4", p) for p in native_targets.PLATFORMS}
        result = self.detect(actions, {"action_packs/base/v0.3.4": assets}, "--force", "true")
        self.assertEqual(len(self.libraries(result)), len(native_targets.PLATFORMS))

    def test_batches_stay_under_the_matrix_limit_and_continue(self):
        actions = [action(f"pack{i}", "identity", "1.0.0") for i in range(10)]
        # 10 actions x 4 desktop platforms = 40 jobs; 10 packs x 9 platforms = 90 pack library jobs.
        result = self.detect(actions, {}, "--max-jobs", "50")
        self.assertEqual(len(result["matrix"]), 50)
        self.assertEqual(result["has_more"], "true")
        self.assertEqual(result["remaining"], 80)
        library_packs = {pack for pack, _ in self.libraries(result)}
        self.assertTrue(library_packs <= {p["name"] for p in result["packs"]})

    def test_every_real_pack_without_plugins_is_bundleable(self):
        packs = detect_changed_actions.bundleable_packs(detect_changed_actions.discover_all_actions(ROOT))
        self.assertTrue({"base", "audio_basics", "image_basics", "linalg_basics", "math_basics",
                         "nn_basics", "tensor_basics", "tensor_stats"} <= set(packs))
        self.assertFalse({"image_opencv", "linalg_blas", "tensor_blas"} & set(packs))


class NamingTests(unittest.TestCase):
    def test_release_asset_names(self):
        self.assertEqual(native_targets.pack_asset("base", "0.3.4", "android-aarch64"), "morflow_pack_base-0.3.4-android-aarch64.so")
        self.assertEqual(native_targets.pack_asset("base", "0.3.4", "ios-arm64-sim"), "morflow_pack_base-0.3.4-ios-arm64-sim.dylib")
        self.assertEqual(native_targets.capi_asset("windows-x86_64"), "libmorflow_capi-windows-x86_64.dll")
        self.assertEqual(native_targets.capi_asset("ios-arm64"), "libmorflow_capi-ios-arm64.dylib")

    def test_library_files_are_what_the_engine_loads(self):
        self.assertEqual(native_targets.library_file("morflow_pack_base", "android-x86_64"), "libmorflow_pack_base.so")
        self.assertEqual(native_targets.library_file("morflow_capi", "windows-x86_64"), "morflow_capi.dll")
        self.assertEqual(native_targets.library_file("morflow_capi", "darwin-arm64"), "libmorflow_capi.dylib")

    def test_every_platform_is_desktop_or_mobile(self):
        self.assertEqual(set(native_targets.PLATFORMS), set(native_targets.DESKTOP_PLATFORMS) | set(native_targets.MOBILE_PLATFORMS) | set(native_targets.APP_ONLY_PLATFORMS))
        self.assertEqual({p for p, _ in detect_changed_actions.TARGET_PLATFORMS}, set(native_targets.DESKTOP_PLATFORMS))
        for platform, ext in detect_changed_actions.TARGET_PLATFORMS:
            self.assertEqual(native_targets.extension(platform), ext)

    def test_unknown_platform_is_rejected(self):
        with self.assertRaises(SystemExit):
            native_targets.triple("android-arm")

    def test_android_environment_points_at_the_ndk(self):
        with tempfile.TemporaryDirectory() as ndk:
            import platform as host
            system = {"Linux": "linux-x86_64", "Darwin": "darwin-x86_64", "Windows": "windows-x86_64"}[host.system()]
            bin_dir = Path(ndk, "toolchains", "llvm", "prebuilt", system, "bin")
            bin_dir.mkdir(parents=True)
            (bin_dir / f"aarch64-linux-android{native_targets.ANDROID_API_LEVEL}-clang").touch()
            env = native_targets.cargo_env("android-aarch64", {"ANDROID_NDK_HOME": ndk})
            self.assertTrue(env["CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"].endswith("aarch64-linux-android24-clang"))
            self.assertTrue(env["AR_aarch64_linux_android"].endswith("llvm-ar"))
            with self.assertRaises(SystemExit):
                native_targets.cargo_env("android-x86_64", {"ANDROID_NDK_HOME": ndk})

    def test_ios_environment_sets_the_deployment_target(self):
        self.assertEqual(native_targets.cargo_env("ios-arm64", {})["IPHONEOS_DEPLOYMENT_TARGET"], "13.0")
        self.assertEqual(native_targets.cargo_env("ios-arm64", {"IPHONEOS_DEPLOYMENT_TARGET": "15.0"})["IPHONEOS_DEPLOYMENT_TARGET"], "15.0")


if __name__ == "__main__":
    unittest.main()
