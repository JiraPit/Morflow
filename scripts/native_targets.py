#!/usr/bin/env python3
"""Platforms Morflow ships native libraries for, and how to build for each.

Release workflows and scripts/build_pack.py / scripts/build_capi.py share this
table, so every native library is built and named the same way everywhere.
"""
import os
from pathlib import Path
import platform as host
import shutil

# Platform name -> Rust target, CI runner and library file naming.
PLATFORMS = {
    "linux-x86_64": {"triple": "x86_64-unknown-linux-gnu", "os": "ubuntu-latest"},
    "linux-aarch64": {"triple": "aarch64-unknown-linux-gnu", "os": "ubuntu-24.04-arm"},
    "darwin-arm64": {"triple": "aarch64-apple-darwin", "os": "macos-latest"},
    "windows-x86_64": {"triple": "x86_64-pc-windows-msvc", "os": "windows-latest"},
    "android-aarch64": {"triple": "aarch64-linux-android", "os": "ubuntu-latest"},
    "android-x86_64": {"triple": "x86_64-linux-android", "os": "ubuntu-latest"},
    "ios-arm64": {"triple": "aarch64-apple-ios", "os": "macos-latest"},
    "ios-arm64-sim": {"triple": "aarch64-apple-ios-sim", "os": "macos-latest"},
}

DESKTOP_PLATFORMS = ["linux-x86_64", "linux-aarch64", "darwin-arm64", "windows-x86_64"]
MOBILE_PLATFORMS = ["android-aarch64", "android-x86_64", "ios-arm64", "ios-arm64-sim"]

# Oldest Android API level and iOS version the libraries run on. Flutter's
# own minimums are Android 24 (7.0) and iOS 13.
ANDROID_API_LEVEL = 24
IOS_DEPLOYMENT_TARGET = "13.0"


def host_platform() -> str:
    """This machine's platform name."""
    system, machine = host.system(), host.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    name = {"Linux": f"linux-{arch}", "Windows": f"windows-{arch}", "Darwin": "darwin-arm64"}.get(system, "")
    if name not in DESKTOP_PLATFORMS:
        raise SystemExit(f"Unsupported host {system} {machine}")
    return name


def extension(platform: str) -> str:
    if platform.startswith("windows-"):
        return "dll"
    if platform.startswith(("darwin-", "ios-")):
        return "dylib"
    return "so"


def library_file(stem: str, platform: str) -> str:
    """The file name the operating system loads, e.g. libmorflow_capi.so."""
    prefix = "" if platform.startswith("windows-") else "lib"
    return f"{prefix}{stem}.{extension(platform)}"


def triple(platform: str) -> str:
    try:
        return PLATFORMS[platform]["triple"]
    except KeyError:
        raise SystemExit(f"Unknown platform {platform}; expected one of {', '.join(PLATFORMS)}") from None


def android_ndk(env: dict) -> Path:
    for variable in ("ANDROID_NDK_HOME", "ANDROID_NDK_LATEST_HOME", "ANDROID_NDK_ROOT"):
        if env.get(variable):
            return Path(env[variable])
    sdk = Path(env.get("ANDROID_HOME") or Path.home() / "Android" / "Sdk")
    versions = sorted((sdk / "ndk").glob("*"), key=lambda p: [int(x) for x in p.name.split(".") if x.isdigit()])
    if not versions:
        raise SystemExit("Building for Android needs the NDK: set ANDROID_NDK_HOME")
    return versions[-1]


def cargo_env(platform: str, env: dict | None = None) -> dict:
    """Environment for `cargo build --target <triple(platform)>`."""
    env = dict(os.environ if env is None else env)
    target = triple(platform)
    if platform.startswith("android-"):
        system = {"Linux": "linux-x86_64", "Darwin": "darwin-x86_64", "Windows": "windows-x86_64"}[host.system()]
        toolchain = android_ndk(env) / "toolchains" / "llvm" / "prebuilt" / system / "bin"
        clang = toolchain / f"{target}{ANDROID_API_LEVEL}-clang"
        if not clang.exists():
            raise SystemExit(f"The Android NDK has no {clang}")
        underscored = target.replace("-", "_")
        env[f"CARGO_TARGET_{underscored.upper()}_LINKER"] = str(clang)
        env[f"CC_{underscored}"] = str(clang)
        env[f"AR_{underscored}"] = str(toolchain / "llvm-ar")
    if platform.startswith("ios-"):
        env.setdefault("IPHONEOS_DEPLOYMENT_TARGET", IOS_DEPLOYMENT_TARGET)
    return env


def copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def pack_asset(pack: str, version: str, platform: str) -> str:
    """A pack library's file name in the pack's GitHub release."""
    return f"morflow_pack_{pack}-{version}-{platform}.{extension(platform)}"


def capi_asset(platform: str) -> str:
    """The C interface library's file name in the engine's GitHub release."""
    return f"libmorflow_capi-{platform}.{extension(platform)}"
