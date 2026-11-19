#!/usr/bin/env python3
"""Check that the Dart package can be published for a dart-v<version> tag.

    python3 scripts/check_dart_release.py --tag dart-v0.3.3

The tag must name the version in bindings/dart/pubspec.yaml: the engine
version, or the engine version plus a build number (dart-v0.3.3+1) for a
release that changes only the Dart package. The engine version in the tag,
lib/src/version.dart and pipeline/Cargo.toml must agree, and the engine's
GitHub release v<version> must hold the C interface library for every
platform, since the package's build hook downloads it from there. Publishing before the
engine release finishes would ship a package no app can build.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_engine_version import parse_cargo_version  # noqa: E402
import native_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def dart_versions(root: Path) -> dict:
    pubspec = re.search(r"^version:\s*(\S+)\s*$", (root / "bindings/dart/pubspec.yaml").read_text(), re.M)
    constant = re.search(r"engineVersion = '([^']+)'", (root / "bindings/dart/lib/src/version.dart").read_text())
    return {
        "bindings/dart/pubspec.yaml": pubspec.group(1) if pubspec else None,
        "bindings/dart/lib/src/version.dart": constant.group(1) if constant else None,
        "pipeline/Cargo.toml": parse_cargo_version(root / "pipeline" / "Cargo.toml"),
    }


def problems(tag: str, versions: dict, release_assets: set | None) -> list[str]:
    match = re.fullmatch(r"dart-v((\d+\.\d+\.\d+)(\+\d+)?)", tag)
    if not match:
        return [f"{tag} is not a dart-v<major>.<minor>.<patch>[+<build>] tag"]
    package, version = match.group(1), match.group(2)
    expected = {file: version for file in versions}
    expected["bindings/dart/pubspec.yaml"] = package
    found = [f"{file} has {found}" for file, found in versions.items() if found != expected[file]]
    if found:
        return [f"The tag is {tag}, but " + "; ".join(found)]
    if release_assets is None:
        return [f"The engine release v{version} does not exist yet; publish after it is released"]
    missing = [asset for asset in map(native_targets.capi_asset, native_targets.PLATFORMS) if asset not in release_assets]
    if missing:
        return [f"The engine release v{version} has no {', '.join(missing)} yet"]
    return []


def release_assets(repo: str, tag: str, token: str | None) -> set | None:
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {asset["name"] for asset in json.load(response).get("assets", [])}
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="The pushed tag, such as dart-v0.3.3")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow"))
    args = parser.parse_args()

    versions = dart_versions(ROOT)
    version = versions["pipeline/Cargo.toml"]
    package = versions["bindings/dart/pubspec.yaml"]
    assets = release_assets(args.repo, f"v{version}", os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    found = problems(args.tag, versions, assets)
    for problem in found:
        print(f"Error: {problem}", file=sys.stderr)
    if not found:
        print(f"Ready to publish morflow {package} (engine {version}) to pub.dev")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
