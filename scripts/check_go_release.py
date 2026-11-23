#!/usr/bin/env python3
"""Check that the Go package is ready for a bindings/go/v<version> tag.

    python3 scripts/check_go_release.py --tag bindings/go/v0.3.5

Go programs fetch the package from that tag (the Go module proxy picks it up
by itself), so bindings/go/CHANGELOG.md must describe the version. The
package's EngineVersion must be the engine in pipeline/Cargo.toml, and the
engine's GitHub release v<version> must hold the C interface library for
every desktop platform, since the package downloads it from there on first
use.
"""
import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dart_release import release_assets  # noqa: E402
from detect_engine_version import parse_cargo_version  # noqa: E402
import native_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# The platforms Go programs run on: every desktop platform, and Intel Macs.
GO_PLATFORMS = native_targets.DESKTOP_PLATFORMS + ["darwin-x86_64"]


def go_versions(root: Path) -> dict:
    package = root / "bindings" / "go"
    constant = re.search(r'EngineVersion = "([^"]+)"', (package / "version.go").read_text())
    return {
        "engine_constant": constant.group(1) if constant else None,
        "engine": parse_cargo_version(root / "pipeline" / "Cargo.toml"),
        "changelog": re.findall(r"^## (\S+)", (package / "CHANGELOG.md").read_text(), re.M),
    }


def problems(tag: str, versions: dict, assets: set | None) -> list[str]:
    match = re.fullmatch(r"bindings/go/v(\d+\.\d+\.\d+)", tag)
    if not match:
        return [f"{tag} is not a bindings/go/v<major>.<minor>.<patch> tag"]
    version = match.group(1)
    if version not in versions["changelog"]:
        return [f"The tag is {tag}, but bindings/go/CHANGELOG.md has no entry for {version}"]
    engine = versions["engine"]
    if versions["engine_constant"] != engine:
        return [f"EngineVersion in bindings/go/version.go is {versions['engine_constant']}, but pipeline/Cargo.toml is engine {engine}"]
    if assets is None:
        return [f"The engine release v{engine} does not exist yet; tag the Go package after it is released"]
    missing = [a for a in map(native_targets.capi_asset, GO_PLATFORMS) if a not in assets]
    if missing:
        return [f"The engine release v{engine} has no {', '.join(missing)} yet"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="The tag, such as bindings/go/v0.3.5")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow"))
    args = parser.parse_args()
    versions = go_versions(ROOT)
    assets = release_assets(args.repo, f"v{versions['engine']}", os.environ.get("GITHUB_TOKEN"))
    found = problems(args.tag, versions, assets)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if not found:
        print(f"{args.tag} is ready: engine {versions['engine']}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
