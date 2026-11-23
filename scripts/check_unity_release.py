#!/usr/bin/env python3
"""Check that the Unity package is ready for a unity-v<version> tag.

    python3 scripts/check_unity_release.py --tag unity-v0.1.0

Projects install the package from the tag's Git URL
(…/Morflow.git?path=bindings/unity#unity-v<version>), so the tag must name
the version in bindings/unity/package.json and Morflow.PackageVersion, and
CHANGELOG.md must describe it. Morflow.EngineVersion must be the engine in
pipeline/Cargo.toml, and the engine's GitHub release v<version> must hold the
C interface library for every platform, since the package downloads it from
there.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dart_release import release_assets  # noqa: E402
from detect_engine_version import parse_cargo_version  # noqa: E402
import native_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "bindings" / "unity"


def unity_versions(root: Path) -> dict:
    package = root / "bindings" / "unity"
    source = (package / "Runtime" / "Morflow.cs").read_text()
    constant = lambda name: (m.group(1) if (m := re.search(rf'{name} = "([^"]+)"', source)) else None)  # noqa: E731
    return {
        "package": json.loads((package / "package.json").read_text()).get("version"),
        "package_constant": constant("PackageVersion"),
        "engine_constant": constant("EngineVersion"),
        "engine": parse_cargo_version(root / "pipeline" / "Cargo.toml"),
        "changelog": re.findall(r"^## (\S+)", (package / "CHANGELOG.md").read_text(), re.M),
    }


def problems(tag: str, versions: dict, assets: set | None) -> list[str]:
    match = re.fullmatch(r"unity-v(\d+\.\d+\.\d+)", tag)
    if not match:
        return [f"{tag} is not a unity-v<major>.<minor>.<patch> tag"]
    version = match.group(1)
    found = []
    if versions["package"] != version:
        found.append(f"bindings/unity/package.json has {versions['package']}")
    if versions["package_constant"] != version:
        found.append(f"Morflow.PackageVersion is {versions['package_constant']}")
    if version not in versions["changelog"]:
        found.append("CHANGELOG.md has no entry for it")
    if found:
        return [f"The tag is {tag}, but " + "; ".join(found)]
    engine = versions["engine"]
    if versions["engine_constant"] != engine:
        return [f"Morflow.EngineVersion is {versions['engine_constant']}, but pipeline/Cargo.toml is engine {engine}"]
    if assets is None:
        return [f"The engine release v{engine} does not exist yet; tag the package after it is released"]
    missing = [a for a in map(native_targets.capi_asset, native_targets.PLATFORMS) if a not in assets]
    if missing:
        return [f"The engine release v{engine} has no {', '.join(missing)} yet"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="The tag, such as unity-v0.1.0")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow"))
    args = parser.parse_args()
    versions = unity_versions(ROOT)
    assets = release_assets(args.repo, f"v{versions['engine']}", os.environ.get("GITHUB_TOKEN"))
    found = problems(args.tag, versions, assets)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if not found:
        print(f"{args.tag} is ready: package {versions['package']}, engine {versions['engine']}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
