#!/usr/bin/env python3
"""Lay out locally built native libraries like Morflow's GitHub releases.

    python3 scripts/stage_release.py --platform android-x86_64 --out target/staged-releases

builds the C interface library and every pack library for each --platform
with the same scripts the release workflows use, under
<out>/<tag>/<release asset> with a checksums.txt per release. Point the Dart
build hook at it to test apps before anything is published:

    hooks:
      user_defines:
        morflow:
          release_url: path/to/target/staged-releases
"""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_capi  # noqa: E402
import build_pack  # noqa: E402
from detect_changed_actions import bundleable_packs, discover_all_actions  # noqa: E402
from detect_engine_version import parse_cargo_version  # noqa: E402
import native_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def write_checksums(release: Path) -> None:
    lines = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n"
        for f in sorted(release.iterdir()) if f.is_file() and f.name != "checksums.txt"
    ]
    (release / "checksums.txt").write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", action="append", required=True, choices=list(native_targets.PLATFORMS))
    parser.add_argument("--pack", action="append", help="Pack to build (default: every pack without plugins)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    engine = args.out / f"v{parse_cargo_version(ROOT / 'pipeline' / 'Cargo.toml')}"
    packs = bundleable_packs(discover_all_actions(ROOT))
    selected = args.pack or sorted(packs)
    releases = {engine}
    for platform in args.platform:
        build_capi.build(platform, engine, release_asset=True)
        for pack in selected:
            release = args.out / "action_packs" / pack / f"v{packs[pack]}"
            build_pack.build(pack, platform, release, release_asset=True)
            releases.add(release)
    for release in releases:
        write_checksums(release)
        print(f"{release}: {len(list(release.iterdir())) - 1} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
