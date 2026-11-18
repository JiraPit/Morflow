#!/usr/bin/env python3
"""Build the C interface library (bindings/capi) that the Dart package loads.

    python3 scripts/build_capi.py --platform android-aarch64 --out dist/
    python3 scripts/build_capi.py --platform ios-arm64 --release-asset --out dist/

`--out` receives the library as libmorflow_capi.<ext>, or with
`--release-asset` under its name in the engine's GitHub release
(libmorflow_capi-<platform>.<ext>).
"""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def build(platform: str, out: Path, release_asset: bool = False) -> Path:
    target = native_targets.triple(platform)
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "morflow-capi", "--target", target],
                   cwd=ROOT, env=native_targets.cargo_env(platform), check=True)
    built = ROOT / "target" / target / "release" / native_targets.library_file("morflow_capi", platform)
    name = native_targets.capi_asset(platform) if release_asset else built.name
    copied = native_targets.copy(built, out / name)
    print(f"{copied} ({copied.stat().st_size} bytes)")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", required=True, choices=list(native_targets.PLATFORMS))
    parser.add_argument("--out", type=Path, required=True, help="Directory to copy the library into")
    parser.add_argument("--release-asset", action="store_true", help="Name the library as its release asset")
    args = parser.parse_args()
    build(args.platform, args.out, args.release_asset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
