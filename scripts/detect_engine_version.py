#!/usr/bin/env python3
"""
Morflow Engine Version Detection Script
Reads the engine version from pipeline/Cargo.toml and decides what CI does:

- A pushed tag v<version> releases that version. The tag must match the
  version in pipeline/Cargo.toml.
- A branch push builds every platform as a check when engine files changed in
  the pushed commits, and never publishes.
- A manual run builds every platform and never publishes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_cargo_version(cargo_toml_path: Path) -> str:
    """Extracts package version from a Cargo.toml file."""
    if not cargo_toml_path.exists():
        return "0.1.0"

    in_package = False
    with open(cargo_toml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[package]"):
                in_package = True
                continue
            elif line.startswith("[") and in_package:
                break

            if in_package and line.startswith("version"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return "0.1.0"


ENGINE_PREFIXES = (
    "pipeline/",
    "core_types/",
    "parser/",
    "bindings/",
    "Cargo.toml",
    "Cargo.lock",
    ".github/workflows/build-engine.yml",
    "scripts/detect_engine_version.py",
    "scripts/package-node.mjs",
    "scripts/publish-node.mjs",
    "scripts/publish_crates.py",
)


def get_git_changed_files() -> list:
    """Changed files across the whole push, or the latest commit for other runs."""
    push_base = os.environ.get("GITHUB_BASE_SHA", "")
    base = push_base if push_base and set(push_base) != {"0"} else "HEAD~1"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", base],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            diff_res = subprocess.run(
                ["git", "diff", "--name-only", base, "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return [f.strip() for f in diff_res.stdout.splitlines() if f.strip()]
        return []
    except Exception as e:
        print(f"Warning: Failed to get git diff: {e}", file=sys.stderr)
        return []


def touches_engine(changed_files: list) -> bool:
    return any(
        f == prefix or f.startswith(prefix)
        for f in changed_files
        for prefix in ENGINE_PREFIXES
    )


def main():
    parser = argparse.ArgumentParser(description="Detect Morflow engine version for CI release.")
    parser.add_argument("--event", default="push", help="GitHub event name")
    parser.add_argument("--ref", default="", help="Full git ref that triggered the run (GITHUB_REF)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    pipeline_cargo = repo_root / "pipeline" / "Cargo.toml"
    version = parse_cargo_version(pipeline_cargo)
    tag_name = f"v{version}"

    should_release = args.ref.startswith("refs/tags/")
    if should_release:
        pushed_tag = args.ref.removeprefix("refs/tags/")
        if pushed_tag != tag_name:
            print(
                f"Error: tag {pushed_tag} does not match engine version {version} in "
                f"pipeline/Cargo.toml. Tag the commit that sets the version as {tag_name}.",
                file=sys.stderr,
            )
            sys.exit(1)
        should_build = True
    elif args.event == "push":
        changed_files = get_git_changed_files()
        should_build = not changed_files or touches_engine(changed_files)
    else:
        should_build = True

    print(f"Engine Version: {version}")
    print(f"Release Tag: {tag_name}")
    print(f"Should Build: {should_build}")
    print(f"Should Release: {should_release}")

    # Set GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")
            f.write(f"tag_name={tag_name}\n")
            f.write(f"should_build={'true' if should_build else 'false'}\n")
            f.write(f"should_release={'true' if should_release else 'false'}\n")


if __name__ == "__main__":
    main()
