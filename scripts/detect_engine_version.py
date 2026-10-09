#!/usr/bin/env python3
"""
Morflow Engine Version Detection Script
Reads engine version from pipeline/Cargo.toml, checks git diff / release existence,
and outputs parameters for GitHub Actions.
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


def get_git_changed_files() -> list:
    """Retrieves changed files in the latest commit using git diff."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD~1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            diff_res = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
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


def main():
    parser = argparse.ArgumentParser(description="Detect Morflow engine version for CI release.")
    parser.add_argument("--event", default="push", help="GitHub event name")
    parser.add_argument("--force", default="false", help="Force build regardless of diff")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    pipeline_cargo = repo_root / "pipeline" / "Cargo.toml"
    version = parse_cargo_version(pipeline_cargo)
    tag_name = f"v{version}"

    force_build = args.force.lower() in ("true", "1", "yes")

    if args.event == "workflow_dispatch" or force_build:
        should_build = True
    else:
        # Push event: check if any engine/binding file changed
        changed_files = get_git_changed_files()
        if not changed_files:
            should_build = True
        else:
            engine_prefixes = (
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
            should_build = any(
                any(f.startswith(prefix) or f == prefix for prefix in engine_prefixes)
                for f in changed_files
            )

    print(f"Engine Version: {version}")
    print(f"Release Tag: {tag_name}")
    print(f"Should Build & Release: {should_build}")

    # Set GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")
            f.write(f"tag_name={tag_name}\n")
            f.write(f"should_build={'true' if should_build else 'false'}\n")
    else:
        print(f"\nGitHub Output:\nversion={version}\ntag_name={tag_name}\nshould_build={'true' if should_build else 'false'}")


if __name__ == "__main__":
    main()
