#!/usr/bin/env python3
"""
Morflow Engine Version Detection Script
Reads the engine version from pipeline/Cargo.toml and builds and releases it
only when GitHub has no release v<version> yet, the same rule action packs and
plugins follow. A released version is never rebuilt or republished: changing
the engine needs a version bump.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
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


def release_exists(repo: str, tag: str, token: str = None) -> bool:
    """Whether the release exists. Any error other than 404 stops the run, so an
    unreachable API never reads as "unreleased"."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Morflow-CI"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            json.loads(response.read().decode("utf-8"))
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def main():
    parser = argparse.ArgumentParser(description="Detect whether the Morflow engine version needs releasing.")
    parser.add_argument("--repo", default="", help="GitHub owner/repo (e.g. JiraPit/Morflow)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    version = parse_cargo_version(repo_root / "pipeline" / "Cargo.toml")
    tag_name = f"v{version}"
    released = release_exists(repo, tag_name, token)
    should_build = not released

    print(f"Engine Version: {version}")
    print(f"Release Tag: {tag_name}")
    print(f"Already Released: {released}")
    print(f"Should Build & Release: {should_build}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")
            f.write(f"tag_name={tag_name}\n")
            f.write(f"should_build={'true' if should_build else 'false'}\n")


if __name__ == "__main__":
    sys.exit(main())
