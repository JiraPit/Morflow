#!/usr/bin/env python3
"""
Morflow Plugin Change Detection Script
Discovers plugins in plugins/*, compares each plugin version against its GitHub
Release, and outputs a build matrix with one entry per platform binary that is
not yet published. A published binary is never rebuilt: the release rejects
changed assets, so a new binary needs a new plugin version.
"""

import argparse
import json
import os
from pathlib import Path

from detect_changed_actions import get_release_assets
from package_plugin import plugin_metadata

# Runner for each platform plugins are released for.
PLUGIN_TARGETS = {
    "linux-x86_64": {"os": "ubuntu-24.04", "ext": "so"},
    "linux-aarch64": {"os": "ubuntu-24.04-arm", "ext": "so"},
    "darwin-arm64": {"os": "macos-latest", "ext": "dylib"},
    "windows-x86_64": {"os": "windows-latest", "ext": "dll"},
}


def discover_plugins(repo_root: Path) -> list:
    return [plugin_metadata(manifest) for manifest in sorted((repo_root / "plugins").glob("*/Cargo.toml"))]


def asset_name(plugin: dict, platform: str) -> str:
    ext = PLUGIN_TARGETS[platform]["ext"]
    return f"{plugin['name']}_plugin-{plugin['version']}-{platform}.{ext}"


def missing_platforms(plugin: dict, release_assets: set) -> list:
    return [platform for platform in PLUGIN_TARGETS if asset_name(plugin, platform) not in release_assets]


def build_job(plugin: dict, platform: str) -> dict:
    return {"plugin": plugin, "target": {"platform": platform, **PLUGIN_TARGETS[platform]}}


def main():
    parser = argparse.ArgumentParser(description="Detect unpublished Morflow plugin binaries for CI build.")
    parser.add_argument("--repo", default="", help="GitHub owner/repo (e.g. JiraPit/Morflow)")
    parser.add_argument("--force", default="false", help="Build every platform regardless of existing releases")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    force_build = args.force.lower() in ("true", "1", "yes")

    jobs, pending = [], []
    for plugin in discover_plugins(repo_root):
        tag = f"plugins/{plugin['name']}/v{plugin['version']}"
        missing = list(PLUGIN_TARGETS) if force_build else missing_platforms(plugin, get_release_assets(repo, tag, token))
        if missing:
            pending.append(plugin)
            jobs.extend(build_job(plugin, platform) for platform in missing)
        print(f"  - {plugin['name']} v{plugin['version']}: {', '.join(missing) if missing else 'already released'}")

    print(f"Plugin build jobs: {len(jobs)}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix={json.dumps(jobs)}\n")
            f.write(f"plugins={json.dumps(pending)}\n")
            f.write(f"has_builds={'true' if jobs else 'false'}\n")


if __name__ == "__main__":
    main()
