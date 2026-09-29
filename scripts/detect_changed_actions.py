#!/usr/bin/env python3
"""
Morflow Action Pack Change Detection Script
Discovers action packages in actions/*/*, inspects version changes and existing
GitHub Releases, limits matrix size to avoid GitHub Actions' 256-configuration limit,
and outputs dynamic build matrix JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

TARGET_PLATFORMS = [
    ("linux-x86_64", "so"),
    ("darwin-arm64", "dylib"),
    ("windows-x86_64", "dll"),
]


def parse_cargo_toml(path: Path) -> dict:
    """Simple parser for package name and version from Cargo.toml without external dependencies."""
    name = None
    version = None
    in_package = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[package]"):
                in_package = True
                continue
            elif line.startswith("[") and in_package:
                break

            if in_package:
                if line.startswith("name"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        name = parts[1].strip().strip('"').strip("'")
                elif line.startswith("version"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        version = parts[1].strip().strip('"').strip("'")

    return {"name": name, "version": version}


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
        else:
            return []
    except Exception as e:
        print(f"Warning: Failed to get git diff: {e}", file=sys.stderr)
        return []


def discover_all_actions(repo_root: Path) -> list:
    """Discovers all actions located in actions/<pack>/<action>/Cargo.toml."""
    actions_dir = repo_root / "actions"
    actions = []

    if not actions_dir.exists():
        return actions

    for pack_dir in sorted(actions_dir.iterdir()):
        if not pack_dir.is_dir():
            continue
        pack_name = pack_dir.name
        for action_dir in sorted(pack_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            cargo_path = action_dir / "Cargo.toml"
            if cargo_path.exists():
                meta = parse_cargo_toml(cargo_path)
                if meta["name"] and meta["version"]:
                    actions.append({
                        "name": meta["name"],
                        "pack": pack_name,
                        "version": meta["version"],
                        "path": str(action_dir.relative_to(repo_root)),
                    })
    return actions


def get_release_assets(repo: str, tag: str, token: str = None) -> set:
    """Fetches the set of asset filenames for a release tag from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Morflow-CI",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {a["name"] for a in data.get("assets", [])}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Release tag doesn't exist yet
            return set()
        print(f"Warning: HTTP {e.code} while fetching assets for {tag}: {e.reason}", file=sys.stderr)
        return set()
    except Exception as e:
        print(f"Warning: Failed to fetch assets for {tag}: {e}", file=sys.stderr)
        return set()


def is_action_fully_released(act: dict, release_assets: set) -> bool:
    """Checks if all required platform binaries for this action version are present in the release."""
    act_name = act["name"]
    version = act["version"]
    for plat, ext in TARGET_PLATFORMS:
        expected = f"{act_name}_action-{version}-{plat}.{ext}"
        if expected not in release_assets:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Detect changed Morflow actions for CI build.")
    parser.add_argument("--event", default="push", help="GitHub event name (push, workflow_dispatch, etc.)")
    parser.add_argument("--pack", default="all", help="Target action pack or 'all'")
    parser.add_argument("--action", default="all", help="Target action name or 'all'")
    parser.add_argument("--force", default="false", help="Force build regardless of existing releases")
    parser.add_argument("--limit", type=int, default=50, help="Max actions per matrix batch (50 * 3 = 150 <= 256)")
    parser.add_argument("--repo", default="", help="GitHub owner/repo (e.g. JiraPit/Morflow)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "JiraPit/Morflow")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    force_build = args.force.lower() in ("true", "1", "yes")

    all_actions = discover_all_actions(repo_root)

    # 1. Filter by requested pack / action
    candidates = []
    for act in all_actions:
        if args.pack != "all" and act["pack"] != args.pack:
            continue
        if args.action != "all" and act["name"] != args.action:
            continue
        candidates.append(act)

    print(f"Total discovered actions matching filter: {len(candidates)}")

    # 2. Skip actions that are already fully released on GitHub Releases (unless forced)
    unbuilt_actions = []
    already_released_count = 0
    release_cache = {}

    for act in candidates:
        if force_build:
            unbuilt_actions.append(act)
            continue

        pack = act["pack"]
        version = act["version"]
        tag = f"action_packs/{pack}/v{version}"

        if tag not in release_cache:
            release_cache[tag] = get_release_assets(repo, tag, token)

        assets = release_cache[tag]
        if is_action_fully_released(act, assets):
            already_released_count += 1
        else:
            unbuilt_actions.append(act)

    print(f"Already released actions skipped: {already_released_count}")
    print(f"Total actions requiring build: {len(unbuilt_actions)}")

    # 3. Limit batch size to stay safely within GitHub Actions matrix limit (max 256 configurations)
    # With 3 target platforms (Linux, macOS, Windows), limit=50 produces 150 configurations.
    limit = max(1, args.limit)
    selected_actions = unbuilt_actions[:limit]
    remaining_count = len(unbuilt_actions) - len(selected_actions)
    has_more = remaining_count > 0
    has_actions = len(selected_actions) > 0

    # Group packs for selected actions
    packs_dict = {}
    for act in selected_actions:
        p = act["pack"]
        v = act["version"]
        if p not in packs_dict:
            packs_dict[p] = v

    packs_list = [{"name": k, "version": v} for k, v in packs_dict.items()]

    matrix_json = json.dumps(selected_actions)
    packs_json = json.dumps(packs_list)

    print(f"Selected for current batch: {len(selected_actions)} actions ({len(selected_actions) * len(TARGET_PLATFORMS)} matrix jobs)")
    for act in selected_actions:
        print(f"  - [{act['pack']}] {act['name']} v{act['version']}")
    print(f"Remaining for subsequent triggers: {remaining_count}")
    print(f"Has more batches: {has_more}")

    # Set GitHub Actions step outputs
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"packs={packs_json}\n")
            f.write(f"has_actions={'true' if has_actions else 'false'}\n")
            f.write(f"has_more={'true' if has_more else 'false'}\n")
            f.write(f"remaining_count={remaining_count}\n")
    else:
        print(f"\nGitHub Output:\nmatrix={matrix_json}\npacks={packs_json}\nhas_actions={'true' if has_actions else 'false'}\nhas_more={'true' if has_more else 'false'}\nremaining_count={remaining_count}")


if __name__ == "__main__":
    main()
