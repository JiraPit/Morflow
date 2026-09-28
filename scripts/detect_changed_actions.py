#!/usr/bin/env python3
"""
Morflow Action Pack Change Detection Script
Discovers action packages in actions/*/*, inspects version changes via Cargo.toml/git,
and outputs dynamic build matrix JSON for GitHub Actions.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


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
        # Check if HEAD~1 exists
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
            # First commit or shallow clone
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


def main():
    parser = argparse.ArgumentParser(description="Detect changed Morflow actions for CI build.")
    parser.add_argument("--event", default="push", help="GitHub event name (push, workflow_dispatch, etc.)")
    parser.add_argument("--pack", default="all", help="Target action pack or 'all'")
    parser.add_argument("--action", default="all", help="Target action name or 'all'")
    parser.add_argument("--force", default="false", help="Force build regardless of git diff")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    all_actions = discover_all_actions(repo_root)
    force_build = args.force.lower() in ("true", "1", "yes")

    selected_actions = []

    if args.event == "workflow_dispatch" or force_build:
        for act in all_actions:
            if args.pack != "all" and act["pack"] != args.pack:
                continue
            if args.action != "all" and act["name"] != args.action:
                continue
            selected_actions.append(act)
    else:
        # Push event: check changed files
        changed_files = get_git_changed_files()
        if not changed_files:
            # If unable to determine diff, build all discovered actions as fallback
            selected_actions = all_actions
        else:
            for act in all_actions:
                act_prefix = act["path"] + "/"
                # Check if any changed file belongs to this action or core_types
                has_changed = any(f.startswith(act_prefix) or f.startswith("core_types/") for f in changed_files)
                if has_changed:
                    selected_actions.append(act)

    # Group packs
    packs_dict = {}
    for act in selected_actions:
        p = act["pack"]
        v = act["version"]
        if p not in packs_dict:
            packs_dict[p] = v

    packs_list = [{"name": k, "version": v} for k, v in packs_dict.items()]
    has_actions = len(selected_actions) > 0

    matrix_json = json.dumps(selected_actions)
    packs_json = json.dumps(packs_list)

    print(f"Discovered total actions: {len(all_actions)}")
    print(f"Selected actions to build: {len(selected_actions)}")
    for act in selected_actions:
        print(f"  - [{act['pack']}] {act['name']} v{act['version']} ({act['path']})")
    print(f"Action packs targeted: {[p['name'] for p in packs_list]}")

    # Set GitHub Actions step output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"packs={packs_json}\n")
            f.write(f"has_actions={'true' if has_actions else 'false'}\n")
    else:
        print(f"\nGitHub Output:\nmatrix={matrix_json}\npacks={packs_json}\nhas_actions={'true' if has_actions else 'false'}")


if __name__ == "__main__":
    main()
