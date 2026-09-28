#!/usr/bin/env python3
"""
Morflow Python CLI
Provides 'prep', 'clean', 'spec', 'search', and 'list' subcommands when installed via pip.
"""

import argparse
import os
import platform
import re
import shutil
import sys
import urllib.request
from pathlib import Path

KNOWN_ACTIONS = [
    ("base", "identity"),
    ("base", "to_tensor"),
    ("audio_essentials", "to_audio"),
    ("audio_essentials", "to_pcm"),
    ("audio_essentials", "to_wav"),
    ("audio_essentials", "gain"),
    ("audio_essentials", "normalize"),
    ("audio_essentials", "biquad_filter"),
    ("audio_essentials", "compressor"),
    ("audio_essentials", "limiter"),
    ("audio_essentials", "noise_gate"),
    ("audio_essentials", "stereo_widen"),
    ("audio_essentials", "resample"),
    ("audio_essentials", "stft"),
    ("audio_essentials", "delay"),
    ("image_essentials", "to_image"),
    ("image_essentials", "resize"),
    ("image_essentials", "crop"),
    ("image_essentials", "pad"),
    ("image_essentials", "color_adjust"),
    ("image_essentials", "gaussian_blur"),
    ("image_essentials", "edge_detect"),
    ("image_essentials", "sharpen"),
    ("image_essentials", "threshold"),
    ("image_essentials", "rotate"),
    ("image_essentials", "flip"),
    ("image_essentials", "blend"),
    ("image_essentials", "morphology"),
]


def get_host_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("aarch64", "arm64"):
            return "linux-aarch64", "so"
        return "linux-x86_64", "so"
    elif system == "darwin":
        if machine in ("aarch64", "arm64"):
            return "darwin-arm64", "dylib"
        return "darwin-x86_64", "dylib"
    elif system == "windows":
        return "windows-x86_64", "dll"
    return "linux-x86_64", "so"


def resolve_action_cache_dir(custom_path=None):
    if custom_path:
        return Path(custom_path)
    env_path = os.environ.get("MORFLOW_ACTIONS_PATH")
    if env_path and env_path.strip():
        return Path(env_path.strip())
    return Path.home() / ".morflow" / "actions"


def normalize_pack_name(pack: str) -> str:
    pack = pack.strip().lower()
    if pack in ("audio_essential", "audio_essentials"):
        return "audio_essentials"
    elif pack in ("image_essential", "image_essentials"):
        return "image_essentials"
    elif pack == "base":
        return "base"
    return pack


def parse_full_action_path(path_str: str):
    clean = path_str.replace("::", ".").strip()
    parts = [p.strip() for p in clean.split(".") if p.strip()]

    if len(parts) >= 3:
        pack = normalize_pack_name(parts[0])
        version = parts[1]
        action = parts[2]
        return pack, version, action
    elif len(parts) == 2:
        pack = normalize_pack_name(parts[0])
        action = parts[1]
        return pack, "latest", action
    elif len(parts) == 1:
        action = parts[0]
        for pack, act in KNOWN_ACTIONS:
            if act.lower() == action.lower():
                return pack, "latest", act
        return "base", "latest", action
    return "base", "latest", path_str


def extract_actions_from_morf(source: str):
    """Extracts imports and action calls from .morf source."""
    imports = {}
    actions = []

    for line in source.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check for import lines: import audio_essentials as audio or use audio_essentials::*
        import_match = re.match(r"^(?:import|use)\s+([a-zA-Z0-9_]+)(?:\s+as\s+([a-zA-Z0-9_]+))?", line)
        if import_match:
            pack = import_match.group(1)
            alias = import_match.group(2) or pack
            imports[alias] = pack

        # Match action calls in chain: >> action_name(...) or >> action_name
        for action_match in re.finditer(r">>\s*([a-zA-Z0-9_.:]+)", line):
            act_full = action_match.group(1).split("(")[0].strip()
            if act_full in ("emit", "resurface"):
                continue
            if act_full not in actions:
                actions.append(act_full)

    # Resolve actions to (pack, action_name)
    resolved = []
    for act in actions:
        if "::" in act:
            pack, name = act.split("::", 1)
            resolved.append((normalize_pack_name(pack), name))
        elif "." in act:
            pack, name = act.split(".", 1)
            resolved.append((normalize_pack_name(pack), name))
        else:
            found = False
            for k_pack, k_act in KNOWN_ACTIONS:
                if k_act == act:
                    resolved.append((k_pack, act))
                    found = True
                    break
            if not found:
                resolved.append(("base", act))

    return resolved


def cmd_prep(args):
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: Pipeline file '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    source = file_path.read_text(encoding="utf-8")
    actions = extract_actions_from_morf(source)

    if not actions:
        print(f"No action calls found in '{file_path}'. Nothing to prepare.")
        return

    platform_name, ext = get_host_platform()
    cache_dir = resolve_action_cache_dir(args.path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" Morflow Action Pre-Downloader (Python CLI)")
    print(f" Pipeline: {file_path}")
    print(f" Host Platform: {platform_name} (.{ext})")
    print(f" Cache Directory: {cache_dir}")
    print(f" Repository: {args.repo}")
    print("==================================================")

    prepared = 0
    cached = 0

    for pack, action_name in actions:
        pack_dir = cache_dir / pack
        pack_dir.mkdir(parents=True, exist_ok=True)

        target_file_pack = pack_dir / f"{action_name}_action.{ext}"
        target_file_root = cache_dir / f"{action_name}_action.{ext}"

        if not args.force and (target_file_pack.exists() or target_file_root.exists()):
            print(f"  [✓ Cached] [{pack}] {action_name}")
            cached += 1
            continue

        print(f"  [↓ Downloading] [{pack}] {action_name} v{args.action_version}...")
        binary_filename = f"{action_name}_action-{args.action_version}-{platform_name}.{ext}"

        urls = [
            f"https://github.com/{args.repo}/releases/download/action_packs%2F{pack}%2Fv{args.action_version}/{binary_filename}",
            f"https://github.com/{args.repo}/releases/download/action_packs/{pack}/v{args.action_version}/{binary_filename}",
        ]

        downloaded = False
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Morflow-CLI/0.1.0"})
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()
                    target_file_pack.write_bytes(data)
                    target_file_root.write_bytes(data)
                    print(f"    ✓ Successfully cached to {target_file_pack}")
                    downloaded = True
                    prepared += 1
                    break
            except Exception:
                continue

        if not downloaded:
            # Check local repository builds
            local_candidates = [
                Path(f"target/release/actions/{pack}/{action_name}_action.{ext}"),
                Path(f"target/release/actions/{action_name}_action.{ext}"),
                Path(f"actions/{pack}/{action_name}/target/release/lib{action_name}.{ext}"),
            ]
            copied = False
            for cand in local_candidates:
                if cand.exists():
                    shutil.copy2(cand, target_file_pack)
                    shutil.copy2(cand, target_file_root)
                    print(f"    ✓ Copied local build artifact from {cand}")
                    copied = True
                    prepared += 1
                    break

            if not copied:
                print(
                    f"    ✗ Warning: Could not download remote binary or find local artifact for [{pack}] {action_name}.",
                    file=sys.stderr,
                )

    print(f"\nSummary: {prepared} action(s) prepared, {cached} already cached.")
    print(f"All actions ready in {cache_dir} for offline runtime execution.\n")


def cmd_clean(args):
    cache_dir = resolve_action_cache_dir(args.path)
    print("==================================================")
    print(" Morflow Action Cache Cleaner (Python CLI)")
    print(f" Target Directory: {cache_dir}")
    print("==================================================")

    if not cache_dir.exists():
        print("Cache directory does not exist. Nothing to clean.")
        return

    deleted_count = 0
    for item in cache_dir.iterdir():
        if item.is_file():
            item.unlink()
            deleted_count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            deleted_count += 1

    print(f"✓ Cleaned {deleted_count} items from {cache_dir}")


def cmd_spec(args):
    pack, _version, action_name = parse_full_action_path(args.action)

    # 1. Check local files in repository / development environment
    local_candidates = [
        Path(f"actions/{pack}/{action_name}/SPEC.md"),
        Path(f"../actions/{pack}/{action_name}/SPEC.md"),
        resolve_action_cache_dir(None) / pack / action_name / "SPEC.md",
    ]

    for cand in local_candidates:
        if cand.exists():
            try:
                print(cand.read_text(encoding="utf-8"), end="")
                return
            except Exception:
                pass

    # 2. Fetch raw SPEC.md from GitHub
    url = f"https://raw.githubusercontent.com/{args.repo}/main/actions/{pack}/{action_name}/SPEC.md"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Morflow-CLI/0.1.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            print(content, end="")
    except Exception:
        print(
            f"Error: SPEC.md not found for action '{pack}.latest.{action_name}' (checked local paths and {url}).",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_search(args):
    q = args.query.lower()
    matches = []
    for pack, act in KNOWN_ACTIONS:
        act_lower = act.lower()
        pos = act_lower.find(q)
        if pos != -1:
            matches.append((pos, len(act), act, pack))

    # Rank by: 1) earlier substring position, 2) shorter action length, 3) alphabetical
    matches.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    top_matches = matches[: args.limit]

    if not top_matches:
        print(f"No matching actions found for query '{args.query}'.")
    else:
        for _pos, _len, act, pack in top_matches:
            print(f"{pack}.latest.{act}")


def cmd_list(args):
    cache_dir = resolve_action_cache_dir(args.path)
    _platform, ext = get_host_platform()
    suffix = f"_action.{ext}"
    found_actions = set()

    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir():
                pack_name = item.name
                for sub_item in item.iterdir():
                    if sub_item.is_file() and sub_item.name.endswith(suffix):
                        act_name = sub_item.name[: -len(suffix)]
                        found_actions.add(f"{pack_name}.latest.{act_name}")
            elif item.is_file() and item.name.endswith(suffix):
                act_name = item.name[: -len(suffix)]
                pack_name = "base"
                for k_pack, k_act in KNOWN_ACTIONS:
                    if k_act == act_name:
                        pack_name = k_pack
                        break
                found_actions.add(f"{pack_name}.latest.{act_name}")

    dev_target = Path("target/release/actions")
    if dev_target.exists():
        for item in dev_target.iterdir():
            if item.is_dir():
                pack_name = item.name
                for sub_item in item.iterdir():
                    if sub_item.is_file() and sub_item.name.endswith(suffix):
                        act_name = sub_item.name[: -len(suffix)]
                        found_actions.add(f"{pack_name}.latest.{act_name}")

    sorted_actions = sorted(list(found_actions))
    if not sorted_actions:
        print(f"No installed actions found in {cache_dir}.")
        print("Run 'morflow prep <pipeline.morf>' to download required actions.")
    else:
        for act in sorted_actions:
            print(act)


def main():
    parser = argparse.ArgumentParser(
        prog="morflow",
        description="Morflow - High-performance modular dataflow pipeline engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prep command
    prep_parser = subparsers.add_parser(
        "prep",
        help="Pre-downloads all actions required by a .morf pipeline ahead of time for offline execution",
    )
    prep_parser.add_argument("file", help="Path to the .morf pipeline definition file")
    prep_parser.add_argument("--path", help="Custom action cache directory")
    prep_parser.add_argument("--repo", default="JiraPit/Morflow", help="GitHub repository to download action binaries from")
    prep_parser.add_argument("--action-version", default="0.1.0", help="Action Pack version tag")
    prep_parser.add_argument("--force", action="store_true", help="Force re-download even if action is already cached")

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Cleans and removes all cached action binaries from the action path",
    )
    clean_parser.add_argument("--path", help="Custom action cache directory to clean")

    # Spec command
    spec_parser = subparsers.add_parser(
        "spec",
        help="Views the raw SPEC.md documentation for a specified action",
    )
    spec_parser.add_argument("action", help="Full action path (e.g. image_essential.latest.color_adjust)")
    spec_parser.add_argument("--repo", default="JiraPit/Morflow", help="GitHub repository to fetch spec from")

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Performs fuzzy search for actions by name and returns top matching full action paths",
    )
    search_parser.add_argument("query", help="Search query (e.g. color, blur, resample)")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="Lists all action paths installed locally in the action cache",
    )
    list_parser.add_argument("--path", help="Custom action cache directory to inspect")

    args = parser.parse_args()
    if args.command == "prep":
        cmd_prep(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "spec":
        cmd_spec(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
