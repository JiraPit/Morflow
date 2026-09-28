#!/usr/bin/env python3
"""
Morflow Python CLI
Provides 'prep' and 'clean' subcommands when installed via pip.
"""

import argparse
import os
import platform
import re
import shutil
import sys
import urllib.request
from pathlib import Path


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
            resolved.append((pack, name))
        elif "." in act:
            pack, name = act.split(".", 1)
            resolved.append((pack, name))
        else:
            # Check default pack guesses
            audio_actions = {
                "gain", "normalize", "biquad_filter", "compressor", "limiter",
                "noise_gate", "stereo_widen", "resample", "stft", "delay",
                "to_audio", "to_pcm", "to_wav"
            }
            image_actions = {
                "blend", "color_adjust", "crop", "edge_detect", "flip",
                "gaussian_blur", "morphology", "pad", "resize", "rotate",
                "sharpen", "threshold", "to_image"
            }
            if act in audio_actions:
                resolved.append(("audio_essentials", act))
            elif act in image_actions:
                resolved.append(("image_essentials", act))
            elif act in ("identity", "to_tensor"):
                resolved.append(("base", act))
            else:
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

    args = parser.parse_args()
    if args.command == "prep":
        cmd_prep(args)
    elif args.command == "clean":
        cmd_clean(args)


if __name__ == "__main__":
    main()
