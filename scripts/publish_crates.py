#!/usr/bin/env python3
"""Publish Morflow's three Rust crates in Cargo dependency order."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ("core_types/Cargo.toml", "parser/Cargo.toml", "pipeline/Cargo.toml")


def unpublished_packages():
    pending = []
    for manifest in MANIFESTS:
        metadata = tomllib.loads((ROOT / manifest).read_text())["package"]
        name, version = metadata["name"], metadata["version"]
        request = Request(
            f"https://crates.io/api/v1/crates/{name}/{version}",
            headers={
                "User-Agent": "Morflow release automation (https://github.com/JiraPit/Morflow)"
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                published = json.load(response)["version"]
            if published["num"] != version or published["crate"] != name:
                raise RuntimeError(f"Unexpected registry response for {name}@{version}")
            print(f"Already published: {name}@{version}")
        except HTTPError as error:
            if error.code != 404:
                raise
            pending.append(name)
            print(f"Ready to publish: {name}@{version}")
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", action="store_true", help="Upload the unpublished versions"
    )
    args = parser.parse_args()
    pending = unpublished_packages()
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"pending={'true' if pending else 'false'}\n")
    if args.publish and pending:
        command = ["cargo", "publish", "--locked"]
        for name in pending:
            command.extend(["-p", name])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
