#!/usr/bin/env python3
"""Preserve earlier assets/checksums when a pack release is built in batches."""
from pathlib import Path
import shutil
import sys


def merge(previous, current):
    for asset in previous.iterdir():
        if not asset.is_file() or asset.name == 'checksums.txt':
            continue
        target = current / asset.name
        if target.exists() and target.read_bytes() != asset.read_bytes():
            raise ValueError(f'Release asset {asset.name} changed; publish a new version')
        if not target.exists():
            shutil.copyfile(asset, target)


if __name__ == '__main__':
    merge(Path(sys.argv[1]), Path(sys.argv[2]))
