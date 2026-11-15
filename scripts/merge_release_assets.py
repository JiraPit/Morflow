#!/usr/bin/env python3
"""Preserve earlier assets/checksums when a pack release is built in batches."""
import json
from pathlib import Path
import shutil
import sys


def same_metadata(previous, current):
    """Metadata files match when their JSON matches. Earlier releases may hold
    a copy written with CRLF line endings by a Windows build job."""
    try:
        return json.loads(previous.read_bytes()) == json.loads(current.read_bytes())
    except ValueError:
        return False


def merge(previous, current):
    for asset in previous.iterdir():
        if not asset.is_file() or asset.name == 'checksums.txt':
            continue
        target = current / asset.name
        if target.exists() and target.read_bytes() != asset.read_bytes():
            if not (asset.name.endswith('_METADATA.json') and same_metadata(asset, target)):
                raise ValueError(f'Release asset {asset.name} changed; publish a new version')
            # Keep the published bytes so existing checksums stay valid.
            target.unlink()
        if not target.exists():
            shutil.copyfile(asset, target)


if __name__ == '__main__':
    merge(Path(sys.argv[1]), Path(sys.argv[2]))
