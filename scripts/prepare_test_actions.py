#!/usr/bin/env python3
"""Package built debug actions with explicit local latest aliases for SDK tests."""
import argparse
from pathlib import Path
from package_action import action_metadata, host_platform, package_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['debug', 'release'], default='debug')
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    cache = args.cache or repo / 'target' / args.profile / 'actions'
    _, ext = host_platform()
    prefix = '' if ext == 'dll' else 'lib'
    for manifest in sorted((repo / 'actions').glob('*/*/Cargo.toml')):
        binary = repo / 'target' / args.profile / f'{prefix}{action_metadata(manifest)["library"]}.{ext}'
        package_action(manifest, binary, cache, latest=True)
    print(f'Local test fixtures ready in {cache}')


if __name__ == '__main__':
    main()
