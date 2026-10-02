#!/usr/bin/env python3
"""Package a plugin with explicit versioned filenames and verified receipts."""
import argparse
import json
from pathlib import Path
import tomllib

from package_action import atomic_write, cache_lock, host_platform, sha256


def plugin_metadata(manifest):
    cargo = tomllib.loads(manifest.read_text())
    package = cargo['package']
    return {'name': package['metadata']['morflow']['plugin'],
            'version': package['version'], 'library': cargo['lib']['name']}


def package_plugin(manifest, binary, cache, target=None, latest=False):
    metadata = plugin_metadata(manifest)
    name, concrete = metadata['name'], metadata['version']
    target = target or host_platform()[0]
    ext = 'dll' if target.startswith('windows-') else 'dylib' if target.startswith('darwin-') else 'so'
    data = binary.read_bytes()
    with cache_lock(cache, 'maintenance'):
        for requested in [concrete] + (['latest'] if latest else []):
            path = cache / name / f'{name}_plugin-{requested}-{target}.{ext}'
            receipt = {'name': name, 'version': requested, 'platform': target,
                       'concrete_version': concrete, 'repository': 'local-build', 'sha256': sha256(data)}
            with cache_lock(cache, f'{name}/{requested} ({target})'):
                atomic_write(path, data)
                atomic_write(Path(str(path) + '.json'), json.dumps(receipt, indent=2).encode())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('target/release/plugins'))
    parser.add_argument('--platform')
    parser.add_argument('--latest', action='store_true')
    args = parser.parse_args()
    print(package_plugin(args.manifest, args.binary, args.cache, args.platform, args.latest))


if __name__ == '__main__':
    main()
