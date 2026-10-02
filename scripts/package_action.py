#!/usr/bin/env python3
"""Package a local Cargo action with versioned filenames and checksum receipts.

--latest explicitly creates a local development/test alias. Production latest
aliases are refreshed by morflow prep/install against published checksums.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import tomllib


def host_platform():
    systems = {'Linux': ('linux', 'so'), 'Darwin': ('darwin', 'dylib'), 'Windows': ('windows', 'dll')}
    system, ext = systems[platform.system()]
    arch = {'AMD64': 'x86_64', 'aarch64': 'arm64' if system == 'darwin' else 'aarch64'}.get(platform.machine(), platform.machine())
    return f'{system}-{arch}', ext


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def cache_lock(root, key):
    directory = root / '.locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (sha256(key.encode()) + '.lck')).open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def action_metadata(manifest):
    cargo = tomllib.loads(manifest.read_text())
    package = cargo['package']
    return {
        'name': package.get('metadata', {}).get('morflow', {}).get('action', package['name']),
        'package': package['name'],
        'library': cargo.get('lib', {}).get('name', package['name'].replace('-', '_')),
        'version': package['version'],
        'plugins': package.get('metadata', {}).get('morflow', {}).get('plugins', []),
    }


def package_action(manifest, binary, cache, target=None, latest=False):
    metadata = action_metadata(manifest)
    action, concrete = metadata['name'], metadata['version']
    pack = manifest.parent.parent.name
    host, ext = host_platform()
    target = target or host
    ext = 'dll' if target.startswith('windows-') else 'dylib' if target.startswith('darwin-') else 'so'
    data = binary.read_bytes()
    spec_path = manifest.parent / 'SPEC.md'
    spec = spec_path.read_bytes() if spec_path.exists() else None
    plugin_metadata = json.dumps({'plugins': metadata['plugins']}, indent=2).encode()
    with cache_lock(cache, 'maintenance'):
        for requested in [concrete] + (['latest'] if latest else []):
            filename = f'{action}_action-{requested}-{target}.{ext}'
            path = cache / pack / filename
            identity = {'pack': pack, 'version': requested, 'action': action, 'platform': target}
            receipt = dict(identity, concrete_version=concrete, repository='local-build', sha256=sha256(data), plugins=metadata['plugins'])
            with cache_lock(cache, f'{pack}/{requested}/{action} ({target})'):
                atomic_write(path, data)
                if spec is not None:
                    atomic_write(Path(str(path) + '.SPEC.md'), spec)
                atomic_write(Path(str(path) + '.json'), json.dumps(receipt, indent=2).encode())
                with cache_lock(cache, f'catalog-{pack}-{requested}'):
                    catalog_path = cache / pack / '.catalogs' / f'{requested}.json'
                    catalog = {'pack': pack, 'requested_version': requested, 'concrete_version': concrete,
                               'repository': 'local-build', 'checksums': {}}
                    if catalog_path.exists():
                        previous = json.loads(catalog_path.read_text())
                        if previous['concrete_version'] == concrete and previous['repository'] == 'local-build':
                            catalog = previous
                    catalog['checksums'][f'{action}_action-{concrete}-{target}.{ext}'] = sha256(data)
                    atomic_write(cache / pack / (action + '_METADATA.json'), plugin_metadata)
                    catalog['checksums'][f'{action}_METADATA.json'] = sha256(plugin_metadata)
                    if spec is not None:
                        catalog['checksums'][f'{action}_SPEC.md'] = sha256(spec)
                    atomic_write(catalog_path, json.dumps(catalog, indent=2).encode())
                if requested == 'latest':
                    concrete_catalog = dict(catalog, requested_version=concrete)
                    with cache_lock(cache, f'catalog-{pack}-{concrete}'):
                        atomic_write(cache / pack / '.catalogs' / f'{concrete}.json', json.dumps(concrete_catalog, indent=2).encode())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('target/release/actions'))
    parser.add_argument('--platform')
    parser.add_argument('--latest', action='store_true')
    args = parser.parse_args()
    print(package_action(args.manifest, args.binary, args.cache, args.platform, args.latest))


if __name__ == '__main__':
    main()
