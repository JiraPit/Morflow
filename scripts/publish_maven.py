#!/usr/bin/env python3
"""Publish the Java bindings to Maven Central through the Central Portal.

Takes the jar, sources jar, Javadoc jar and POM built by CI, signs each with
the imported GPG key, adds the checksums Central requires, uploads the bundle
and waits for the deployment to validate. A version already on Maven Central is
skipped: published versions are permanent there, as on every other registry.

Environment: MAVEN_CENTRAL_USERNAME and MAVEN_CENTRAL_PASSWORD (a Central
Portal user token) and MAVEN_GPG_PASSPHRASE for the imported signing key.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

API = "https://central.sonatype.com/api/v1/publisher"
# With AUTOMATIC publishing a deployment passes VALIDATED -> PUBLISHING -> PUBLISHED.
VALIDATED_STATES = {"VALIDATED", "PUBLISHING", "PUBLISHED"}


def auth_header() -> str:
    token = f"{os.environ['MAVEN_CENTRAL_USERNAME']}:{os.environ['MAVEN_CENTRAL_PASSWORD']}"
    return "Bearer " + base64.b64encode(token.encode()).decode()


def api(method: str, path: str, params: dict) -> dict:
    request = urllib.request.Request(
        f"{API}/{path}?{urllib.parse.urlencode(params)}",
        method=method,
        headers={"Authorization": auth_header(), "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def artifact_files(bundle_dir: Path, artifact: str, version: str) -> list:
    base = f"{artifact}-{version}"
    files = [bundle_dir / f"{base}{suffix}" for suffix in (".jar", "-sources.jar", "-javadoc.jar", ".pom")]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Maven artifacts: {', '.join(missing)}")
    return files


def gpg_sign(path: Path) -> None:
    subprocess.run(
        ["gpg", "--batch", "--yes", "--pinentry-mode", "loopback", "--passphrase-fd", "0",
         "--armor", "--detach-sign", str(path)],
        input=os.environ["MAVEN_GPG_PASSPHRASE"].encode(), check=True,
    )


def build_bundle(files: list, group: str, artifact: str, version: str, out: Path, sign=gpg_sign) -> Path:
    """Zip the artifacts in Maven repository layout with signatures and checksums."""
    layout = Path(*group.split("."), artifact, version)
    staging = out / "staging" / layout
    staging.mkdir(parents=True)
    for source in files:
        target = staging / source.name
        target.write_bytes(source.read_bytes())
        sign(target)
    for path in sorted(staging.iterdir()):
        data = path.read_bytes()
        for algorithm in ("md5", "sha1"):
            Path(f"{path}.{algorithm}").write_text(hashlib.new(algorithm, data).hexdigest())
    bundle = out / f"{artifact}-{version}-bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.iterdir()):
            archive.write(path, str(layout / path.name))
    return bundle


def upload(bundle: Path, name: str) -> str:
    query = urllib.parse.urlencode({"name": name, "publishingType": "AUTOMATIC"})
    result = subprocess.run(
        ["curl", "-sS", "--fail-with-body", "-H", f"Authorization: {auth_header()}",
         "-F", f"bundle=@{bundle}", f"{API}/upload?{query}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Central Portal upload failed: {result.stdout or result.stderr}")
    return result.stdout.strip()


def wait_until_validated(deployment: str, timeout: int) -> str:
    deadline = time.monotonic() + timeout
    while True:
        status = api("POST", "status", {"id": deployment})
        state = status.get("deploymentState")
        print(f"Deployment {deployment}: {state}")
        if state == "FAILED":
            raise RuntimeError("Central Portal rejected the deployment: " + json.dumps(status.get("errors"), indent=2))
        if state in VALIDATED_STATES:
            return state
        if time.monotonic() > deadline:
            raise TimeoutError(f"Deployment {deployment} still {state} after {timeout}s")
        time.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--group", default="org.morflow")
    parser.add_argument("--artifact", default="morflow")
    parser.add_argument("--version", required=True)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait for validation")
    args = parser.parse_args()

    published = api("GET", "published", {"namespace": args.group, "name": args.artifact, "version": args.version})
    if published.get("published"):
        print(f"{args.group}:{args.artifact}:{args.version} is already on Maven Central.")
        return 0

    files = artifact_files(args.bundle_dir, args.artifact, args.version)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = build_bundle(files, args.group, args.artifact, args.version, Path(tmp))
        deployment = upload(bundle, f"{args.artifact}-{args.version}")
    print(f"Uploaded {args.group}:{args.artifact}:{args.version} as deployment {deployment}")
    state = wait_until_validated(deployment, args.timeout)
    print(f"Deployment validated ({state}); Maven Central publishes it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
