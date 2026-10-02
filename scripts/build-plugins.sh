#!/usr/bin/env bash
# Build native plugins independently of the engine and actions.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
requested="${1:-all}"
case "$(uname -s)" in
    Darwin*) ext=dylib; prefix=lib ;;
    CYGWIN*|MINGW*|MSYS*) ext=dll; prefix= ;;
    Linux*) ext=so; prefix=lib ;;
    *) echo 'Unsupported build platform' >&2; exit 1 ;;
esac
built=0
for manifest in plugins/*/Cargo.toml; do
    [ -f "$manifest" ] || continue
    name="$(basename "$(dirname "$manifest")")"
    [ "$requested" = all ] || [ "$requested" = "$name" ] || continue
    # Explicitly require shared OpenCV. The crate builds only its generated glue.
    if [ "$name" = opencv-bridge ]; then
        export OPENCV_LINK_LIBS="${MORFLOW_OPENCV_LINK_LIBS:-dylib=opencv_core,dylib=opencv_imgproc}"
        IFS=',' read -ra link_libraries <<< "$OPENCV_LINK_LIBS"
        for library in "${link_libraries[@]}"; do
            case "$library" in dylib=*) ;; *) echo 'OpenCV libraries must be explicitly linked as dylib=<name>' >&2; exit 1 ;; esac
        done
    fi
    cargo build --release --manifest-path "$manifest"
    library="$(python3 -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); from package_plugin import plugin_metadata; print(plugin_metadata(Path(sys.argv[1]))["library"])' "$manifest")"
    python3 scripts/package_plugin.py "$manifest" "target/release/${prefix}${library}.${ext}"
    built=$((built + 1))
done
if [ "$built" -eq 0 ]; then echo "No plugins matched: $requested" >&2; exit 1; fi
echo "Plugin build complete: $built built in target/release/plugins"
