#!/bin/bash
# Standalone Action Pack Builder
# Compiles Morflow action packages into native dynamic libraries (.so, .dylib, .dll)
# without building the engine, parser, or language bindings.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_DIR="target/release"

TARGET_PACK="${1:-all}"
TARGET_ACTION="${2:-all}"

# Detect OS and binary extension
UNAME_S="$(uname -s)"
case "$UNAME_S" in
    Linux*)     EXT="so"; PREFIX="lib" ;;
    Darwin*)    EXT="dylib"; PREFIX="lib" ;;
    CYGWIN*|MINGW*|MSYS*) EXT="dll"; PREFIX="" ;;
    *)          EXT="so"; PREFIX="lib" ;;
esac

echo "=================================================="
echo " Morflow Action Pack Builder"
echo " Host OS: $UNAME_S (producing .$EXT libraries)"
echo " Target Pack: $TARGET_PACK | Target Action: $TARGET_ACTION"
echo "=================================================="

# Function to build and copy a single action.
# Returns 0 if the action was built, 1 if it was filtered out by the
# target pack/action arguments. A hard failure exits the script.
build_action() {
    local pack="$1"
    local action="$2"

    if [ "$TARGET_PACK" != "all" ] && [ "$TARGET_PACK" != "$pack" ]; then
        return 1
    fi
    if [ "$TARGET_ACTION" != "all" ] && [ "$TARGET_ACTION" != "$action" ]; then
        return 1
    fi

    echo "--> Compiling [$pack] $action..."
    cargo build --release --package "$action"

    mkdir -p "$TARGET_DIR/actions/$pack"

    local src_file="$TARGET_DIR/${PREFIX}${action}.${EXT}"
    if [ ! -f "$src_file" ]; then
        src_file="$TARGET_DIR/${action}.${EXT}"
    fi

    if [ -f "$src_file" ]; then
        local dest_action_file="${action}_action.${EXT}"
        cp "$src_file" "$TARGET_DIR/actions/$pack/$dest_action_file"
        cp "$src_file" "$TARGET_DIR/actions/$dest_action_file"
        echo "    ✓ Packaged to $TARGET_DIR/actions/$pack/$dest_action_file"
        return 0
    else
        echo "    ✗ Error: Compiled library not found at $src_file"
        exit 1
    fi
}

# Discover every action under actions/<pack>/<action>/ and build it.
# The action tree is the single source of truth, so new actions and packs are
# picked up automatically without editing this script.
BUILT=0
SKIPPED=0

for pack_dir in actions/*/; do
    pack="$(basename "$pack_dir")"
    for action_dir in "$pack_dir"*/; do
        [ -f "$action_dir/Cargo.toml" ] || continue
        act="$(basename "$action_dir")"
        build_action "$pack" "$act" && BUILT=$((BUILT + 1)) || SKIPPED=$((SKIPPED + 1))
    done
done

echo ""
echo "=================================================="
echo " Action Pack build complete! ($BUILT built, $SKIPPED skipped)"
echo " Binaries located in $TARGET_DIR/actions/"
echo "=================================================="
