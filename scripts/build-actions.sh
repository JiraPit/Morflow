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

# Function to build and copy a single action
build_action() {
    local pack="$1"
    local action="$2"

    if [ "$TARGET_PACK" != "all" ] && [ "$TARGET_PACK" != "$pack" ]; then
        return 0
    fi
    if [ "$TARGET_ACTION" != "all" ] && [ "$TARGET_ACTION" != "$action" ]; then
        return 0
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
    else
        echo "    ✗ Error: Compiled library not found at $src_file"
        exit 1
    fi
}

mkdir -p "$TARGET_DIR/actions/base"
mkdir -p "$TARGET_DIR/actions/audio_essentials"
mkdir -p "$TARGET_DIR/actions/image_essentials"

# Base Actions
for act in identity to_tensor; do
    build_action "base" "$act"
done

# Image Essentials Actions
for act in to_image resize crop pad color_adjust gaussian_blur edge_detect sharpen threshold rotate flip blend morphology; do
    build_action "image_essentials" "$act"
done

# Audio Essentials Actions
for act in to_audio to_pcm to_wav gain normalize biquad_filter compressor limiter noise_gate stereo_widen resample stft delay; do
    build_action "audio_essentials" "$act"
done

echo ""
echo "=================================================="
echo " Action Pack build complete!"
echo " Binaries located in $TARGET_DIR/actions/"
echo "=================================================="
