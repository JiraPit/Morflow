#!/bin/bash
set -e

TARGET_DIR="target/release"

echo "Building in release mode..."
cargo build --release

echo "Creating action pack directories..."
mkdir -p "$TARGET_DIR/actions/base"
mkdir -p "$TARGET_DIR/actions/image_essentials"
mkdir -p "$TARGET_DIR/actions/audio_essentials"

echo "Copying action pack artifacts..."

copy_action() {
    local pack="$1"
    local name="$2"
    for ext in so dylib dll; do
        local lib="$TARGET_DIR/lib${name}.${ext}"
        if [ -f "$lib" ]; then
            local action_name="${name}_action.${ext}"
            cp "$lib" "$TARGET_DIR/actions/$pack/$action_name"
            cp "$lib" "$TARGET_DIR/actions/$action_name"
            echo "  [$pack] $name -> actions/$pack/$action_name"
        fi
    done
}

# Base pack
for act in identity to_tensor; do
    copy_action "base" "$act"
done

# Image Essentials pack
for act in to_image resize crop pad color_adjust gaussian_blur edge_detect sharpen threshold rotate flip blend morphology; do
    copy_action "image_essentials" "$act"
done

# Audio Essentials pack
for act in to_audio to_pcm to_wav gain normalize biquad_filter compressor limiter noise_gate stereo_widen resample stft delay; do
    copy_action "audio_essentials" "$act"
done

echo "Done. Actions organized into ActionPacks in $TARGET_DIR/actions/"
ls -la "$TARGET_DIR/actions/"