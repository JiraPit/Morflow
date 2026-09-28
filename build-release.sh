#!/bin/bash
set -e

TARGET_DIR="target/release"

echo "Building in release mode..."
cargo build --release

echo "Creating actions directory..."
mkdir -p "$TARGET_DIR/actions"

echo "Copying actions..."
shopt -s nullglob
for lib in "$TARGET_DIR"/lib*.so "$TARGET_DIR"/lib*.dylib "$TARGET_DIR"/*.dll; do
    if [ -f "$lib" ]; then
        base=$(basename "$lib")
        name="${base#lib}"
        name="${name%.*}"
        ext="${base##*.}"
        new_name="${name}_action.${ext}"
        cp "$lib" "$TARGET_DIR/actions/$new_name"
        echo "  Copied: $base -> $new_name"
    fi
done
shopt -u nullglob

echo "Done. Actions in $TARGET_DIR/actions/"
ls -la "$TARGET_DIR/actions/"