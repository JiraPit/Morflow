#!/bin/bash
# Morflow Core & Bindings Builder
# Compiles core_types, parser, pipeline engine host/CLI, and all language bindings
# (Python, JavaScript/Node.js, Java JNI + Java JAR).
# Actions and plugins are excluded (use build-actions.sh and build-plugins.sh).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_DIR="target/release"

echo "=================================================="
echo " Morflow Core & Bindings Builder"
echo " (Excluding action packs)"
echo "=================================================="

echo "--> Building Engine Core (core_types, parser, pipeline)..."
cargo build --release -p core_types -p parser -p pipeline

echo "--> Building Language Bindings (Python, Node.js, Java JNI)..."
cargo build --release -p morflow-py -p morflow-node -p morflow-jni

echo "--> Building Java JAR package..."
if command -v mvn &> /dev/null; then
    (cd bindings/java && mvn package -DskipTests)
    mkdir -p "$TARGET_DIR"
    # Report the JAR that was actually produced rather than a hardcoded name.
    jar="$(ls -t bindings/java/target/morflow-*.jar 2>/dev/null | head -1)"
    if [ -n "$jar" ]; then
        cp "$jar" "$TARGET_DIR/"
        echo "    ✓ Java JAR -> $TARGET_DIR/$(basename "$jar")"
    else
        echo "    ✗ Error: no morflow-*.jar found in bindings/java/target/"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo " Engine & Bindings build complete!"
echo " Artifacts located in $TARGET_DIR/"
echo "=================================================="