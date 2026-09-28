#!/bin/bash
# Morflow Engine & Bindings Builder
# Compiles core_types, parser, pipeline engine host, and all language bindings
# (Python, JavaScript/Node.js, Java JNI + Java JAR).
# Action packs are strictly excluded (use scripts/build-actions.sh for actions).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_DIR="target/release"

echo "=================================================="
echo " Morflow Engine & Bindings Builder"
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
    cp bindings/java/target/morflow-*.jar "$TARGET_DIR/" 2>/dev/null || true
    echo "    ✓ Java JAR -> $TARGET_DIR/morflow-0.1.0.jar"
fi

echo ""
echo "=================================================="
echo " Engine & Bindings build complete!"
echo " Artifacts located in $TARGET_DIR/"
echo "=================================================="