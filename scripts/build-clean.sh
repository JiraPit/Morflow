#!/usr/bin/env bash
# Remove local build outputs throughout the workspace, including examples,
# bindings, actions, and plugins. Dependency environments and media are retained.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
case "${1:-}" in
    '') ;;
    --dry-run) dry_run=true ;;
    -h|--help)
        echo "Usage: $0 [--dry-run]"
        echo 'Clean build outputs and Python caches throughout the repository.'
        echo 'Use --dry-run to list outputs without deleting them.'
        exit 0
        ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd "$repo_root"
# Do not follow symlinks or descend into dependencies, environments, or Git.
# Prune matched output directories so each tree is removed only once.
outputs=$(mktemp)
trap 'rm -f -- "$outputs"' EXIT
find . \( -type d \( -name .git -o -name node_modules \
    -o -name .venv -o -name venv -o -name env -o -name ENV \) -prune \) -o \
    \( -type d \( -name target -o -name build -o -name dist \
        -o -name .eggs -o -name '*.egg-info' -o -name __pycache__ \
        -o -name .pytest_cache -o -name .ruff_cache \) -print0 -prune \) -o \
    \( -type d -path './bindings/js/npm' -print0 -prune \) -o \
    \( -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \
        -o -name '*.dll' -o -name '*.node' -o -name '*.class' \
        -o -name '*.jar' -o -name '*.whl' -o -name '*.pyc' -o -name '*.pyo' \
        -o -path './bindings/js/native.d.ts' \) -print0 \) > "$outputs"

count=0
while IFS= read -r -d '' output; do
    if "$dry_run"; then
        printf 'Would remove: %s\n' "$output"
    else
        printf 'Removing: %s\n' "$output"
        rm -rf -- "$output"
    fi
    count=$((count + 1))
done < "$outputs"
if "$dry_run"; then
    printf 'Dry run complete: %s output paths found.\n' "$count"
else
    printf 'Build cleanup complete: %s output paths removed.\n' "$count"
fi
