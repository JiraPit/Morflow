#!/bin/sh
# Morflow CLI Installer
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/JiraPit/Morflow/main/install.sh | bash
#
# Custom options:
#   curl -fsSL https://raw.githubusercontent.com/JiraPit/Morflow/main/install.sh | bash -s -- --dir /usr/local/bin
#   MORFLOW_VERSION=0.1.2 curl -fsSL https://raw.githubusercontent.com/JiraPit/Morflow/main/install.sh | bash
#
# Environment variables:
#   MORFLOW_INSTALL_DIR  Directory to install the binary to (default: ~/.morflow/bin)
#   MORFLOW_VERSION      Specific version to install (default: latest engine release)
#   MORFLOW_REPO         GitHub repository (default: JiraPit/Morflow)

set -e

REPO="${MORFLOW_REPO:-JiraPit/Morflow}"
INSTALL_DIR="${MORFLOW_INSTALL_DIR:-$HOME/.morflow/bin}"
VERSION="${MORFLOW_VERSION:-}"
FALLBACK_VERSION="0.1.2"

# Parse optional arguments when invoked as `bash -s -- [args]`
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--dir|--install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Morflow CLI Installer"
            echo ""
            echo "Usage: install.sh [options]"
            echo ""
            echo "Options:"
            echo "  -d, --dir <path>       Install directory (default: ~/.morflow/bin)"
            echo "  -v, --version <ver>    Specific version to install (default: latest)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MORFLOW_INSTALL_DIR    Install directory"
            echo "  MORFLOW_VERSION        Version to install"
            echo "  MORFLOW_REPO           GitHub repository (owner/name)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information."
            exit 1
            ;;
    esac
done

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)   PLATFORM_OS="linux" ;;
        Darwin)  PLATFORM_OS="darwin" ;;
        *)
            echo "Error: Unsupported operating system: $OS"
            echo "Morflow currently supports Linux (x86_64) and macOS (Apple Silicon arm64)."
            exit 1
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)    PLATFORM_ARCH="x86_64" ;;
        aarch64|arm64)   PLATFORM_ARCH="arm64" ;;
        *)
            echo "Error: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    if [ "$PLATFORM_OS" = "linux" ]; then
        if [ "$PLATFORM_ARCH" = "x86_64" ]; then
            PLATFORM="linux-x86_64"
        else
            echo "Error: Pre-built Linux binaries are currently available for x86_64 only."
            echo "You can build from source via 'cargo build --release -p pipeline'."
            exit 1
        fi
    elif [ "$PLATFORM_OS" = "darwin" ]; then
        if [ "$PLATFORM_ARCH" = "arm64" ]; then
            PLATFORM="darwin-arm64"
        else
            echo "Error: Pre-built macOS binaries are currently available for Apple Silicon (arm64)."
            echo "You can build from source via 'cargo build --release -p pipeline'."
            exit 1
        fi
    fi

    ARCHIVE_EXT="tar.gz"
}

resolve_version() {
    if [ -n "$VERSION" ]; then
        echo "$VERSION" | sed 's/^v//'
        return
    fi

    RELEASES_URL="https://api.github.com/repos/${REPO}/releases"
    DETECTED_TAG=""

    if command -v curl >/dev/null 2>&1; then
        DETECTED_TAG="$(curl -fsSL "$RELEASES_URL" 2>/dev/null \
            | grep '"tag_name": "v[0-9]' | head -1 \
            | sed 's/.*"tag_name": "v\([^"]*\)".*/\1/' || true)"
    elif command -v wget >/dev/null 2>&1; then
        DETECTED_TAG="$(wget -qO- "$RELEASES_URL" 2>/dev/null \
            | grep '"tag_name": "v[0-9]' | head -1 \
            | sed 's/.*"tag_name": "v\([^"]*\)".*/\1/' || true)"
    fi

    if [ -n "$DETECTED_TAG" ]; then
        echo "$DETECTED_TAG"
    else
        echo "$FALLBACK_VERSION"
    fi
}

main() {
    detect_platform

    echo "==================================================="
    echo " Morflow CLI Installer"
    echo "==================================================="
    echo ""

    VER="$(resolve_version)"
    ARCHIVE_NAME="morflow-v${VER}-${PLATFORM}.${ARCHIVE_EXT}"
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${VER}/${ARCHIVE_NAME}"

    echo "  Version:       v${VER}"
    echo "  Platform:      ${PLATFORM}"
    echo "  Install Dir:   ${INSTALL_DIR}"
    echo ""

    mkdir -p "$INSTALL_DIR"

    TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'morflow-install')"
    TMP_ARCHIVE="${TMP_DIR}/${ARCHIVE_NAME}"

    cleanup() {
        rm -rf "$TMP_DIR"
    }
    trap cleanup EXIT INT TERM

    echo "  [↓] Downloading ${ARCHIVE_NAME}..."

    DOWNLOAD_SUCCESS=0
    if command -v curl >/dev/null 2>&1; then
        if curl -fsSL -o "$TMP_ARCHIVE" "$DOWNLOAD_URL" 2>/dev/null; then
            DOWNLOAD_SUCCESS=1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q -O "$TMP_ARCHIVE" "$DOWNLOAD_URL" 2>/dev/null; then
            DOWNLOAD_SUCCESS=1
        fi
    fi

    if [ "$DOWNLOAD_SUCCESS" -ne 1 ] || [ ! -s "$TMP_ARCHIVE" ]; then
        echo ""
        echo "  Error: Failed to download ${ARCHIVE_NAME}"
        echo "  URL: ${DOWNLOAD_URL}"
        echo ""
        echo "  Please check releases at: https://github.com/${REPO}/releases"
        echo ""
        echo "  Alternative installs (no compilation required):"
        echo "    pip install morflow          # Python CLI"
        echo "    npx morflow                  # Node.js CLI"
        exit 1
    fi

    echo "  [⇥] Extracting binary..."
    tar -xzf "$TMP_ARCHIVE" -C "$TMP_DIR"

    BIN_PATH=""
    if [ -f "$TMP_DIR/morflow" ]; then
        BIN_PATH="$TMP_DIR/morflow"
    elif [ -f "$TMP_DIR/bin/morflow" ]; then
        BIN_PATH="$TMP_DIR/bin/morflow"
    else
        BIN_PATH="$(find "$TMP_DIR" -name "morflow" -type f | head -1)"
    fi

    if [ -z "$BIN_PATH" ]; then
        echo "  Error: Could not locate 'morflow' executable inside the downloaded archive."
        exit 1
    fi

    chmod +x "$BIN_PATH"
    mv "$BIN_PATH" "$INSTALL_DIR/morflow"

    echo ""
    echo "  ✓ Installed morflow binary to: ${INSTALL_DIR}/morflow"

    # Verify binary executes
    INSTALLED_VER="$("${INSTALL_DIR}/morflow" --version 2>/dev/null || echo "morflow v${VER}")"
    echo "  ✓ Verified executable: ${INSTALLED_VER}"
    echo ""

    # PATH checks and instructions
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*)
            echo "  ✓ ${INSTALL_DIR} is already in your \$PATH."
            ;;
        *)
            echo "  -------------------------------------------------"
            echo "  ACTION REQUIRED: Add to PATH"
            echo "  -------------------------------------------------"
            echo "  To run 'morflow' from any terminal, add the install"
            echo "  directory to your PATH by running:"
            echo ""
            SHELL_NAME="$(basename "${SHELL:-sh}")"
            case "$SHELL_NAME" in
                zsh)
                    echo "    echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.zshrc"
                    echo "    source ~/.zshrc"
                    ;;
                fish)
                    echo "    fish_add_path ${INSTALL_DIR}"
                    ;;
                *)
                    echo "    echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
                    echo "    source ~/.bashrc"
                    ;;
            esac
            echo ""
            ;;
    esac

    echo "==================================================="
    echo " Installation complete!"
    echo "==================================================="
    echo ""
    echo " Quick start:"
    echo "   morflow search gain                # Search action registry"
    echo "   morflow spec base/latest/identity  # View action docs & spec"
    echo "   morflow prep pipeline.morf         # Pre-download pipeline actions"
    echo "   morflow list                       # List installed actions"
    echo ""
}

main "$@"
