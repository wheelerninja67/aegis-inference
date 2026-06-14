#!/usr/bin/env bash
set -e

echo "========================================"
echo " Aegis Inference 1.58-bit Engine Setup  "
echo "========================================"

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Linux" ]; then
    TARGET="aegis-linux-amd64"
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        TARGET="aegis-darwin-arm64"
    else
        echo "[-] Apple Intel is not actively optimized. Compiling from source is recommended."
        exit 1
    fi
else
    echo "[-] Unsupported OS: $OS"
    exit 1
fi

echo "[*] Detecting latest GitHub release..."
LATEST_URL=$(curl -s https://api.github.com/repos/wheelerninja67/aegis-inference/releases/latest | grep "browser_download_url.*$TARGET" | cut -d '"' -f 4)

if [ -z "$LATEST_URL" ]; then
    echo "[-] Could not find release artifact for $TARGET. Are there any releases published?"
    exit 1
fi

echo "[*] Downloading $TARGET from $LATEST_URL..."
curl -sL "$LATEST_URL" -o /tmp/aegis_inference
chmod +x /tmp/aegis_inference

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

echo "[*] Installing to $INSTALL_DIR/aegis..."
mv /tmp/aegis_inference "$INSTALL_DIR/aegis"

echo "[+] Installation successful."
echo "Run 'aegis --help' to get started. Make sure $INSTALL_DIR is in your PATH."
