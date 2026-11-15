#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Sentinel Pi setup requires Linux." >&2
    exit 1
fi

MACHINE="$(uname -m)"
if [[ "$MACHINE" == arm* && "$MACHINE" != "aarch64" ]]; then
    echo "A 64-bit ARM64 operating system is required on Raspberry Pi." >&2
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This setup script expects Raspberry Pi OS or another apt-based system." >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install PortAudio and Python venv support." >&2
    exit 1
fi

echo "Installing system audio and virtual environment dependencies..."
sudo apt-get update
sudo apt-get install -y libportaudio2 python3-venv

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "Sentinel requires Python 3.11 or newer." >&2
    exit 1
fi

if [[ ! -f "models/silero_vad.onnx" ]]; then
    echo "Missing models/silero_vad.onnx. Run this from a complete Sentinel checkout." >&2
    exit 1
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[live]'

echo
echo "Setup complete. Connect a microphone, then run: ./run_sentinel.sh"
