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
sudo apt-get install -y libportaudio2 python3-venv curl unzip

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

mkdir -p models
VOSK_MODEL="vosk-model-small-en-us-0.15"
if [[ ! -d "models/$VOSK_MODEL" ]]; then
    echo "Downloading the offline English phrase model (~40 MB)..."
    MODEL_TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$MODEL_TMP_DIR"' EXIT
    curl --fail --location --retry 3 \
        "https://alphacephei.com/vosk/models/$VOSK_MODEL.zip" \
        --output "$MODEL_TMP_DIR/$VOSK_MODEL.zip"
    unzip -q "$MODEL_TMP_DIR/$VOSK_MODEL.zip" -d "$MODEL_TMP_DIR"
    mv "$MODEL_TMP_DIR/$VOSK_MODEL" "models/$VOSK_MODEL"
    rm -rf "$MODEL_TMP_DIR"
    trap - EXIT
fi

if [[ ! -s "models/yamnet.tflite" ]]; then
    echo "Downloading the YAMNet emergency-sound classifier..."
    MODEL_TMP_FILE="models/yamnet.tflite.tmp"
    curl --fail --location --retry 3 \
        "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" \
        --output "$MODEL_TMP_FILE"
    mv "$MODEL_TMP_FILE" models/yamnet.tflite
fi

echo
echo "Setup complete. Connect a microphone, then run: ./run_sentinel.sh"
