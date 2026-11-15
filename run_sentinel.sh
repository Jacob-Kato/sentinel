#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # The local .env file is git-ignored and must not be committed.
    source "$ROOT_DIR/.env"
    set +a
fi

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c \
    'import numpy, onnxruntime, sounddevice, soxr' >/dev/null 2>&1; then
    echo "Sentinel dependencies are missing; running first-time setup..."
    bash "$ROOT_DIR/scripts/setup_pi.sh"
fi

exec "$ROOT_DIR/.venv/bin/python" -m sentinel.live
