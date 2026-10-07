#!/usr/bin/env bash
# Create the corpus render environment (Genesis 0.3.14 with the Vulkan backend, Python 3.12), used only by
# go2nav.corpus.render (through `python -m go2nav.corpus.datagen render --render-python VENV/bin/python`).
set -euo pipefail
VENV="${1:-.venv-render}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "torch==2.12.0" "torchvision==0.27.0"
grep -vE '^(#|torch==|torchvision==)' "$HERE/requirements-render.lock.txt" | "$VENV/bin/pip" install --no-deps -r /dev/stdin
echo "render environment ready: $VENV (run the renderer with PYTHONPATH=<repository> EGL_DEVICE_ID=<gpu> PYOPENGL_PLATFORM=egl)"
