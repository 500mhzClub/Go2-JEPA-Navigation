#!/usr/bin/env bash
# Create the Go2-JEPA-Navigation environment (ROCm 7.2, Python 3.12).
set -euo pipefail
VENV="${1:-.venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/rocm7.2 \
  "torch==2.12.0+rocm7.2" "torchvision==0.27.0+rocm7.2" "triton-rocm"
grep -vE '^(#|torch==|torchvision==|triton==|triton-rocm==)' "$HERE/requirements-rocm7.2.lock.txt" \
  | "$VENV/bin/pip" install --no-deps -r /dev/stdin
"$VENV/bin/pip" install --no-deps -e "$HERE/.."
echo "environment ready: $VENV"
