#!/bin/bash
# R7b (moving obstacles, seen-free map clearing on), queued behind R7 (dynamic-r7); --resume makes a rerun continue.
set -euo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=$W-lewm-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
COMMIT=${1:?commit}
CONFIG=${2:-configs/experiments/r7b_map_decay.yaml}
while systemctl --user is-active -q dynamic-r7 dynamic-r7b; do sleep 60; done
git -C $B checkout -q --detach $COMMIT
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
echo "[$(date +%H:%M:%S)] $CONFIG"
$PY -W ignore -m go2nav.eval.cohort $CONFIG --workers 16 --lewm-lanes 6 --resume "${@:3}"
echo "[$(date +%H:%M:%S)] done"
