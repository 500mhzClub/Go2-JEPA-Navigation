#!/bin/bash
# R7 (moving obstacles), queued behind the M3 cohort (loco-m3); --resume makes a rerun continue.
set -euo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=$W-lewm-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
while systemctl --user is-active -q loco-m3; do sleep 60; done
git -C $B checkout -q --detach 3a96cd0
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
echo "[$(date +%H:%M:%S)] R7"
$PY -W ignore -m go2nav.eval.cohort configs/experiments/r7_moving_obstacles.yaml --workers 16 --lewm-lanes 6 --resume
echo "[$(date +%H:%M:%S)] done"
