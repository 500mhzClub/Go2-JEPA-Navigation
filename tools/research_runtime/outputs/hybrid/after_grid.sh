#!/bin/bash
# P5's hybrid missions (configs/experiments/hybrid_development.yaml) on the hybrid bench worktree's pinned commit, once
# the GPU is free: H-JEPA v2 finished and the grid LeWM service (grid-after-hjepa) ended, and the corpus-v2 heads
# exist. At most six LeWM missions at a time (--lewm-lanes 6). `--resume` continues an interrupted run.
set -euo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-hybrid-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
until [ -f $W/outputs/hjepa/go2_3level_v2/hjepa.pt ] && [ -f $W/outputs/lewm/go2_v2_heads/heads.pt ] \
  && ! systemctl --user is-active --quiet corpus-v2 grid-after-hjepa; do sleep 120; done
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
echo "[$(date +%H:%M:%S)] hybrid at $(git -C $B rev-parse --short HEAD)"
$PY -W ignore -m go2nav.eval.cohort configs/experiments/hybrid_development.yaml --workers 7 --lewm-lanes 6 \
  ${RESUME:+--resume}
echo "[$(date +%H:%M:%S)] done"
