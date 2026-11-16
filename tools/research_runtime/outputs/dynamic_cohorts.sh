#!/bin/bash
# One cohort service on the bench worktree's checked-out commit: bash outputs/dynamic_cohorts.sh CONFIG WORKERS --cohort NAME ...
set -euo pipefail
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-lewm-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
echo "[$(date +%H:%M:%S)] $1 $(git -C $B rev-parse --short HEAD) ${*:3}"
$PY -W ignore -m go2nav.eval.cohort $1 --workers $2 --lewm-lanes 6 "${@:3}"
echo "[$(date +%H:%M:%S)] done"
