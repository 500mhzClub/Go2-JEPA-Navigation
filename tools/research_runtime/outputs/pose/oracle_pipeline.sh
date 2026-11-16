#!/bin/bash
# P10 step 1 on the pinned worktree; the gate checked every minute (oracle_check.py). Resumable: rerun.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-pose-bench
M=outputs/pose
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date +%H:%M)] $*" >> $M/oracle.log; }
log "start"
$PY -W ignore -m go2nav.eval.cohort configs/experiments/pose_oracle_rude.yaml --workers 6 --lewm-lanes 6 --resume >> $M/oracle_missions.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 60
  v=$($PY $M/oracle_check.py | tail -1)
  case "$v" in FAIL*) log "stopped: $($PY $M/oracle_check.py | head -1) $v"; systemctl --user stop pose-oracle; exit 0;; esac
done
log "done: $($PY $M/oracle_check.py | tr '\n' ' ')"
