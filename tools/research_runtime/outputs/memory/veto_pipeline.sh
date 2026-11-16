#!/bin/bash
# P9b (docs/dynamic/PLAN.md) on the pinned worktree: walkers that do not yield with the memory as a veto, checked by the
# stopping rule every minute (outputs/memory/veto_check.py); only on PASS the four-walker and fixed-record cohorts, checked
# likewise. Resumable: rerun and the cohorts --resume.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-memory-bench
M=outputs/memory
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date +%H:%M)] $*" >> $M/veto.log; }
phase() {   # config, checker argument
  $PY -W ignore -m go2nav.eval.cohort $1 --workers 6 --lewm-lanes 6 --resume >> $M/veto_missions.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    v=$($PY $M/veto_check.py $2 | tail -1)
    case "$v" in *FAIL*) log "$2 stopped: $v"; systemctl --user stop memory-veto; exit 0;; esac
  done
  v=$($PY $M/veto_check.py $2 | tail -1); log "$2 done: $v"
  case "$v" in PASS*|*" PASS") return 0;; *) return 1;; esac
}
log "start"
phase configs/experiments/walker_memory_veto.yaml rude || exit 0
phase configs/experiments/walker_memory_veto_rest.yaml rest
log "end"
