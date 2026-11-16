#!/bin/bash
# P13 items 2 and 6 (no-slip) on the pinned worktree: the range-sensor reference, walkers that do not yield first (the measure), then the
# fixed record's and four walkers' suites (phase D's references). Resumable: rerun. Stops if reference_check says BROKEN.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-a-bench
M=outputs/phase_a
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date +%H:%M)] $*" >> $M/reference.log; }
log "start"
for group in "rsn_rude_obst1_c5 camn_rude_obst1_c5 rsn_rude_obst2_c5 camn_rude_obst2_c5" "rsn_obst1_c5 rsn_obst2_c5" "rsn_obst4_c5"; do
  args=""; for c in $group; do args="$args --cohort $c"; done
  log "suite: $group"
  $PY -W ignore -m go2nav.eval.cohort configs/experiments/noslip_reference.yaml $args --workers 6 --lewm-lanes 6 --resume >> $M/reference_missions.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    v=$($PY $M/reference_check.py | tail -1)
    case "$v" in BROKEN*) log "stopped: $v"; $PY $M/reference_check.py >> $M/reference.log; systemctl --user stop phase-a-reference; exit 0;; esac
  done
  log "done: $group :: $($PY $M/reference_check.py | tr '\n' ' ')"
done
log "all suites done"
