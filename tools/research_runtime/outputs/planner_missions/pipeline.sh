#!/bin/bash
# P18: the smoke test, then (only if it passes) the suites, from the pinned worktree ../Go2-JEPA-Navigation-p18.
# 6 missions at once on the R9700. Resumable: rerun. The script stops a suite itself under the stop rule.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-p18
M=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/planner_missions
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $M/pipeline.log; }
cohorts() {  # run cohorts as one suite under the stop rule
  args=""; for c in "$@"; do args="$args --cohort $c"; done
  log "suite: $* ($(git rev-parse --short HEAD))"
  $PY -W ignore -m go2nav.eval.cohort configs/experiments/cost_decision.yaml $args --workers 6 --lewm-lanes 6 --resume >> $M/missions.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    v=$($PY $M/check.py suite "$@" | tail -1)
    if [ "$v" = BROKEN ]; then log "stopped by the stop rule: $($PY $M/check.py suite "$@" | head -1)"; kill $pid; wait $pid 2>/dev/null; return 1; fi
  done
  log "done: $* :: $($PY $M/check.py suite "$@" | head -1)"
}
log "start"
if ! grep -q "^smoke: " $M/pipeline.log; then
  cohorts cost_smoke_still_c5 || { log "smoke: FAIL (stop rule)"; exit 0; }
  cohorts cost_smoke_obst1_c5 || { log "smoke: FAIL (stop rule)"; exit 0; }
  $PY $M/check.py smoke >> $M/pipeline.log
  echo "smoke: $($PY $M/check.py smoke | tail -1)" >> $M/pipeline.log
fi
grep -q "^smoke: PASS" $M/pipeline.log || { log "the smoke did not pass: no suites"; exit 0; }
cohorts cost_still_c5 rsn_still_c5 || exit 0
cohorts cost_obst1_c5 cost_obst2_c5 || exit 0
cohorts cost_obst4_c5 || exit 0
log "all suites done"
