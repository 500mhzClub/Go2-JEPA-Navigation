#!/bin/bash
# P18: the smoke test, then (only if it passes) the suites, from the pinned worktree ../Go2-JEPA-Navigation-p18.
# 6 missions at once on the R9700. Resumable: rerun. The script stops a suite itself under the stop rule.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-p18
M=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/planner_missions
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $M/pipeline3.log; }
cohorts() {  # run cohorts as one suite under the stop rule
  args=""; for c in "$@"; do args="$args --cohort $c"; done
  log "suite: $* ($(git rev-parse --short HEAD))"
  $PY -W ignore -m go2nav.eval.cohort configs/experiments/cost_decision.yaml $args --workers 6 --lewm-lanes 6 --resume >> $M/missions3.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    v=$($PY $M/check.py suite "$@" | tail -1)
    if [ "$v" = BROKEN ]; then log "stopped by the stop rule: $($PY $M/check.py suite "$@" | head -1)"; kill $pid; wait $pid 2>/dev/null; return 1; fi
  done
  log "done: $* :: $($PY $M/check.py suite "$@" | head -1)"
}
log "waiting for the polite rerun"
until grep -q "done: cost2_obst1_c5 cost2_obst2_c5" $M/pipeline2.log; do
  systemctl --user is-active -q planner-missions2 || { log "planner-missions2 ended without the polite suites: nothing run"; exit 0; }
  sleep 20
done
systemctl --user stop planner-missions2
sleep 20
log "planner-missions2 stopped after the polite suites (the queued still and four-walker reruns are not run)"
cohorts cost2r_obst1_c5 || exit 0
log "repeat done"
