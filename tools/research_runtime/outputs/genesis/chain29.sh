#!/bin/bash
# P24 2d (approved 15 November): the suites from the pinned checkout ../Go2-JEPA-Navigation-p24 (8d6455e), each under
# the stop rule (outputs/genesis/check.py; a broken suite stops the chain for diagnosis). 6 at once on the R9700; the
# frozen rule in the environment. Resumable: finished missions are kept (--resume).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-p24; G=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/genesis
L=$G/chain.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1 PYTORCH_TUNABLEOP_ENABLED=0
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
suite() {  # kind (polite or rude), cohorts
  local kind=$1; shift
  local args=""; for c in "$@"; do args="$args --cohort $c"; done
  log "suite: $* ($(git rev-parse --short HEAD))"
  $PY -W ignore -m go2nav.eval.cohort configs/experiments/p24_tapes.yaml $args --workers 6 --lewm-lanes 6 --resume \
    >> $G/missions29.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if [ "$($PY $G/check.py suite $kind "$@" | tail -1)" = BROKEN ]; then
      log "stopped by the stop rule: $($PY $G/check.py suite $kind "$@" | head -1)"; kill $pid; wait $pid 2>/dev/null; return 1
    fi
  done
  log "done: $* :: $($PY $G/check.py suite $kind "$@" | head -1)"
}
log "chain29 start $(git rev-parse --short HEAD)"
suite polite tapes_still_m5 || exit 0
suite polite tapes_obst1_m5 tapes_obst2_m5 || exit 0
suite polite tapes_obst4_m5 || exit 0
suite rude tapes_rude1_m5 tapes_rude2_m5 || exit 0
suite polite tapes_obst4_cv || exit 0
suite rude tapes_rude1_cv tapes_rude2_cv || exit 0
log "chain29 done"
