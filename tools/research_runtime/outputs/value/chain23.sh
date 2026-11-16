#!/bin/bash
# P22b (approved 14 November) round 0: 5,000 missions of the ring tracker's planner under the frozen rule on the corpus's
# layouts (data.value_missions; one plan in ten explores), then packed. CPU, 26 workers. Resumable: finished missions
# are skipped. Training waits for the decision on its loss and bar (a).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; V=$W/outputs/value; L=$V/chain.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain23 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
cpu -m go2nav.data.value_missions build --out $V/data0 --missions 5000 --val 500 --workers 26 > $V/data0.log 2>&1
code=$?; log "data0 built, exit $code: $(tail -1 $V/data0.log | cut -c1-300)"
[ $code -eq 0 ] || exit $code
cpu -m go2nav.data.value_missions pack $V/data0 --workers 26 >> $V/data0.log 2>&1
code=$?; log "data0 packed, exit $code: $(tail -1 $V/data0.log | cut -c1-200)"
log "chain23 done"
