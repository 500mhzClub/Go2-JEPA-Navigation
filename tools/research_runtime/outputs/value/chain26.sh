#!/bin/bash
# P22b item 8's noise floor (approved 15 November): at the 582 validation moments of the branches, each tape run again
# three times with new walkers' dice (eval.value_noise), and the chooser that knows each tape's expected cost scored as
# the ranking bar scores the value. CPU, 26 workers. Resumable: finished moments are skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; V=$W/outputs/value; L=$V/chain.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain26 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.value_noise run --branches $V/branches \
  --out $V/noise --samples 3 --workers 26 > $V/noise.log 2>&1
code=$?; log "noise floor, exit $code: $(tail -1 $V/noise.log | cut -c1-600)"
log "chain26 done"
