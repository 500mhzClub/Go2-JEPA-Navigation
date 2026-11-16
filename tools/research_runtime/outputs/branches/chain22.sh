#!/bin/bash
# P22 item 14 (approved 14 November): the decision bank (eval.decision_bank) under the frozen rule, built once the CPU
# is free (chain21's tracker and true-model suites done; M5's GPU missions keep 6 cores), then seven models scored on
# it: perfect predictions, realistic errors, dice:8, the ring tracker, M5, N4, N5. CPU, 24 workers. Resumable: the
# build skips finished missions; a score already written is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; L=$W/outputs/sensitivity/chain9.log
B=$W/outputs/decision_bank; mkdir -p $B
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
until grep -q "chain21: tracker and true model done" $L; do sleep 60; done
log "chain22 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
if [ ! -f $B/bank/summary.json ]; then
  cpu -m go2nav.eval.decision_bank build --out $B/bank --workers 24 > $B/build.log 2>&1
  code=$?; log "decision bank built, exit $code: $(tr -d '\n ' < $B/bank/summary.json 2>/dev/null | cut -c1-700)"
  [ $code -eq 0 ] || exit $code
fi
score() { [ -f $B/$1.json ] || cpu -m go2nav.eval.decision_bank score --bank $B/bank --model $2 --out $B/$1.json --workers 24 > $B/$1.log 2>&1
  log "bank $1: $(cpu -m go2nav.eval.decision_bank summary $B/$1.json | cut -c1-500)"; }
score true true
score tracker ring_velocity
score realistic mix:noise=0.03,fresh=0.3
score dice8 dice:8
score m5 objects:$O/m5/model.pt
score n4 expect:$O/n4/model.pt
score n5 expect:$O/n5/model.pt
log "chain22 done"
