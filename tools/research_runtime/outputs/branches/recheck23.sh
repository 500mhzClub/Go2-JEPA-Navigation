#!/bin/bash
# P22 item 14, before the merge: the five development switches removed (CONTACT_CAP, FLOOR_EXEMPT, SCENARIOS,
# STILL_FRAMES, CONTACT_S). Under the frozen rule nothing may change: the decision bank rescored for all seven models
# with the new code, every row compared with chain22's. CPU, 24 workers. Resumable: a score already written is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; L=$W/outputs/sensitivity/chain9.log
B=$W/outputs/decision_bank; R=$B/recheck; mkdir -p $R
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "recheck23 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
score() { [ -f $R/$1.json ] || cpu -m go2nav.eval.decision_bank score --bank $B/bank --model $2 --out $R/$1.json --workers 24 > $R/$1.log 2>&1
  same=$(cpu -c "import json,sys; a=json.load(open('$B/$1.json'))['rows']; b=json.load(open('$R/$1.json'))['rows']; print('identical' if a==b else f'DIFFERENT: {sum(x!=y for x,y in zip(a,b))} of {len(a)} rows, lengths {len(a)} {len(b)}')")
  log "recheck $1: $same"; }
score true true
score tracker ring_velocity
score realistic mix:noise=0.03,fresh=0.3
score dice8 dice:8
score m5 objects:$O/m5/model.pt
score n4 expect:$O/n4/model.pt
score n5 expect:$O/n5/model.pt
log "recheck23 done"
