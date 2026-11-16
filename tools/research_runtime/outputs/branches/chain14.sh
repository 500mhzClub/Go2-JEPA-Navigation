#!/bin/bash
# P22 item 12, after chain13: why N4 hits walkers under the scenario rule. Traces of its contact missions (16 draws,
# no still future: the price recorded is the mean breach cost over the draws, about 15,000 x the share that breach),
# then N4 with 64 draws (is 16 too few to see its rarer futures?); the branch build paused (resumed after).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
until ! systemctl --user is-active --quiet sensitivity-chain13; do sleep 30; done
systemctl --user stop branches-build; log "chain14 start; branch build paused"
export GO2NAV_W_STILL=0
[ -f $S/traces_scen_n4/summary.json ] || GO2NAV_SCENARIOS=16 run -m go2nav.eval.mission_traces run --model expect:$O/n4/model.pt --from $S/scen_n4 --out $S/traces_scen_n4 --exits contact_walker --workers 28 > $S/traces_scen_n4.log 2>&1
log "traces scen_n4: $(tr -d '\n ' < $S/traces_scen_n4/summary.json)"
GO2NAV_SCENARIOS=64 suite scen64_n4 expect:$O/n4/model.pt
systemctl --user start branches-build; log "branch build resumed; chain14 done"
