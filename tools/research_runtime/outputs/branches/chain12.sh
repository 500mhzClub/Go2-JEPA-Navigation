#!/bin/bash
# P22 item 12, after chain11: the planner's rule with a walker already under the floor a breach, as the true
# model's rule has it (GO2NAV_FLOOR_EXEMPT=0), priced and capped at 0.3, no still future, on four walkers that do
# not yield (seed 0): the true future, with realistic errors, and N4; the branch build paused (resumed after).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
until ! systemctl --user is-active --quiet sensitivity-chain11; do sleep 30; done
systemctl --user stop branches-build; log "chain12 start; branch build paused"
export GO2NAV_W_STILL=0 GO2NAV_FLOOR_EXEMPT=0
suite floor_priced_true mix:
GO2NAV_CONTACT_CAP=0.3 suite floor_cap3_true mix:
GO2NAV_CONTACT_CAP=0.3 suite floor_cap3_realistic mix:noise=0.03,fresh=0.3
GO2NAV_CONTACT_CAP=0.3 suite floor_cap3_n4 expect:$O/n4/model.pt
systemctl --user start branches-build; log "branch build resumed; chain12 done"
