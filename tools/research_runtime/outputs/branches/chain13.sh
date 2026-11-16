#!/bin/bash
# P22 item 12, after chain12: several futures through the true model's rule (planning.sampling scenario_costs, 16 drawn
# joint futures, GO2NAV_SCENARIOS), no still future unless set, on four walkers that do not yield (seed 0): the true
# future, with realistic errors, 3 cm off, N4, N3, and N4 with its still future; the branch build paused (resumed after).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
until ! systemctl --user is-active --quiet sensitivity-chain12; do sleep 30; done
systemctl --user stop branches-build; log "chain13 start; branch build paused"
export GO2NAV_SCENARIOS=16 GO2NAV_W_STILL=0
suite scen_true mix:
suite scen_realistic mix:noise=0.03,fresh=0.3
suite scen_noise3 mix:noise=0.03
suite scen_n4 expect:$O/n4/model.pt
suite scen_n3 expect:$O/n3/model.pt
GO2NAV_W_STILL=0.1 suite scen_n4_still10 expect:$O/n4/model.pt
systemctl --user start branches-build; log "branch build resumed; chain13 done"
