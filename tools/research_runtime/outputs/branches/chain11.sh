#!/bin/bash
# P22 item 12: the planner's rule with a cap on the chance of contact (planning.sampling CONTACT_CAP, by
# GO2NAV_CONTACT_CAP) and no still future, on four walkers that do not yield (seed 0): the true future alone, the
# true future with realistic errors (3 cm off, a third on new dice), and N4; the branch build paused (resumed after).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
systemctl --user stop branches-build; log "chain11 start; branch build paused"
export GO2NAV_W_STILL=0
for cap in 0.3 0.1 0.5; do
  c=${cap/0./}
  GO2NAV_CONTACT_CAP=$cap suite cap${c}_true mix:
  GO2NAV_CONTACT_CAP=$cap suite cap${c}_realistic mix:noise=0.03,fresh=0.3
  GO2NAV_CONTACT_CAP=$cap suite cap${c}_n4 expect:$O/n4/model.pt
done
systemctl --user start branches-build; log "branch build resumed; chain11 done"
