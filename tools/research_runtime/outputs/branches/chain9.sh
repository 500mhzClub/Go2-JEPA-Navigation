#!/bin/bash
# P22 item 12 (approved 13 November): with the branch build paused (resumed after), traces of N4's contacts and
# timeouts on four walkers that do not yield (seed 0), then the planner's sensitivity: the true future beside set
# weights of false ones (planning.mixture), and N4 with less or no still future, on the same suite.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
mkdir -p $S
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
until ! systemctl --user is-active --quiet branches-chain8; do sleep 30; done
systemctl --user stop branches-build; log "start; branch build paused"
[ -f $S/traces_n4/summary.json ] || run -m go2nav.eval.mission_traces run --model expect:$O/n4/model.pt --from $O/rude4_n4expect_s0 --out $S/traces_n4 --workers 28 > $S/traces_n4.log 2>&1
log "traces n4: $(tr -d '\n ' < $S/traces_n4/summary.json)"
suite mix_true mix:
suite mix_still05 mix:still=0.05
suite mix_still10 mix:still=0.1
suite mix_still20 mix:still=0.2
suite mix_ignore05 mix:ignore=0.05
suite mix_ignore10 mix:ignore=0.1
suite mix_ignore20 mix:ignore=0.2
suite mix_fresh50 mix:fresh=0.5
suite mix_fresh50_still10 mix:fresh=0.5,still=0.1
GO2NAV_W_STILL=0 suite n4expect_still0 expect:$O/n4/model.pt
GO2NAV_W_STILL=0.03 suite n4expect_still03 expect:$O/n4/model.pt
systemctl --user start branches-build; log "branch build resumed; chain9 done"
