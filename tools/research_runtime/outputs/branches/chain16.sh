#!/bin/bash
# P22 item 12: the price of a walker contact under the worked-out rule (GO2NAV_EXACT=1, GO2NAV_RECOVER=1,
# GO2NAV_WALKER_PRICE seconds, times 2 less the share of the horizon gone; walls keep DISCARD), for the true future with
# realistic errors and for N4, on four walkers that do not yield (seed 0); no still future. The branch build is
# stopped (a transient unit: stopping deletes it) and relaunched after.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
systemctl --user stop branches-build; log "chain16 start; branch build stopped at $(ls /mnt/workspace_drive/go2nav_ring/corpus_v1/branches/train | wc -l) files"
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0
for price in 1000 300 100 3000; do
  GO2NAV_WALKER_PRICE=$price suite price${price}_realistic mix:noise=0.03,fresh=0.3
  GO2NAV_WALKER_PRICE=$price suite price${price}_n4 expect:$O/n4/model.pt
done
if systemd-run --user --unit=branches-build --working-directory=$W $W/outputs/branches/build.sh; then log "branch build relaunched; chain16 done"
else log "branch build relaunch FAILED; chain16 done"; fi
