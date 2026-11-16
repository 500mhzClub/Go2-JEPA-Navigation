#!/bin/bash
# P22 item 12: the true model's rule in expectation over every future, worked out (planning.sampling expected_costs,
# GO2NAV_EXACT=1), with and without the way out from under the floor (GO2NAV_RECOVER), no still future, on four
# walkers that do not yield (seed 0). The branch build (a transient unit: stopping deletes it) is relaunched after.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
systemctl --user stop branches-build; log "chain15 start; branch build stopped at $(ls /mnt/workspace_drive/go2nav_ring/corpus_v1/branches/train | wc -l) files"
export GO2NAV_EXACT=1 GO2NAV_W_STILL=0
GO2NAV_RECOVER=0 suite exact_true mix:
GO2NAV_RECOVER=1 suite exact_rec_n4 expect:$O/n4/model.pt
GO2NAV_RECOVER=1 suite exact_rec_true mix:
GO2NAV_RECOVER=0 suite exact_n4 expect:$O/n4/model.pt
GO2NAV_RECOVER=1 suite exact_rec_realistic mix:noise=0.03,fresh=0.3
GO2NAV_RECOVER=1 suite exact_rec_n3 expect:$O/n3/model.pt
if systemd-run --user --unit=branches-build --working-directory=$W $W/outputs/branches/build.sh; then log "branch build relaunched; chain15 done"
else log "branch build relaunch FAILED; chain15 done"; fi
