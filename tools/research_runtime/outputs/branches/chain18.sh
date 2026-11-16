#!/bin/bash
# P22 item 13, after chain17: contact calibration again with each wrong choice's cost error split by term (time,
# clearance, progress, contact; eval.contact_calibration), the same rule (worked out, the way out, a walker contact at
# 300 s), four walkers that do not yield, seed 0: N4, the ring tracker, realistic errors. Waits for chain17; stops the
# branch build it relaunches (a transient unit: stopping deletes it) and relaunches it after.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity
C=$W/outputs/calibration; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
while systemctl --user is-active --quiet sensitivity-chain17; do sleep 60; done
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
calib() { [ -f $C/$1/summary.json ] || run -m go2nav.eval.contact_calibration run --out $C/$1 --model $2 --workers 28 > $C/$1.log 2>&1
  log "$1: $(run -c "import json; s=json.load(open('$C/$1/summary.json')); print(json.dumps({k: s.get(k) for k in ('round_trips', 'contacts', 'probes', 'wrong_choices', 'causes', 'wrong_choice_error_by_term', 'wrong_choice_largest_term')}))")"; }
systemctl --user stop branches-build 2>/dev/null; log "chain18 start; branch build stopped at $(ls /mnt/workspace_drive/go2nav_ring/corpus_v1/branches/train | wc -l) files"
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
calib terms_n4 expect:$O/n4/model.pt
calib terms_tracker ring_velocity
calib terms_realistic mix:noise=0.03,fresh=0.3
if systemd-run --user --unit=branches-build --working-directory=$W $W/outputs/branches/build.sh; then log "branch build relaunched; chain18 done"
else log "branch build relaunch FAILED; chain18 done"; fi
