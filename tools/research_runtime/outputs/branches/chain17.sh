#!/bin/bash
# P22 item 13 (approved 13 November): under the worked-out rule with the way out and a walker contact at 300 s
# (GO2NAV_EXACT=1, GO2NAV_RECOVER=1, GO2NAV_WALKER_PRICE=300; no still future), on four walkers that do not yield:
# 1. the mission bar again: the ring tracker on seeds 0-3, N4 on seeds 1-3 (seed 0 is price300_n4);
# 2. contact calibration at plans near walkers (eval.contact_calibration), seed 0: N4, the ring tracker, realistic errors;
# 3. the ceilings: perfect predictions on seeds 0-3, realistic errors on seeds 1-3 (seed 0 is price300_realistic).
# The branch build (a transient unit: stopping deletes it) is relaunched after. Resumable: a finished suite is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity
C=$W/outputs/calibration; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W; mkdir -p $C
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
calib() { [ -f $C/$1/summary.json ] || run -m go2nav.eval.contact_calibration run --out $C/$1 --model $2 --workers 28 > $C/$1.log 2>&1
  log "$1: $(run -c "import json; s=json.load(open('$C/$1/summary.json')); print(json.dumps({k: s[k] for k in ('round_trips', 'contacts', 'probes', 'tapes', 'truly_within_share', 'auc', 'brier', 'brier_of_base_rate', 'misses', 'false_alarms', 'chosen_truly_within', 'mean_regret_s', 'wrong_choices', 'causes', 'avoidable_misses')}))")"; }
systemctl --user stop branches-build 2>/dev/null; log "chain17 start; branch build stopped at $(ls /mnt/workspace_drive/go2nav_ring/corpus_v1/branches/train | wc -l) files"
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
for seed in 0 1 2 3; do suite rebase_tracker_s$seed ring_velocity $seed; done
for seed in 1 2 3; do suite rebase_n4_s$seed expect:$O/n4/model.pt $seed; done
calib calib_n4 expect:$O/n4/model.pt
calib calib_tracker ring_velocity
calib calib_realistic mix:noise=0.03,fresh=0.3
for seed in 0 1 2 3; do suite rebase_true_s$seed mix: $seed; done
for seed in 1 2 3; do suite rebase_realistic_s$seed mix:noise=0.03,fresh=0.3 $seed; done
if systemd-run --user --unit=branches-build --working-directory=$W $W/outputs/branches/build.sh; then log "branch build relaunched; chain17 done"
else log "branch build relaunch FAILED; chain17 done"; fi
