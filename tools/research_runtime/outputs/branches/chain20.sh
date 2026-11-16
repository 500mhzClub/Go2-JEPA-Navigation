#!/bin/bash
# P22 item 13 (approved 14 November): the honest-uncertainty oracle. dice:8 (planning.mixture.DiceModel: each walker's
# true state, kind and rules under eight new sets of dice a plan, at equal weight, never the world's) under the
# worked-out rule (way out, a walker contact at 300 s, no still future) on four walkers that do not yield: seeds 0-3 at
# once (7 workers each: a plan takes 0.77 s, so a mission that times out takes about 15 minutes), then its contact
# calibration with the cost split by term (seed 0, 28 workers). If it passes the ring tracker's 16.25 round trips the
# model is the gap; if it freezes as N4 does the rule is. Resumable: finished suites are skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; S=$W/outputs/sensitivity; C=$W/outputs/calibration; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers $4 --seed $3 --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
calib() { [ -f $C/$1/summary.json ] || run -m go2nav.eval.contact_calibration run --out $C/$1 --model $2 --workers 28 > $C/$1.log 2>&1
  log "$1: $(run -c "import json; s=json.load(open('$C/$1/summary.json')); print(json.dumps({k: s.get(k) for k in ('round_trips', 'contacts', 'probes', 'auc', 'brier', 'misses', 'false_alarms', 'chosen_truly_within', 'mean_regret_s', 'wrong_choices', 'causes', 'wrong_choice_error_by_term', 'wrong_choice_largest_term')}))")"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain20 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
for seed in 0 1 2 3; do suite rebase_dice8_s$seed dice:8 $seed 7 & done
wait
calib terms_dice8 dice:8
log "chain20 done"
