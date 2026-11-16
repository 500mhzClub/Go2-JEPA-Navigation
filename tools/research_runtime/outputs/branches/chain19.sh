#!/bin/bash
# P22 item 13 (approved 13 November): N5.
# 1. Windows branched from the states N4's planner reaches (data.planner_branches: 1,000 train and 100 val missions on
#    corpus layouts, four walkers of a drawn kind, at most 40 points of 6 tapes a mission; CPU, 28 workers), after the
#    random-tape branch build finishes.
# 2. N5: N4 fine-tuned on them (configs/object/n5.yaml; GPU), after P23's s2 frees the card.
# 3. N5 under the worked-out rule (way out, a walker contact at 300 s, no still future) on four walkers that do not
#    yield: its contact calibration with the cost split by term (seed 0), then seeds 0-3.
# Resumable: finished missions, the trained model and finished suites are skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; S=$W/outputs/sensitivity
C=$W/outputs/calibration; L=$S/chain9.log
CORPUS=/mnt/workspace_drive/go2nav_ring/corpus_v1; MAZES=/mnt/workspace_drive/go2nav_lewm_v2/mazes
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed ${3:-0} --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
calib() { [ -f $C/$1/summary.json ] || run -m go2nav.eval.contact_calibration run --out $C/$1 --model $2 --workers 28 > $C/$1.log 2>&1
  log "$1: $(run -c "import json; s=json.load(open('$C/$1/summary.json')); print(json.dumps({k: s.get(k) for k in ('round_trips', 'contacts', 'probes', 'auc', 'brier', 'misses', 'false_alarms', 'chosen_truly_within', 'wrong_choices', 'causes', 'wrong_choice_error_by_term', 'wrong_choice_largest_term')}))")"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
while systemctl --user is-active --quiet branches-build; do sleep 60; done
log "chain19 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files); random-tape branches: $(ls $CORPUS/branches/train | wc -l) train episodes"
if [ ! -f $CORPUS/planner_branches/summary.json ]; then
  run -m go2nav.data.planner_branches build $CORPUS --mazes-root $MAZES --model expect:$O/n4/model.pt --workers 28 \
      > $W/outputs/branches/planner_build.log 2>&1
  code=$?; log "planner branches exit $code: $(tr -d '\n ' < $CORPUS/planner_branches/summary.json 2>/dev/null | cut -c1-400)"
  [ $code -eq 0 ] || exit $code
fi
while systemctl --user is-active --quiet slots-s2; do sleep 60; done
if [ ! -f $O/n5/model.pt ]; then
  log "N5 training start (GPU: $(rocm-smi --showmemuse 2>/dev/null | grep -m1 'VRAM%' | tr -s ' '))"
  outputs/object_dynamics/train_modes.sh n5 &
  sleep 240; log "GPU after start: $(rocm-smi --showmemuse 2>/dev/null | grep -m1 'VRAM%' | tr -s ' ') $(rocm-smi --showuse 2>/dev/null | grep -m1 'GPU use' | tr -s ' ')"
  wait
fi
[ -f $O/n5/model.pt ] || { log "N5 did not finish"; exit 1; }
log "N5 trained: $(cat $O/n5/planner_branches.json | tr -d '\n') $(tail -1 $O/n5/validation.jsonl | cut -c1-900)"
calib terms_n5 expect:$O/n5/model.pt
for seed in 0 1 2 3; do suite rebase_n5_s$seed expect:$O/n5/model.pt $seed; done
log "chain19 done"
