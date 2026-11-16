#!/bin/bash
# P22 item 14 (approved 14 November; docs/REVIEW_2026-11-14.md): M5, the one-future correction to constant velocity
# that passes (a) and has never run in a mission, under the frozen rule (worked out, the way out, a walker contact at
# 300 s, no still future): four walkers that do not yield, seeds 0-3, then 1-2 walkers not yielding, polite and
# reactive, seed 0. GPU inference (0.14 s a plan, against 1.2 s on one core), at most 6 mission workers. Beside it on
# the CPU, once the oracle's four suites are in, the tracker and the true model on the same three suites under the same
# rule. chain20's calibration of dice:8 is stopped there: the decision bank scores dice:8 on fixed states instead.
# Resumable: finished suites are skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
gpu() { OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() {  # runner, name, model, kind, walkers, seed, workers
  local runner=$1 name=$2 model=$3 kind=$4 walkers=$5 seed=$6 workers=$7
  [ -f $S/$name/summary.json ] || $runner -m go2nav.eval.kinematic_missions run --out $S/$name --kind $kind --horizon 6.4 \
      --terminal walkers --walkers $walkers --workers $workers --seed $seed --model $model > $S/$name.log 2>&1
  log "$name: $(cpu -m go2nav.eval.kinematic_missions summary $S/$name | tr -d '\n ' | cut -c1-260)"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
M5=objects:$W/outputs/object_dynamics/m5/model.pt
log "chain21 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
(
  for seed in 0 1 2 3; do suite gpu rebase_m5_s$seed $M5 rude 4 $seed 6; done
  for kind in rude polite reactive; do suite gpu frozen_${kind}12_m5_s0 $M5 $kind "1 2" 0 6; done
  log "chain21: M5 done"
) &
(
  sleep 240; log "chain21 GPU after start: $(rocm-smi --showmemuse 2>/dev/null | grep -m1 'VRAM%' | tr -s ' ') $(rocm-smi --showuse 2>/dev/null | grep -m1 'GPU use' | tr -s ' ')"
) &
(
  until [ "$(ls $S/rebase_dice8_s{0,1,2,3}/summary.json 2>/dev/null | wc -l)" -eq 4 ]; do sleep 60; done
  sleep 120                                   # chain20 logs each suite's line just after its summary
  systemctl --user stop dice-chain20 2>/dev/null
  log "chain21: dice-chain20 stopped before its calibration (the decision bank scores dice:8)"
  for model in ring_velocity true; do
    for kind in rude polite reactive; do suite cpu frozen_${kind}12_${model}_s0 $model $kind "1 2" 0 20; done
  done
  log "chain21: tracker and true model done"
) &
wait
log "chain21 done"
