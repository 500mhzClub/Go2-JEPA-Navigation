#!/bin/bash
# P22 item 12, after chain9: the true future as copies a few centimetres off the truth (a learned model's error), on
# four walkers that do not yield (seed 0), with the branch build paused (resumed after).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; S=$W/outputs/sensitivity; L=$S/chain9.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $S/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $S/$1 --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers 28 --seed 0 --model $2 > $S/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $S/$1 | tr -d '\n ' | cut -c1-260)"; }
until ! systemctl --user is-active --quiet sensitivity-chain9; do sleep 30; done
systemctl --user stop branches-build; log "chain10 start; branch build paused"
suite mix_noise3 mix:noise=0.03
suite mix_noise1 mix:noise=0.01
suite mix_noise3_still10 mix:noise=0.03,still=0.1
systemctl --user start branches-build; log "branch build resumed; chain10 done"
