#!/bin/bash
# P22b item 8 (approved 14 November, after V0): the branches (data.value_branches: 2,000 training and 300 validation
# missions of round 0, two moments each, six tapes run for 3.2 s and then under the frozen rule); V2 trained on the GPU
# on the branches, round 0's moments and consistency along them; then in order (a') ranking, (a) offline, (b) traps,
# each a bar. If all pass, missions (c) as chain24 set them out: M5 with V2 at 3.2 s on four walkers that do not yield
# (seeds 0-3) and 1-2 walkers of each kind (seed 0) on the GPU (6 workers), M5 at 3.2 s without it beside them; on the
# CPU beside them (seed 0, 20 workers): the tracker, perfect foresight and the honest oracle with V2. Resumable.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; V=$W/outputs/value; S=$V/suites; L=$V/chain.log
O=$W/outputs/object_dynamics; B=$W/outputs/decision_bank/bank; M5=objects:$O/m5/model.pt; R=$V/v2
mkdir -p $S
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
gpu() { OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
json() { cpu -c "import json; r=json.load(open('$1')); print($2)"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain25 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
if [ ! -f $V/branches/packed/val.npz ]; then
  cpu -m go2nav.data.value_branches build --data $V/data0 --out $V/branches --train 2000 --val 300 --workers 26 \
    > $V/branches.log 2>&1 || { log "branches: build failed: $(tail -2 $V/branches.log | cut -c1-300)"; exit 1; }
  log "branches built: $(tail -1 $V/branches.log | cut -c1-300)"
  cpu -m go2nav.data.value_branches pack $V/branches --workers 26 >> $V/branches.log 2>&1 || { log "branches: pack failed"; exit 1; }
  log "branches packed: $(tail -1 $V/branches.log | cut -c1-300)"
fi
if [ ! -f $R/done ]; then
  gpu -m go2nav.training.value --data $V/data0/packed --branches $V/branches/packed --out $R --steps 8000 --batch 1024 \
    > $V/v2.log 2>&1 || { log "v2: training failed: $(tail -2 $V/v2.log | cut -c1-300)"; exit 1; }
  touch $R/done
  log "v2 trained: $(tail -1 $R/train.log | cut -c1-700)"
fi
[ -f $R/ranking.json ] || gpu -m go2nav.eval.value_measures ranking --branches $V/branches/packed --model $R/model.pt \
  --out $R/ranking.json >> $V/v2.log 2>&1 || { log "v2: (a') failed to run"; exit 1; }
log "v2 (a') ranking: $(json $R/ranking.json "json.dumps({k: v for k, v in r.items() if k not in ('model', 'branches')})")"
[ -f $R/offline.json ] || gpu -m go2nav.eval.value_measures offline --data $V/data0/packed --model $R/model.pt \
  --out $R/offline.json >> $V/v2.log 2>&1 || { log "v2: (a) failed to run"; exit 1; }
log "v2 (a) offline: $(json $R/offline.json "json.dumps({k: r[k] for k in ('all', 'near', 'passes')})" | cut -c1-520)"
[ -f $R/traps.json ] || cpu -m go2nav.eval.value_measures traps --bank $B --model $R/model.pt --out $R/traps.json \
  --workers 11 >> $V/v2.log 2>&1 || { log "v2: (b) failed to run"; exit 1; }
log "v2 (b) traps: $(json $R/traps.json "json.dumps({k: r[k] for k in ('timeouts', 'above_price', 'passes')})")"
ok=$(cpu -c "import json; print(int(all(json.load(open('$R/%s.json' % m))['passes'] is True for m in ('ranking', 'offline', 'traps'))))")
if [ "$ok" != 1 ]; then log "chain25: V2 fails (a'), (a) or (b); (c) not run (the stop rule)"; exit 0; fi
suite() {  # runner, name, model, kind, walkers, seed, workers, value (or none)
  local runner=$1 name=$2 model=$3 kind=$4 walkers=$5 seed=$6 workers=$7 value=$8
  [ -f $S/$name/summary.json ] || $runner -m go2nav.eval.kinematic_missions run --out $S/$name --kind $kind --horizon 3.2 \
    --terminal walkers --walkers $walkers --workers $workers --seed $seed --model $model \
    $( [ "$value" != none ] && echo --value $value ) > $S/$name.log 2>&1
  log "$name: $(cpu -m go2nav.eval.kinematic_missions summary $S/$name | tr -d '\n ' | cut -c1-260)"; }
VALUE=$R/model.pt
(
  suite cpu v2_rude4_tracker_s0 ring_velocity rude 4 0 20 $VALUE
  suite cpu v2_rude4_true_s0 true rude 4 0 20 $VALUE
  suite cpu v2_rude4_dice8_s0 dice:8 rude 4 0 20 $VALUE
  log "chain25: CPU suites done"
) &
for s in 0 1 2 3; do suite gpu v2_rude4_m5_s$s $M5 rude 4 $s 6 $VALUE; done
for k in rude polite reactive; do suite gpu v2_${k}12_m5_s0 $M5 $k "1 2" 0 6 $VALUE; done
suite gpu short_rude4_m5_s0 $M5 rude 4 0 6 none
wait
log "chain25: (c) four that do not yield, M5 with V2: $(cpu -c "
import json
rows = [json.load(open('$S/v2_rude4_m5_s%d/summary.json' % s))['summary']['all'] for s in range(4)]
print('round trips', [r['round_trips'] for r in rows], 'mean', sum(r['round_trips'] for r in rows)/4, 'contact stops',
      [r['exits'].get('contact_walker', 0) for r in rows], 'mean', sum(r['exits'].get('contact_walker', 0) for r in rows)/4)")"
log "chain25 done"
