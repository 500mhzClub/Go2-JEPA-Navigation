#!/bin/bash
# P22b (approved 14 November). Round 0 rebuilt (exploring only tapes predicted clear of the walls and the walkers);
# V0 trained on the GPU; (a) and (b). If V0 passes (a): round 1 with V0 in the loop at 3.2 s, V1 on both rounds, (a) for
# both on both rounds' validation splits and (b) for V1. The better by (a), if it passes (a) and (b), in missions (c):
# M5 with it at 3.2 s on four walkers that do not yield (seeds 0-3) and 1-2 walkers of each kind (seed 0) on the GPU
# (6 workers), M5 at 3.2 s without it beside them; on the CPU beside them (seed 0, four that do not yield, 20 workers):
# the tracker, perfect foresight and the honest oracle with it. Resumable: finished steps are skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; V=$W/outputs/value; S=$V/suites; L=$V/chain.log
O=$W/outputs/object_dynamics; B=$W/outputs/decision_bank/bank; M5=objects:$O/m5/model.pt
mkdir -p $S
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
gpu() { OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
json() { cpu -c "import json; r=json.load(open('$1')); print($2)"; }
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain24 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
build() {  # name, extra build arguments
  local name=$1; shift
  [ -f $V/$name/packed/val.npz ] && return 0
  cpu -m go2nav.data.value_missions build --out $V/$name --missions 5000 --val 500 --workers 26 "$@" > $V/$name.log 2>&1 \
    || { log "$name: build failed"; exit 1; }
  log "$name built: $(tail -1 $V/$name.log | cut -c1-300)"
  cpu -m go2nav.data.value_missions pack $V/$name --workers 26 >> $V/$name.log 2>&1 || { log "$name: pack failed"; exit 1; }
  log "$name packed: $(tail -1 $V/$name.log | cut -c1-200)"; }
train() {  # run, packed folders
  local run=$1; shift
  [ -f $V/$run/done ] && return 0
  gpu -m go2nav.training.value --data "$@" --out $V/$run --steps 20000 --batch 1024 > $V/$run.log 2>&1 \
    || { log "$run: training failed"; exit 1; }
  touch $V/$run/done
  log "$run trained: $(tail -1 $V/$run/train.log | cut -c1-420)"; }
offline() {  # run, output name, packed folders
  local run=$1 name=$2; shift 2
  [ -f $V/$run/$name.json ] || gpu -m go2nav.eval.value_measures offline --data "$@" --model $V/$run/model.pt \
    --out $V/$run/$name.json >> $V/$run.log 2>&1 || { log "$run: (a) failed to run"; exit 1; }
  log "$run (a) $name: $(json $V/$run/$name.json "json.dumps({k: r[k] for k in ('all', 'near', 'passes')})" | cut -c1-520)"; }
traps() {  # run
  [ -f $V/$1/traps.json ] || cpu -m go2nav.eval.value_measures traps --bank $B --model $V/$1/model.pt \
    --out $V/$1/traps.json --workers 11 >> $V/$1.log 2>&1 || { log "$1: (b) failed to run"; exit 1; }
  log "$1 (b): $(json $V/$1/traps.json "json.dumps({k: r[k] for k in ('timeouts', 'above_price', 'passes')})")"; }
suite() {  # runner, name, model, kind, walkers, seed, workers, value (or none)
  local runner=$1 name=$2 model=$3 kind=$4 walkers=$5 seed=$6 workers=$7 value=$8
  [ -f $S/$name/summary.json ] || $runner -m go2nav.eval.kinematic_missions run --out $S/$name --kind $kind --horizon 3.2 \
    --terminal walkers --walkers $walkers --workers $workers --seed $seed --model $model \
    $( [ "$value" != none ] && echo --value $value ) > $S/$name.log 2>&1
  log "$name: $(cpu -m go2nav.eval.kinematic_missions summary $S/$name | tr -d '\n ' | cut -c1-260)"; }

build data0
train v0 $V/data0/packed
offline v0 offline $V/data0/packed
traps v0
if [ "$(json $V/v0/offline.json "int(r['passes'])")" != 1 ]; then log "chain24: V0 fails (a); stopped (the stop rule)"; exit 0; fi
build data1 --value $V/v0/model.pt --horizon 3.2
train v1 $V/data0/packed $V/data1/packed
offline v0 both $V/data0/packed $V/data1/packed
offline v1 both $V/data0/packed $V/data1/packed
traps v1
best=$(cpu -c "import json; a, b = (json.load(open('$V/%s/both.json' % v))['all']['ratio'] for v in ('v0', 'v1')); print('v0' if a <= b else 'v1')")
ok=$(cpu -c "import json; print(int(json.load(open('$V/$best/both.json'))['passes'] is True and json.load(open('$V/$best/traps.json'))['passes'] is True))")
log "chain24: the better by (a) is $best; passes (a) and (b): $ok"
if [ "$ok" != 1 ]; then log "chain24: $best fails (a) or (b); (c) not run (the stop rule)"; exit 0; fi
VALUE=$V/$best/model.pt
(
  suite cpu value_rude4_tracker_s0 ring_velocity rude 4 0 20 $VALUE
  suite cpu value_rude4_true_s0 true rude 4 0 20 $VALUE
  suite cpu value_rude4_dice8_s0 dice:8 rude 4 0 20 $VALUE
  log "chain24: CPU suites done"
) &
for s in 0 1 2 3; do suite gpu value_rude4_m5_s$s $M5 rude 4 $s 6 $VALUE; done
for k in rude polite reactive; do suite gpu value_${k}12_m5_s0 $M5 $k "1 2" 0 6 $VALUE; done
suite gpu short_rude4_m5_s0 $M5 rude 4 0 6 none
wait
log "chain24: (c) four that do not yield, M5 with $best: $(cpu -c "
import json
rows = [json.load(open('$S/value_rude4_m5_s%d/summary.json' % s))['summary']['all'] for s in range(4)]
print('round trips', [r['round_trips'] for r in rows], 'mean', sum(r['round_trips'] for r in rows)/4, 'contact stops',
      [r['exits'].get('contact_walker', 0) for r in rows], 'mean', sum(r['exits'].get('contact_walker', 0) for r in rows)/4)")"
log "chain24 done"
