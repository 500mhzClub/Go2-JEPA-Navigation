#!/bin/bash
# P22 item 8: once N0 and N1 have trained (GPU): (a) and (b) on the CPU, then the planner suites (the model on each
# mission's own core): four walkers that do not yield on seeds 0-3, then 1-2 walkers that do not yield and the
# reactive suite on seed 0, then the true model on seeds 1-3. Resumable. Log: chain5.log.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/chain5.log; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() {  # name, model, kind, walkers (quoted), seed
  [ -f $O/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $O/$1 --kind $3 --horizon 6.4 --terminal walkers \
      --walkers $4 --workers 28 --seed $5 --model $2 > $O/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $O/$1 | cut -c1-150)"
}
for r in n1 n0; do
  until [ -f $O/$r/model.pt ] && ! systemctl --user is-active --quiet objects-$r; do sleep 30; done
  log "$r trained: $(tail -1 $O/$r/validation.jsonl | cut -c1-420)"
  [ -f $O/${r}_offline.json ] || run -m go2nav.eval.object_measures offline --model modes:$O/$r/model.pt --out $O/${r}_offline.json --workers 26 > $O/${r}_offline.log 2>&1
  [ -f $O/${r}_reaction.json ] || run -m go2nav.eval.object_measures reaction --model modes:$O/$r/model.pt --out $O/${r}_reaction.json --workers 26 > $O/${r}_reaction.log 2>&1
  log "$r: $($PY -c "
import json; d=json.load(open('$O/${r}_offline.json')); e=json.load(open('$O/${r}_reaction.json'))
print('offline all', ' '.join(f\"{h} {d[h+'_all']['model']:.3f}/{d[h+'_all']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; waiting', ' '.join(f\"{h} {d[h+'_waiting']['model']:.3f}/{d[h+'_waiting']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; reaction', ' '.join(f\"{h} {e[h]['slope']} {e[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
done
for seed in 0 1 2 3; do for r in n0 n1; do suite rude4_${r}_s$seed modes:$O/$r/model.pt rude 4 $seed; done; done
for r in n0 n1; do suite rude12_${r}_s0 modes:$O/$r/model.pt rude "1 2" 0; suite reactive12_${r}_s0 modes:$O/$r/model.pt reactive "1 2" 0; done
for seed in 1 2 3; do suite rude4_true_s$seed true rude 4 $seed; done
log "chain5 done"
