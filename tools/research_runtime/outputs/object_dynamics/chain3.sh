#!/bin/bash
# P22 (c), 12 November: the planner on four walkers that do not yield (40 missions), the object model's inference on
# the CPU pool (12 GPU workers livelocked overnight: the known limit is 6). M2 and M4 on seed 0 side by side; a model
# goes on to seeds 1-3 only if seed 0 beats the tracker (13 round trips, 16 contact stops); then the true model on
# seeds 1-3; then M5 and M6's measures once they have trained. Resumable. Log: chain3.log.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/chain3.log; }
cd $W
suite() {  # name, model, seed, workers
  [ -f $O/$1/summary.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$1 \
      --kind rude --horizon 6.4 --terminal walkers --walkers 4 --workers $4 --seed $3 --model $2 > $O/$1.log 2>&1
  log "$1: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/$1 | cut -c1-150)"
}
better() {  # name of a finished suite: beats the tracker's seed 0?
  $PY -c "
import json, sys; s=json.load(open('$O/$1/summary.json'))['summary']['all']
sys.exit(0 if s['round_trips'] >= 15 or s['exits'].get('contact_walker', 0) <= 10 else 1)"
}
log "start $(git rev-parse --short HEAD)"
suite rude4_m2_s0 objects:$O/m2/model.pt 0 14 &
suite rude4_m4_s0 objects:$O/m4/model.pt 0 14 &
wait
for r in m2 m4; do
  if better rude4_${r}_s0; then
    for seed in 1 2 3; do suite rude4_${r}_s$seed objects:$O/$r/model.pt $seed 28; done
  else log "$r: no better than the tracker on seed 0; seeds 1-3 not run"; fi
done
for seed in 0 1 2 3; do suite rude4_true_s$seed true $seed 28; done
until [ -f $O/m5/model.pt ] && [ -f $O/m6/model.pt ] && ! systemctl --user is-active --quiet objects-m5 && ! systemctl --user is-active --quiet objects-m6; do sleep 60; done
for r in m5 m6; do
  [ -f $O/${r}_offline.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures offline \
      --model $O/$r/model.pt --out $O/${r}_offline.json --workers 26 > $O/${r}_offline.log 2>&1
  [ -f $O/${r}_reaction.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures reaction \
      --model $O/$r/model.pt --out $O/${r}_reaction.json --workers 26 > $O/${r}_reaction.log 2>&1
  log "$r: $($PY -c "
import json; d=json.load(open('$O/${r}_offline.json')); e=json.load(open('$O/${r}_reaction.json'))
print('offline all', ' '.join(f\"{h} {d[h+'_all']['model']:.3f}/{d[h+'_all']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; waiting', ' '.join(f\"{h} {d[h+'_waiting']['model']:.3f}/{d[h+'_waiting']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; reaction', ' '.join(f\"{h} {e[h]['slope']} {e[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
done
log "chain3 done"
