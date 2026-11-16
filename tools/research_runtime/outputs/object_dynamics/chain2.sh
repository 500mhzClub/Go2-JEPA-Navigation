#!/bin/bash
# P22 revised, the night of 11 November. In order: (1) once M0 has trained, the slot trainer's smoke alone on the GPU;
# (2) M2, M3 and M4 train on the GPU while M0 and M1's offline measure runs on the CPU; (3) each new run's offline
# measure (a) and its reaction (b); (4) the planner on four walkers that do not yield, four seeds, for the run best by
# (a) (and the tracker and the true model on the same seeds, CPU); (5) the slot model's training alone on the GPU, if
# its smoke ran clean. All training on the GPU. Resumable steps; log: outputs/object_dynamics/chain2.log.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; SL=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots
O=$W/outputs/object_dynamics; C=$W/outputs/camera_slots; F=/mnt/workspace_drive/go2nav_dino/features_vits14
R=/mnt/workspace_drive/go2nav_ring/corpus_v1
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/chain2.log; }
cd $W
until [ -f $O/m0/model.pt ] && ! systemctl --user is-active --quiet objects-m0; do sleep 30; done
log "M0 trained ($(git rev-parse --short HEAD))"

# (1) the slot smoke, alone on the GPU
start=$(date '+%Y-%m-%d %H:%M:%S')
( cd $SL; rm -rf $C/smoke_alone
  PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/smoke_alone --set optimiser.steps=800 \
      --set runtime.log_every=100 --set runtime.validate_every=400 --set runtime.validation_clips=128 > $C/smoke_alone.log 2>&1 )
smoke=$?; nf=$(journalctl -k -q --since "$start" 2>/dev/null | grep -c "page fault")
log "slot smoke alone: exit $smoke, GPU page faults $nf, steps logged $(wc -l < $C/smoke_alone/train.jsonl 2>/dev/null)"

# (2) the three runs train on the GPU; M0 and M1 are measured on the CPU meanwhile
for r in m2 m3 m4; do
  ( PYTHONPATH=. $PY -W ignore -m go2nav.training.object_jepa train --corpus $R --out $O/$r --config configs/object/$r.yaml \
        >> $O/$r.log 2>&1; log "$r trained: exit $?" ) &
done
offline() {  # run: measure (a), and print PASS or FAIL by the revised bars
  [ -f $O/$1_offline.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures offline \
      --model $O/$1/model.pt --out $O/$1_offline.json --workers 22 > $O/$1_offline.log 2>&1
  $PY -c "
import json; d=json.load(open('$O/$1_offline.json'))
ok=all(d[f'1.6s_{s}']['model'] <= d[f'1.6s_{s}']['velocity'] and d[f'3.2s_{s}']['model'] <= .5*d[f'3.2s_{s}']['velocity']
       and d[f'6.4s_{s}']['model'] <= .5*d[f'6.4s_{s}']['velocity'] for s in ('all', 'waiting'))
row=lambda s: ' '.join(f\"{h} {d[f'{h}_{s}']['model']:.3f}/{d[f'{h}_{s}']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s'))
print(('PASS' if ok else 'FAIL')+' model/velocity all: '+row('all')+' ; waiting: '+row('waiting'))"
}
for r in m0 m1; do log "$r offline (a): $(offline $r)"; done
wait
log "M2, M3 and M4 trained"

# (3) the new runs: (a), then the reaction (b) for each (reported; ten minutes on the CPU)
best=""; best_error=9
for r in m2 m3 m4; do
  [ -f $O/$r/model.pt ] || { log "$r has no model"; continue; }
  verdict=$(offline $r); log "$r offline (a): $verdict"
  [ -f $O/${r}_reaction.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures reaction \
      --model $O/$r/model.pt --out $O/${r}_reaction.json --workers 22 > $O/${r}_reaction.log 2>&1
  log "$r reaction (b): $($PY -c "
import json; d=json.load(open('$O/${r}_reaction.json'))
print(d['windows'], 'windows;', ' '.join(f\"{h}: slope {d[h]['slope']} {d[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
  e=$($PY -c "import json; d=json.load(open('$O/${r}_offline.json')); print(d['1.6s_all']['model']+d['3.2s_all']['model'])")
  if $PY -c "import sys; sys.exit(0 if $e < $best_error else 1)"; then best=$r; best_error=$e; fi
done
log "best by (a): $best"

# (4) the planner: four walkers that do not yield, four seeds. The references on the CPU, then the best run (GPU).
suite() {  # name, model, seed
  [ -f $O/$1/summary.json ] || OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$1 --kind rude \
      --horizon 6.4 --terminal walkers --walkers 4 --workers 12 --seed $3 --model $2 > $O/$1.log 2>&1
  log "$1: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/$1 | cut -c1-150)"
}
for seed in 0 1 2 3; do HIP_VISIBLE_DEVICES= suite rude4_ring_velocity_s$seed ring_velocity $seed; done
if [ -n "$best" ]; then
  for seed in 0 1 2 3; do suite rude4_${best}_s$seed objects:$O/$best/model.pt $seed; done
fi
for seed in 0 1 2 3; do HIP_VISIBLE_DEVICES= suite rude4_true_s$seed true $seed; done

# (5) the slot model, alone on the GPU, if its smoke ran clean
if [ $smoke -eq 0 ] && [ $nf -eq 0 ]; then
  log "slot model: training (60,000 steps)"
  ( cd $SL; PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/s0 >> $C/s0.log 2>&1 )
  log "slot model s0: exit $?"
  [ -f $C/s0/model.pt ] && ( cd $SL; PYTHONPATH=$SL $PY -W ignore -m go2nav.eval.slot_measures score --model $C/s0/model.pt --features $F \
      --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $C/s0_binding.json > $C/s0_binding.log 2>&1; log "slot binding: $(tr -d '\n ' < $C/s0_binding.json | cut -c1-400)" )
else
  log "slot model: not trained (its smoke faulted alone: exit $smoke, page faults $nf)"
fi
log "chain done"
