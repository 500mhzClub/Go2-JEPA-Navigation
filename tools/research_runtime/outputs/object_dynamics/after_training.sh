#!/bin/bash
# After P22's M0 and M1 train: (1) the slot trainer's smoke alone on the GPU; (2) P22 (a) offline for both (CPU);
# (3) only for a run that passes (a): (b) the reaction and (c) the planner on four seeds (GPU); (4) the slot model's
# full training if its smoke ran clean alone. Resumable steps; logs in outputs/object_dynamics/after.log.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; SL=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots
O=$W/outputs/object_dynamics; C=$W/outputs/camera_slots; F=/mnt/workspace_drive/go2nav_dino/features_vits14
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/after.log; }
until [ -f $O/m0/model.pt ] && [ -f $O/m1/model.pt ] && ! systemctl --user is-active --quiet objects-m0 && ! systemctl --user is-active --quiet objects-m1; do sleep 60; done
log "M0 and M1 trained"
faults() { journalctl -k -q --since "$1" 2>/dev/null | grep -c "page fault"; }
start=$(date '+%Y-%m-%d %H:%M:%S')
cd $SL; rm -rf $C/smoke_alone
PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/smoke_alone --set optimiser.steps=800 \
    --set runtime.log_every=100 --set runtime.validate_every=400 --set runtime.validation_clips=128 > $C/smoke_alone.log 2>&1
smoke=$?; nf=$(faults "$start"); log "slot smoke alone: exit $smoke, page faults $nf"
cd $W
for r in m0 m1; do
  [ -f $O/${r}_offline.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures offline \
      --model $O/$r/model.pt --out $O/${r}_offline.json --workers 24 > $O/${r}_offline.log 2>&1
  pass=$($PY -c "
import json; d=json.load(open('$O/${r}_offline.json'))
ok=all(d[f'{h}_{s}']['model'] < d[f'{h}_{s}']['velocity'] for h in ('1.6s','3.2s') for s in ('all','waiting'))
print('PASS' if ok else 'FAIL')")
  log "$r offline (a): $pass :: $(cat $O/${r}_offline.json | tr -d '\n ' | cut -c1-600)"
  if [ "$pass" = PASS ]; then
    [ -f $O/${r}_reaction.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures reaction \
        --model $O/$r/model.pt --out $O/${r}_reaction.json --workers 24 > $O/${r}_reaction.log 2>&1
    log "$r reaction (b): $(head -40 $O/${r}_reaction.log | tr -d '\n ' | cut -c1-700)"
    for seed in 0 1 2 3; do
      OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/${r}_rude_s$seed --kind rude \
          --horizon 6.4 --terminal walkers --walkers 1 2 --workers 12 --seed $seed --model objects:$O/$r/model.pt > $O/${r}_rude_s$seed.log 2>&1
      log "$r rude seed $seed: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/${r}_rude_s$seed | cut -c1-140)"
    done
    OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/${r}_polite_s0 --kind polite \
        --horizon 6.4 --terminal walkers --walkers 1 2 --workers 12 --seed 0 --model objects:$O/$r/model.pt > $O/${r}_polite_s0.log 2>&1
    log "$r polite seed 0: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/${r}_polite_s0 | cut -c1-140)"
  fi
done
if [ $smoke -eq 0 ] && [ $nf -eq 0 ]; then
  log "slot model: full training (60,000 steps)"
  cd $SL; PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/s0 >> $C/s0.log 2>&1
  log "slot model s0: exit $?"
else
  log "slot model: not launched (its smoke faulted alone too)"
fi
log "after-training chain done"
