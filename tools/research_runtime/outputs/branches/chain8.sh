#!/bin/bash
# P22 item 11: N4 (N3 with half of each batch branched windows, the first 2,250 train episodes' branches) once those
# exist; then, with the branch build stopped (resumed after) and the optimism runs done: the stand-off tables (hold and
# the recorded tape; N3 beside N4), (a), (b), tape optimism, and the expected-cost planner on four walkers that do not
# yield (seeds 0-3) and on 1-2 polite, not yielding and reactive walkers (seed 0).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics; L=$W/outputs/branches/chain8.log
B=/mnt/workspace_drive/go2nav_ring/corpus_v1/branches
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $O/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $O/$1 --kind $3 --horizon 6.4 --terminal walkers --walkers $4 --workers 28 --seed $5 --model $2 > $O/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $O/$1 | tr -d '\n ' | cut -c1-200)"; }
r=n4
until $PY -c "import os,sys; s=set(os.listdir('$B/train')); sys.exit(0 if all(f'{i:05d}.npz' in s for i in range(2250)) else 1)"; do sleep 120; done
log "first 2,250 train episodes branched; training $r"
if [ ! -f $O/$r/model.pt ]; then
  outputs/object_dynamics/train_modes.sh $r &
  sleep 240; log "GPU after start: $(rocm-smi --showmemuse 2>/dev/null | grep -m1 'VRAM%' | tr -s ' ') $(rocm-smi --showuse 2>/dev/null | grep -m1 'GPU use' | tr -s ' ')"
  wait
fi
[ -f $O/$r/model.pt ] || { log "$r did not finish"; exit 1; }
log "$r trained: $(cat $O/$r/branches.json | tr -d '\n') $(tail -1 $O/$r/validation.jsonl | cut -c1-700)"
until ! systemctl --user is-active --quiet branches-optimism; do sleep 60; done
systemctl --user stop branches-build; log "branch build stopped for the measures"
run outputs/walker_memory/standoff_mass.py modes:$O/n3/model.pt modes:$O/$r/model.pt > outputs/walker_memory/standoff_mass_$r.txt 2>&1
log "stand-offs, hold: $(tail -9 outputs/walker_memory/standoff_mass_$r.txt | tr '\n' '#' | cut -c1-2600)"
run outputs/walker_memory/standoff_mass_recorded.py modes:$O/n3/model.pt modes:$O/$r/model.pt > outputs/walker_memory/standoff_mass_recorded_$r.txt 2>&1
log "stand-offs, recorded: $(tail -9 outputs/walker_memory/standoff_mass_recorded_$r.txt | tr '\n' '#' | cut -c1-2600)"
run -m go2nav.eval.object_measures offline --model modes:$O/$r/model.pt --out $O/${r}_offline.json --workers 26 > $O/${r}_offline.log 2>&1
run -m go2nav.eval.object_measures reaction --model modes:$O/$r/model.pt --out $O/${r}_reaction.json --workers 26 > $O/${r}_reaction.log 2>&1
log "$r: $($PY -c "
import json; d=json.load(open('$O/${r}_offline.json')); e=json.load(open('$O/${r}_reaction.json'))
print('offline all', ' '.join(f\"{h} {d[h+'_all']['model']:.3f}/{d[h+'_all']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; waiting', ' '.join(f\"{h} {d[h+'_waiting']['model']:.3f}/{d[h+'_waiting']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; reaction', ' '.join(f\"{h} {e[h]['slope']} {e[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
run -m go2nav.eval.tape_optimism run --model expect:$O/$r/model.pt --out outputs/branches/optimism_${r}expect --every 5 --limit 4 --workers 26 > outputs/branches/optimism_${r}expect.log 2>&1
log "optimism ${r}expect: $(tr -d '\n ' < outputs/branches/optimism_${r}expect/summary.json)"
for seed in 0 1 2 3; do suite rude4_${r}expect_s$seed expect:$O/$r/model.pt rude 4 $seed; done
suite polite12_${r}expect_s0 expect:$O/$r/model.pt polite "1 2" 0
suite rude12_${r}expect_s0 expect:$O/$r/model.pt rude "1 2" 0
suite reactive12_${r}expect_s0 expect:$O/$r/model.pt reactive "1 2" 0
systemctl --user start branches-build; log "branch build resumed; chain8 done"
