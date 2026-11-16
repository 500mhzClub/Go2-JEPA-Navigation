#!/bin/bash
# P22 item 10: once N3 has trained: the stand-off table (N3 beside N0), (a), (b), then the expected-cost planner on
# four walkers that do not yield (seeds 0-3), and polite, 1-2 not yielding and reactive (seed 0). CPU evaluation.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/chain7.log; }
cd $W
run() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore "$@"; }
suite() { [ -f $O/$1/summary.json ] || run -m go2nav.eval.kinematic_missions run --out $O/$1 --kind $3 --horizon 6.4 --terminal walkers --walkers $4 --workers 28 --seed $5 --model $2 > $O/$1.log 2>&1
  log "$1: $(run -m go2nav.eval.kinematic_missions summary $O/$1 | cut -c1-170)"; }
r=n3
until [ -f $O/$r/model.pt ] && ! systemctl --user is-active --quiet objects-$r; do sleep 60; done
log "$r trained: $(tail -1 $O/$r/validation.jsonl | cut -c1-420)"
run outputs/walker_memory/standoff.py modes:$O/n0/model.pt modes:$O/$r/model.pt > outputs/walker_memory/standoff_$r.txt 2>&1
log "stand-offs: $(tail -9 outputs/walker_memory/standoff_$r.txt | tr '\n' '#' | cut -c1-1500)"
run -m go2nav.eval.object_measures offline --model modes:$O/$r/model.pt --out $O/${r}_offline.json --workers 26 > $O/${r}_offline.log 2>&1
run -m go2nav.eval.object_measures reaction --model modes:$O/$r/model.pt --out $O/${r}_reaction.json --workers 26 > $O/${r}_reaction.log 2>&1
log "$r: $($PY -c "
import json; d=json.load(open('$O/${r}_offline.json')); e=json.load(open('$O/${r}_reaction.json'))
print('offline all', ' '.join(f\"{h} {d[h+'_all']['model']:.3f}/{d[h+'_all']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; waiting', ' '.join(f\"{h} {d[h+'_waiting']['model']:.3f}/{d[h+'_waiting']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; reaction', ' '.join(f\"{h} {e[h]['slope']} {e[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
for seed in 0 1 2 3; do suite rude4_${r}expect_s$seed expect:$O/$r/model.pt rude 4 $seed; done
suite polite12_${r}expect_s0 expect:$O/$r/model.pt polite "1 2" 0
suite rude12_${r}expect_s0 expect:$O/$r/model.pt rude "1 2" 0
suite reactive12_${r}expect_s0 expect:$O/$r/model.pt reactive "1 2" 0
log "chain7 done"
