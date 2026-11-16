#!/bin/bash
# P22: M5 and M6's offline measure (a) and reaction (b) once each has trained (CPU evaluation). Log: chain4.log.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/object_dynamics
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/chain4.log; }
cd $W
for r in m6 m5; do
  until [ -f $O/$r/model.pt ] && ! systemctl --user is-active --quiet objects-$r; do sleep 60; done
  [ -f $O/${r}_offline.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures offline \
      --model $O/$r/model.pt --out $O/${r}_offline.json --workers 24 > $O/${r}_offline.log 2>&1
  [ -f $O/${r}_reaction.json ] || HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.object_measures reaction \
      --model $O/$r/model.pt --out $O/${r}_reaction.json --workers 24 > $O/${r}_reaction.log 2>&1
  log "$r: $($PY -c "
import json; d=json.load(open('$O/${r}_offline.json')); e=json.load(open('$O/${r}_reaction.json'))
print('offline all', ' '.join(f\"{h} {d[h+'_all']['model']:.3f}/{d[h+'_all']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; waiting', ' '.join(f\"{h} {d[h+'_waiting']['model']:.3f}/{d[h+'_waiting']['velocity']:.3f}\" for h in ('1.6s','3.2s','6.4s')), '; reaction', ' '.join(f\"{h} {e[h]['slope']} {e[h]['slope_95']}\" for h in ('1.6s','3.2s','6.4s')))")"
done
log "chain4 done"
