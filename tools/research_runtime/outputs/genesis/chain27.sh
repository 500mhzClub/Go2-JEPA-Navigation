#!/bin/bash
# P24 stage 1 (approved 15 November): P22's (a) for M5 on corpus v2's 400 Genesis validation units (data.genesis_units),
# then the same measure on the ring corpus as the control (it must give P22's published figures). CPU, 26 workers.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; G=$W/outputs/genesis; L=$G/chain.log; M5=$W/outputs/object_dynamics/m5/model.pt
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $W
cpu() { HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTORCH_TUNABLEOP_ENABLED=0 PYTHONPATH=. $PY -W ignore "$@"; }
log "chain27 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)"
[ -f $G/m5_genesis_offline.json ] || GO2NAV_RING_CORPUS=$G/units_val cpu -m go2nav.eval.object_measures offline \
  --model $M5 --out $G/m5_genesis_offline.json --workers 26 > $G/m5_genesis_offline.log 2>&1
log "M5 on Genesis: exit $?: $(cpu -c "import json; r=json.load(open('$G/m5_genesis_offline.json')); print(json.dumps({k: v for k, v in r.items() if k != 'by'}))" | cut -c1-900)"
[ -f $G/m5_ring_offline.json ] || cpu -m go2nav.eval.object_measures offline --model $M5 --out $G/m5_ring_offline.json \
  --workers 26 > $G/m5_ring_offline.log 2>&1
log "M5 on the ring corpus (control): exit $?: $(cpu -c "import json; r=json.load(open('$G/m5_ring_offline.json')); print(json.dumps({k: v for k, v in r.items() if k != 'by'}))" | cut -c1-900)"
log "chain27 done"
