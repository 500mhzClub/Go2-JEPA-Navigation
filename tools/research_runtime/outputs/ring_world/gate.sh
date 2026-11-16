#!/bin/bash
# P20 item 5: the ring models on P17's ladder (CPU only). Resumable: a finished mission is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/ring_world
cd $W
go() {  # name, kind, model
  echo "[$(date '+%d %b %H:%M:%S')] $1 $(git rev-parse --short HEAD)" >> $O/gate.log
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$1 \
      --kind $2 --horizon 6.4 --terminal walkers --walkers 1 2 --workers 16 --model $3 > $O/$1.log 2>&1
  echo "[$(date '+%d %b %H:%M:%S')] exit $? $1" >> $O/gate.log
}
for m in ring_velocity ring_frozen frozen_seen; do go ${m}_rude rude $m; done
for m in ring_velocity ring_frozen; do go ${m}_polite polite $m; done
