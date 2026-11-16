#!/bin/bash
# P21: train one ring world model run. Resumable: rerun. Usage: train.sh RUN (configs/ring/RUN.yaml).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $W
echo "[$(date '+%d %b %H:%M')] $1 start $(git rev-parse --short HEAD)" >> outputs/ring_model/train.log
PYTHONPATH=. $PY -W ignore -m go2nav.training.ring_jepa train --corpus /mnt/workspace_drive/go2nav_ring/corpus_v1 \
    --out outputs/ring_model/$1 --config configs/ring/$1.yaml >> outputs/ring_model/$1.log 2>&1
echo "[$(date '+%d %b %H:%M')] $1 exit $?" >> outputs/ring_model/train.log
