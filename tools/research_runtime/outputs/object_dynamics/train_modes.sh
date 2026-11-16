#!/bin/bash
# P22: train one several-futures predictor run (configs/object/RUN.yaml). Resumable: rerun.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $W
echo "[$(date '+%d %b %H:%M')] $1 start $(git rev-parse --short HEAD)" >> outputs/object_dynamics/train.log
PYTHONPATH=. $PY -W ignore -m go2nav.training.object_modes train --corpus /mnt/workspace_drive/go2nav_ring/corpus_v1 \
    --out outputs/object_dynamics/$1 --config configs/object/$1.yaml >> outputs/object_dynamics/$1.log 2>&1
echo "[$(date '+%d %b %H:%M')] $1 exit $?" >> outputs/object_dynamics/train.log
