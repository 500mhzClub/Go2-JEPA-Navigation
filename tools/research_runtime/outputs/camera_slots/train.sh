#!/bin/bash
# P23: train the slot model (GPU, full precision), then score its binding. Resumable.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
SL=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots; C=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/camera_slots; F=/mnt/workspace_drive/go2nav_dino/features_vits14
cd $SL
echo "[$(date '+%d %b %H:%M')] s0 start $(git rev-parse --short HEAD)" >> $C/train.log
PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/s0 >> $C/s0.log 2>&1
echo "[$(date '+%d %b %H:%M')] s0 exit $?" >> $C/train.log
[ -f $C/s0/model.pt ] && PYTHONPATH=$SL $PY -W ignore -m go2nav.eval.slot_measures score --model $C/s0/model.pt --features $F --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $C/s0_binding.json > $C/s0_binding.log 2>&1
echo "[$(date '+%d %b %H:%M')] binding: $(tr -d '\n ' < $C/s0_binding.json | cut -c1-400)" >> $C/train.log
