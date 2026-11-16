#!/bin/bash
# P23's second run (s1): VideoSAUR's own choices (configs/slots/s1_videosaur.yaml), GPU, full precision; then its
# binding. Resumable.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
SL=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots; C=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/camera_slots; F=/mnt/workspace_drive/go2nav_dino/features_vits14
cd $SL
echo "[$(date '+%d %b %H:%M')] s1 start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)" >> $C/train.log
PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/s1 --config configs/slots/s1_videosaur.yaml >> $C/s1.log 2>&1
echo "[$(date '+%d %b %H:%M')] s1 exit $?" >> $C/train.log
[ -f $C/s1/model.pt ] && PYTHONPATH=$SL $PY -W ignore -m go2nav.eval.slot_measures score --model $C/s1/model.pt --features $F --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $C/s1_binding.json > $C/s1_binding.log 2>&1
echo "[$(date '+%d %b %H:%M')] s1 binding: $(tr -d '\n ' < $C/s1_binding.json | cut -c1-400)" >> $C/train.log
