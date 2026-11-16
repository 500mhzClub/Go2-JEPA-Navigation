#!/bin/bash
# P23 items 1-2: DINOv2 ViT-S/14 patch features of corpus v2's frames (GPU). Resumable: a finished unit is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots; O=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/camera_slots
cd $W
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> $O/features.log
PYTHONPATH=$W $PY -W ignore -m go2nav.data.dino_features extract --corpus /mnt/workspace_drive/go2nav_lewm_v2 \
    --out /mnt/workspace_drive/go2nav_dino/features_vits14 >> $O/features.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> $O/features.log
