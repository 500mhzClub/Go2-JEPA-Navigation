#!/bin/bash
# P14 control (approved 8 November): configs/hjepa/go2_3level_p14_control.yaml on corpus v2 + P14 stand-off units,
# from the phase-b worktree. Resumable: rerun (systemctl --user start hjepa-p14-control).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b
O=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/hjepa/go2_3level_p14_control
mkdir -p $O
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> $O.log
$PY -W ignore -m go2nav.training.hjepa train --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $O \
    --config configs/hjepa/go2_3level_p14_control.yaml >> $O.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> $O.log
