#!/bin/bash
# P15 (approved 8 November): go2_3level_c on corpus v2 + P14, from the phase-b worktree, after the control finishes (one
# model on the R9700 at a time). Resumable: rerun (systemctl --user start hjepa-phase-c). If the first attempt runs out
# of GPU memory under the 85% cap, it restarts once with every ViT block checkpointed, in its own folder.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
H=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/hjepa
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b
while systemctl --user is-active -q hjepa-p14-control; do sleep 60; done
O=$H/go2_3level_c
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> $O.log
$PY -W ignore -m go2nav.training.hjepa train --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $O \
    --config configs/hjepa/go2_3level_c.yaml >> $O.log 2>&1
code=$?
echo "[$(date '+%d %b %H:%M')] exit $code" >> $O.log
if [ $code -ne 0 ] && grep -qiE "out of memory|OutOfMemory" $O.log; then
  O2=$H/go2_3level_c_ckpt12
  echo "[$(date '+%d %b %H:%M')] out of memory: restart with 12 checkpointed blocks in $O2" >> $O.log
  $PY -W ignore -m go2nav.training.hjepa train --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $O2 \
      --config configs/hjepa/go2_3level_c.yaml --set runtime.checkpoint_blocks=12 >> $O2.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $?" >> $O2.log
fi
