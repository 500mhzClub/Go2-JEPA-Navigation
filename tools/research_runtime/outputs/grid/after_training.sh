#!/bin/bash
# After the grid LeWM (grid-lewm): its mover probe on the CPU (the LeWM v2 retrain was dropped: compiled training was
# never affected by the LayerNorm bug).
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-grid-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $W
until [ -f outputs/lewm/go2_v2_grid/lewm.pt ] && ! systemctl --user is-active --quiet grid-lewm; do sleep 120; done
echo "[$(date +%H:%M)] grid LeWM done"
[ -f outputs/mover_probe/go2_v2_grid/results.json ] || \
  OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.mover_probe probe /mnt/workspace_drive/go2nav_lewm_v2 \
    --world-model outputs/lewm/go2_v2_grid/lewm.pt --train outputs/mover_probe/labels_train.npz \
    --eval outputs/mover_probe/labels_val.npz --out outputs/mover_probe/go2_v2_grid --device cpu \
    > outputs/mover_probe/go2_v2_grid.log 2>&1
echo "[$(date +%H:%M)] grid probe done"
