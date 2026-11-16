#!/bin/bash
# After the frozen-target grid LeWM (frozen-grid-lewm, PLAN P4c): its mover probe on the CPU (the GPU goes to the P4d run),
# then the grid readout on the CPU.
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-grid-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
M=outputs/lewm/go2_v2_frozen_grid/lewm.pt
P=outputs/mover_probe/go2_v2_frozen_grid
cd $W
until [ -f $M ] && ! systemctl --user is-active --quiet frozen-grid-lewm; do sleep 120; done
echo "[$(date +%H:%M)] frozen grid LeWM done"
[ -f $P/results.json ] || \
  OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.mover_probe probe /mnt/workspace_drive/go2nav_lewm_v2 \
    --world-model $M --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz \
    --out $P --device cpu > $P.log 2>&1
echo "[$(date +%H:%M)] probe done $?"
[ -f $P/results_grid.json ] || \
  OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.grid_readout $P --world-model $M \
    --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz --threads 16 \
    > ${P}_readout.log 2>&1
echo "[$(date +%H:%M)] grid readout done $?"
