#!/bin/bash
# P4d (go2_v2_grid_movers) on the GPU once the frozen-target grid LeWM (frozen-grid-lewm) has finished; then its mover
# probe on the CPU and the grid readout.
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-grid-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
O=outputs/lewm/go2_v2_grid_movers
P=outputs/mover_probe/go2_v2_grid_movers
cd $W
until [ -f outputs/lewm/go2_v2_frozen_grid/lewm.pt ] && ! systemctl --user is-active --quiet frozen-grid-lewm; do sleep 60; done
echo "[$(date +%H:%M)] frozen grid done; P4d training"
cd $B
PYTHONPATH=$B $PY -W ignore -m go2nav.training.lewm train --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $O \
  --config configs/lewm/go2_v2_grid_movers.yaml >> $W/outputs/grid/movers_train.log 2>&1 || { echo "training failed"; exit 1; }
cd $W
echo "[$(date +%H:%M)] P4d trained; probe"
[ -f $P/results.json ] || OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.mover_probe probe \
  /mnt/workspace_drive/go2nav_lewm_v2 --world-model $O/lewm.pt --train outputs/mover_probe/labels_train.npz \
  --eval outputs/mover_probe/labels_val.npz --out $P --device cuda:0 > $P.log 2>&1
[ -f $P/results_grid.json ] || OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.grid_readout $P \
  --world-model $O/lewm.pt --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz \
  --threads 16 > ${P}_readout.log 2>&1
echo "[$(date +%H:%M)] P4d probe and readout done"
