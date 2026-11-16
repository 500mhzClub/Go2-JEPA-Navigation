#!/bin/bash
# The mover probe on the final LeWM v2, on the CPU (H-JEPA v2 holds the GPU), once training has written lewm.pt.
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $W
until [ -f outputs/lewm/go2_v2/lewm.pt ]; do sleep 60; done
echo "[$(date +%H:%M)] probe on go2_v2/lewm.pt"
OMP_NUM_THREADS=12 PYTHONPATH=$W $PY -W ignore -m go2nav.eval.mover_probe probe /mnt/workspace_drive/go2nav_lewm_v2 \
  --world-model outputs/lewm/go2_v2/lewm.pt --train outputs/mover_probe/labels_train.npz \
  --eval outputs/mover_probe/labels_val.npz --out outputs/mover_probe/go2_v2 --device cpu
echo "[$(date +%H:%M)] done"
