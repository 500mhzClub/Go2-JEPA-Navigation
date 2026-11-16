#!/bin/bash
# P16: the side step's tapes of the give-way label files (once; no model), on the CPU.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c; M=$W/outputs/measures
cd $W
for name in val train; do
  echo "[$(date '+%d %b %H:%M')] tapes $name $(git rev-parse --short HEAD)" >> $M/tapes.log
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.measures tapes --root /mnt/workspace_drive/go2nav_lewm_v2 --labels $C/give_way_labels_$name.npz \
      --out $M/side_tapes_$name.npy --workers 28 >> $M/tapes.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $?" >> $M/tapes.log
done
