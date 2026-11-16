#!/bin/bash
# P15: phase C's gates on go2_3level_v2 (corpus v2 without P14's stand-off units): the before-data baseline, and the
# pipeline's first run on a trained model. CPU, once the relabelling (side step) is done.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c
sleep 60
while systemctl --user is-active -q phase-c-giveway-labels-train || systemctl --user is-active -q phase-c-giveway-labels-val; do sleep 60; done
cd $W/../Go2-JEPA-Navigation-phase-b
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> $C/v2_gates.log
HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=. $PY -W ignore -m go2nav.eval.phase_c_gates run $W/outputs/hjepa/go2_3level_v2 \
    --out $C/gates_v2 --train-labels $C/give_way_labels_train.npz --eval-labels $C/give_way_labels_val.npz \
    --heads-root /mnt/workspace_drive/go2nav_lewm_v2_heads --device cpu >> $C/v2_gates.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> $C/v2_gates.log
