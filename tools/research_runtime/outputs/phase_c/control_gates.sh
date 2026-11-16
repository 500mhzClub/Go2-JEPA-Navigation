#!/bin/bash
# P15: phase C's gates on the control (go2_3level_p14_control), on the CPU while go2_3level_c trains on the R9700,
# once the control has written hjepa.pt and the training labels are done.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c
RUN=$W/outputs/hjepa/go2_3level_p14_control
until [ -f $RUN/evaluation.json ] && ! systemctl --user is-active -q phase-c-giveway-labels-train; do sleep 120; done
cd $W/../Go2-JEPA-Navigation-phase-b
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> $C/control_gates.log
HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=. $PY -W ignore -m go2nav.eval.phase_c_gates run $RUN --out $C/gates_control \
    --train-labels $C/give_way_labels_train.npz --eval-labels $C/give_way_labels_val.npz \
    --heads-root /mnt/workspace_drive/go2nav_lewm_v2_heads --device cpu >> $C/control_gates.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> $C/control_gates.log
