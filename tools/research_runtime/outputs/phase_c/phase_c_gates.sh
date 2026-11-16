#!/bin/bash
# P15: phase C's gates on go2_3level_c (or its 12-block fallback), on the R9700 once training has finished.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c; H=$W/outputs/hjepa
sleep 600
while systemctl --user is-active -q hjepa-phase-c || systemctl --user is-active -q phase-c-giveway-labels-train; do sleep 300; done
RUN=$H/go2_3level_c; [ -f $H/go2_3level_c_ckpt12/evaluation.json ] && RUN=$H/go2_3level_c_ckpt12
[ -f $RUN/evaluation.json ] || { echo "[$(date '+%d %b %H:%M')] no finished phase C run" >> $C/phase_c_gates.log; exit 1; }
cd $W/../Go2-JEPA-Navigation-phase-b
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD) on $RUN" >> $C/phase_c_gates.log
EGL_DEVICE_ID=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.phase_c_gates run $RUN --out $C/gates_phase_c \
    --train-labels $C/give_way_labels_train.npz --eval-labels $C/give_way_labels_val.npz \
    --obs-root /mnt/workspace_drive/go2nav_lewm_v2_obs --heads-root /mnt/workspace_drive/go2nav_lewm_v2_heads \
    --device cuda:0 >> $C/phase_c_gates.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> $C/phase_c_gates.log
