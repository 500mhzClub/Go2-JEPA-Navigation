#!/bin/bash
# P15: early reads of phase C (informational; the decision is the final model's): its 20k and 40k checkpoints, scored on
# the CPU while it trains on the GPU. Health is skipped (no evaluation.json until the run ends).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c; H=$W/outputs/hjepa/go2_3level_c
cd $W/../Go2-JEPA-Navigation-phase-b
for step in 20000 40000; do
  while [ ! -f $H/hjepa_step$step.pt ]; do sleep 120; done
  sleep 60                                                         # let the write finish
  R=$C/interim_c${step%000}k
  echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD) step $step" >> $C/interim_gates.log
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=20 PYTHONPATH=. $PY -W ignore -m go2nav.eval.phase_c_gates interim $H/hjepa_step$step.pt $R/run >> $C/interim_gates.log 2>&1
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=20 PYTHONPATH=. $PY -W ignore -m go2nav.eval.phase_c_gates run $R/run --out $R \
      --train-labels $C/give_way_labels_train.npz --eval-labels $C/give_way_labels_val.npz \
      --obs-root /mnt/workspace_drive/go2nav_lewm_v2_obs --heads-root /mnt/workspace_drive/go2nav_lewm_v2_heads \
      --device cpu >> $C/interim_gates.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $? step $step" >> $C/interim_gates.log
done
