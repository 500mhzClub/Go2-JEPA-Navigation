#!/bin/bash
# P6: once the shadow run has finished, score it, then train the combined model (resumes from its checkpoint after a crash).
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $W
while systemctl --user is-active --quiet walker-shadow; do sleep 30; done
echo "[$(date +%H:%M)] shadow finished: $(systemctl --user show walker-shadow -p Result --value)"
OMP_NUM_THREADS=1 env -u PYTHONPATH $PY -W ignore -m go2nav.eval.walker_compare outputs/cohorts/ws_static_c5 outputs/cohorts/ws_obst1_c5 \
  outputs/cohorts/ws_obst2_c5 --out outputs/walkers/shadow_report.json --workers 12 > outputs/walkers/shadow_report.log 2>&1
echo "[$(date +%H:%M)] shadow scored; training"
for attempt in 1 2 3 4; do
  $PY -W ignore -m go2nav.training.lewm train --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out outputs/lewm/go2_v2_walkers \
    --config configs/lewm/go2_v2_walkers.yaml >> outputs/walkers/train.log 2>&1 && { echo "[$(date +%H:%M)] trained"; exit 0; }
  echo "[$(date +%H:%M)] training attempt $attempt failed"
  sleep 60
done
exit 1
