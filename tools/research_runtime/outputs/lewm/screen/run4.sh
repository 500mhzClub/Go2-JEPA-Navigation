#!/bin/bash
# Screening round 4 (docs/lewm/STATUS.md): span8 with the 800-ms span weighted 3x (span8w) and span8 plus motion
# supervision from the corpus poses (span8m), 15,000 steps each, scored on T7 itself like round 3. Rerunning resumes.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
export PYTHONPATH=$R
for v in span8w span8m; do
  echo "[$(date +%H:%M)] $v"
  $PY -W ignore -m go2nav.training.lewm_chain outputs/lewm/screen/t7_$v --corpus /mnt/workspace_drive/go2nav_lewm \
    --recordings-root outputs/recordings --config configs/lewm/go2_idm_$v.yaml --set optimiser.steps=15000
  cat outputs/lewm/screen/t7_$v/choice.json
done
echo "[$(date +%H:%M)] screening round 4 done"
