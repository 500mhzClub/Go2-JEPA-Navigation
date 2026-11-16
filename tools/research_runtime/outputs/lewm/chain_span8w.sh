#!/bin/bash
# The full Go2 chain for span8w, the winner of screening round 4 (docs/lewm/STATUS.md): 60,000 steps into
# outputs/lewm/chain_span8w, then the cache, the contexts base readout, three T7 fits and choice.json.
# Rerunning resumes (the chain skips finished stages; training resumes from its last checkpoint).
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
export PYTHONPATH=$R
echo "[$(date +%H:%M)] chain_span8w"
$PY -W ignore -m go2nav.training.lewm_chain outputs/lewm/chain_span8w --corpus /mnt/workspace_drive/go2nav_lewm \
  --recordings-root outputs/recordings --config configs/lewm/go2_idm_span8w.yaml
cat outputs/lewm/chain_span8w/choice.json
echo "[$(date +%H:%M)] chain_span8w done"
