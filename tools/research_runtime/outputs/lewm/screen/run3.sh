#!/bin/bash
# Screening round 3 (docs/lewm/STATUS.md): longer-span inverse dynamics for gate L3, scored on T7 itself (cache, base
# readout from T7's contexts, three fits: L4 = C5's median score, L3 = the chosen fit's probe), after the C5 cohorts
# free the GPU. The baseline is idm100's existing 15,000-step screen. Rerunning resumes (lewm_chain skips finished stages).
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
export PYTHONPATH=$R
while systemctl --user is-active --quiet lewm-c5-bench; do sleep 60; done
chain() { $PY -W ignore -m go2nav.training.lewm_chain outputs/lewm/screen/t7_$1 --corpus /mnt/workspace_drive/go2nav_lewm \
            --recordings-root outputs/recordings --config configs/lewm/$2 --set optimiser.steps=15000; }
mkdir -p outputs/lewm/screen/t7_idm100
[ -e outputs/lewm/screen/t7_idm100/world ] || ln -s ../idm100 outputs/lewm/screen/t7_idm100/world
for v in idm100:go2_idm100.yaml span8:go2_idm_span8.yaml spans:go2_idm_spans.yaml; do
  echo "[$(date +%H:%M)] ${v%%:*}"
  chain ${v%%:*} ${v#*:}
  cat outputs/lewm/screen/t7_${v%%:*}/choice.json
done
echo "[$(date +%H:%M)] screening round 3 done"
