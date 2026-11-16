#!/bin/bash
# P7: wait for the reproduction check, then the held-out run; if the six missions differ, the 80 missions first (PLAN P7).
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
while systemctl --user is-active --quiet single-model-check; do sleep 20; done
verdict=$(PYTHONPATH=$R $PY -W ignore outputs/single_model/compare_repro.py 2>&1 | tee -a outputs/single_model/compare.log | tail -1)
echo "[$(date +%H:%M)] reproduction: $verdict" >> outputs/single_model/queue.log
if [ "$verdict" != REPRODUCED ]; then
    cd $R-single-bench
    $PY -W ignore -m go2nav.eval.cohort outputs/single_model/single_model_rerun.yaml --workers 6 --lewm-lanes 6 --resume >> outputs/single_model/rerun.log 2>&1
    echo "[$(date +%H:%M)] 80-mission fallback ended: $?" >> outputs/single_model/queue.log
fi
$R/outputs/single_model/heldout.sh
