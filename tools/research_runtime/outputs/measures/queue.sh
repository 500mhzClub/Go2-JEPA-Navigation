#!/bin/bash
# P8 on its pinned worktree, after P7's queue: cohorts one after another; a rerun resumes after a crash or reboot.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
while systemctl --user is-active --quiet single-model-queue; do sleep 30; done
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-measures-bench
echo "[$(date +%H:%M)] measures start" >> outputs/measures/queue.log
$PY -W ignore -m go2nav.eval.cohort configs/experiments/working_measures.yaml --workers 14 --lewm-lanes 6 --resume >> outputs/measures/run.log 2>&1
echo "[$(date +%H:%M)] measures ended: $?" >> outputs/measures/queue.log
