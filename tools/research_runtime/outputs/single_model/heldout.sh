#!/bin/bash
# P7's held-out evaluation on the pinned worktree; a rerun resumes after a crash or reboot.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-single-bench
$PY -W ignore -m go2nav.eval.cohort configs/experiments/single_model_heldout.yaml --workers 6 --lewm-lanes 6 --resume >> outputs/single_model/heldout.log 2>&1
echo "[$(date +%H:%M)] held-out run ended: $?" >> outputs/single_model/queue.log
