#!/bin/bash
# R3 (forecast sensitivity and margins, v4 harness), from the frozen r13 worktree; then the cohort checks.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-r13
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "R3 start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/r3_sensitivity_and_margins.yaml --workers 4 --c3-lanes 1 --out outputs/cohorts
echo "R3 done $(date)"
