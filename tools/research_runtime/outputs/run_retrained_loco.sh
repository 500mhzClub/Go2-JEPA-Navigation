#!/bin/bash
# Retrained-locomotion cohort (C1, C2, C4 on the seed-11 policy), from the frozen loco worktree.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-loco
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "loco start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/retrained_locomotion.yaml --workers 3 --c3-lanes 1 --out outputs/cohorts
echo "loco done $(date)"
