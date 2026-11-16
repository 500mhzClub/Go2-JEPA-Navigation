#!/bin/bash
# R5 then R6 cohorts, run from the frozen worktree at commit 009646a (outputs in the main repository).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-run
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
OUT=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/cohorts
echo "R5 start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/r5_dynamics_stage1.yaml --workers 8 --out $OUT
echo "R5 done $(date)"
$PY -m go2nav.eval.cohort configs/experiments/r6_dynamics_stage2.yaml --workers 8 --out $OUT
echo "R6 done $(date)"
