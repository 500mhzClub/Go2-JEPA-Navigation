#!/bin/bash
# Retrained-chain maze bench (C3 shipped vs retrained, prelim 30-49), run from the frozen worktree at commit 2dfe1f6.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "bench start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/retrained_chain.yaml --workers 4 --c3-lanes 2 --out outputs/cohorts
echo "bench done $(date)"
