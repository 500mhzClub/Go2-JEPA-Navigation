#!/bin/bash
# End-to-end stage D resumed: C3 on the models retrained from regenerated data (prelim 30-49).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-resume
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "stage D resume start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/e2e_chain.yaml --resume --workers 2 --c3-lanes 1 --out outputs/cohorts
echo "end-to-end done $(date)"
