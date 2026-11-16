#!/bin/bash
# The stage-2 recordings (R6 cohort s2rec2) for the end-to-end decoder data, into a separate root (R6 runs its own later).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
$PY -m go2nav.eval.cohort configs/experiments/r6_dynamics_stage2.yaml --cohort s2rec2 --workers 4 --out outputs/cohorts_e2e
echo "s2rec2 done $(date)"
