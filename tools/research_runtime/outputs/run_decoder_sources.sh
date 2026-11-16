#!/bin/bash
# Decoder data-source cohorts (v4 harness), from the frozen gate worktree; c3v2_check runs after its gate passes.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "c3v3_onpolicy start $(date)"
$PY -m go2nav.eval.cohort configs/experiments/decoder_sources.yaml --cohort c3v3_onpolicy --workers 4 --out outputs/cohorts
echo "c3v3_onpolicy done $(date)"
