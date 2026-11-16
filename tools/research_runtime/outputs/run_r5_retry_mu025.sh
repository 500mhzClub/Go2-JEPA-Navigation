#!/bin/bash
# R5 retry: the four dyn1v2_mu025_cpu C4 missions that stopped at the then-unported auxiliary-only obstacle branch.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
$PY -m go2nav.eval.cohort configs/experiments/r5_dynamics_stage1.yaml --cohort dyn1v2_mu025_cpu --workers 4 --out outputs/cohorts \
  --retry dyn1v2_mu025_cpu_C4_prelim_test_v1_37_ep0 dyn1v2_mu025_cpu_C4_prelim_test_v1_39_ep0 dyn1v2_mu025_cpu_C4_prelim_test_v1_43_ep0 dyn1v2_mu025_cpu_C4_prelim_test_v1_46_ep0
echo "retry done $(date)"
