#!/bin/bash
# Retry the 27 missions that stopped at branches since ported (commit 5e948d4), one cohort runner per cohort,
# then re-check every affected cohort against its original. The two C3 retries wait for the C3 v4 gate (GPU lanes).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-retry
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
L=outputs/retries; mkdir -p $L
retry() { # experiment cohort workers assignments...
  local e=$1 c=$2 w=$3; shift 3
  $PY -m go2nav.eval.cohort configs/experiments/$e --cohort $c --workers $w --out outputs/cohorts --retry "$@" > $L/$c.log 2>&1
  PYTHONPATH=.:tools/porting $PY tools/porting/cohort_check.py --traces outputs/cohorts/$c > outputs/cohorts/$c.check.txt 2>&1
  echo "$c: $(tail -1 outputs/cohorts/$c.check.txt) $(date +%H:%M)"
}
echo "wave 1 (stage 2) $(git rev-parse --short HEAD) $(date)"
R6=r6_dynamics_stage2.yaml
retry $R6 s2eval_cpu 9 s2eval_cpu_C1_stage2_sets_v1_33_ep1 s2eval_cpu_C1A_stage2_sets_v1_33_ep1 s2eval_cpu_C1A_stage2_sets_v1_39_ep1 s2eval_cpu_C1A_stage2_sets_v1_40_ep0 s2eval_cpu_C1_stage2_sets_v1_47_ep1 s2eval_cpu_C1R_stage2_sets_v1_47_ep1 s2eval_cpu_C1A_stage2_sets_v1_48_ep1 s2eval_cpu_C1_stage2_sets_v1_49_ep0 &
retry $R6 s2rec2 3 s2rec2_C1_stage2_sets_v1_11_ep0 s2rec2_C1_stage2_sets_v1_18_ep0 s2rec2_C1_stage2_sets_v1_28_ep1 &
retry $R6 s2ref_cpu 1 s2ref_cpu_C1_stage2_sets_v1_47_ep1 &
retry r5_dynamics_stage1.yaml dyn1v2_mu025_cpu 1 dyn1v2_mu025_cpu_C4_prelim_test_v1_46_ep0 &
wait
echo "wave 2 (sensitivity) $(date)"
R3=r3_sensitivity_and_margins.yaml
retry $R3 sens_fwdscale0p25 1 sens_fwdscale0p25_C1_prelim_test_v1_40_ep0 &
retry $R3 sens_noise160 5 sens_noise160_C1_prelim_test_v1_33_ep0 sens_noise160_C1_prelim_test_v1_37_ep0 sens_noise160_C1_prelim_test_v1_40_ep0 sens_noise160_C1_prelim_test_v1_46_ep0 sens_noise160_C1_prelim_test_v1_47_ep0 &
retry $R3 sens_noise40 1 sens_noise40_C1_prelim_test_v1_45_ep0 &
retry $R3 sens_noise80 1 sens_noise80_C1_prelim_test_v1_35_ep0 &
retry $R3 sens_turnscale0p5 1 sens_turnscale0p5_C1_prelim_test_v1_43_ep0 &
retry $R3 stage2_c1_lookaround_p95 3 stage2_c1_lookaround_p95_C1_prelim_test_v1_31_ep0 stage2_c1_lookaround_p95_C1_prelim_test_v1_32_ep0 stage2_c1_lookaround_p95_C1_prelim_test_v1_37_ep0 &
wait
echo "wave 3 (C3, after the C3 v4 gate) $(date)"
until [ -f outputs/missions/v4_gates/C3_off_prelim30/result.json ]; do sleep 60; done
retry r5_dynamics_stage1.yaml dyn1v2_mu030_c3 1 dyn1v2_mu030_c3_C3_prelim_test_v1_31_ep0 dyn1v2_mu030_c3_C3_prelim_test_v1_48_ep0
echo "retries done $(date)"
