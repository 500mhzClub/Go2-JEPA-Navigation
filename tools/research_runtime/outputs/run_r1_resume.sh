#!/bin/bash
# R1 resumed after the 14 October interruption (frozen worktree at the --resume commit), then every R1 cohort checked.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-resume
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "R1 resume start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/r1_preliminary_comparison.yaml --resume --workers 6 --c3-lanes 1 --out outputs/cohorts
for c in prelim_trial prelim_on prelim_on_c0 prelim_off prelim_off_extra55; do
  PYTHONPATH=.:tools/porting $PY tools/porting/cohort_check.py --traces outputs/cohorts/$c > outputs/cohorts/$c.check.txt 2>&1
  echo "$c: $(tail -1 outputs/cohorts/$c.check.txt)"
done
echo "R1 resume done $(date)"
