#!/bin/bash
# R1's remaining cohorts (C0 recovery on; recovery off) in parallel with prelim_on, from the same frozen worktree
# (e6817fd), then their checks against the originals. The main R1 runner will stop when it reaches these folders.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-r1
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "R1 rest start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.eval.cohort configs/experiments/r1_preliminary_comparison.yaml --cohort prelim_on_c0 --cohort prelim_off --cohort prelim_off_extra55 --workers 8 --c3-lanes 1 --out outputs/cohorts
for c in prelim_on_c0 prelim_off prelim_off_extra55; do
  PYTHONPATH=.:tools/porting $PY tools/porting/cohort_check.py --traces outputs/cohorts/$c > outputs/cohorts/$c.check.txt 2>&1
  echo "$c: $(tail -1 outputs/cohorts/$c.check.txt)"
done
echo "R1 rest done $(date)"
