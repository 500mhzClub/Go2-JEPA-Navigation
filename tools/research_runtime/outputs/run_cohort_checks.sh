#!/bin/bash
# cohort_check --traces for every R3, R5 and R6 cohort against its original; one .check.txt per cohort.
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
check() { $PY tools/porting/cohort_check.py --traces outputs/cohorts/$1 > outputs/cohorts/$1.check.txt 2>&1; echo "$1: $(tail -1 outputs/cohorts/$1.check.txt)"; }
export -f check; export PY
ls outputs/cohorts | grep -E "^(sens_|margin_|stage2_c1_lookaround|dyn1v2_|c1a3_|s2ref_|s2eval_|s2rec2$)" | grep -v "\.check" | xargs -P 6 -I {} bash -c 'check {}'
echo "checks done $(date)"
