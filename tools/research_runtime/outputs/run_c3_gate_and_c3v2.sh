#!/bin/bash
# Rerun of the C3 v4 gate (the first attempt died of a GPU fault), then the C3-v2 fresh-check cohort (decoder source).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/missions/v4_gates
mv $O/C3_off_prelim30 $O/C3_off_prelim30.gpu_fault; mv $O/C3_off_prelim30.log $O/C3_off_prelim30.gpu_fault.log
( $PY -m go2nav.eval.mission --arm C3 --set prelim_test_v1 --maze 30 --recovery off --coverage-fix off --harness v4 --out $O/C3_off_prelim30 > $O/C3_off_prelim30.log 2>&1; echo "C3_off_prelim30 $(cat $O/C3_off_prelim30/result.json 2>/dev/null) $(date)" ) &
$PY -m go2nav.eval.cohort configs/experiments/decoder_sources.yaml --cohort c3v2_check --workers 2 --c3-lanes 2 --out outputs/cohorts
echo "c3v2_check done $(date)"
wait
