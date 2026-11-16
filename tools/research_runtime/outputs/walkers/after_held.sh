#!/bin/bash
# P6: the one-model test, then the four-walker pairs; each cohort run resumes after a crash.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
$PY -W ignore -m go2nav.eval.cohort configs/experiments/walker_one.yaml --workers 6 --lewm-lanes 6 --resume >> outputs/walkers/one.log 2>&1
echo "[$(date +%H:%M)] one-model run ended: $?" >> outputs/walkers/queue.log
$PY -W ignore -m go2nav.eval.cohort configs/experiments/walker_four.yaml --workers 6 --lewm-lanes 6 --resume >> outputs/walkers/four.log 2>&1
echo "[$(date +%H:%M)] four-walker run ended: $?" >> outputs/walkers/queue.log
