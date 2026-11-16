#!/bin/bash
# P21: once a run has trained, score it with P17's planner (walkers that do not yield, then polite; 6.4 s, seed 0).
# Usage: score.sh RUN. Resumable: a finished mission is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/ring_model
cd $W
until [ -f $O/$1/model.pt ] && ! systemctl --user is-active --quiet ring-$1; do sleep 60; done
for kind in rude polite; do
  echo "[$(date '+%d %b %H:%M')] $1 $kind start $(git rev-parse --short HEAD)" >> $O/score.log
  OMP_NUM_THREADS=2 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$1_$kind --kind $kind \
      --horizon 6.4 --terminal walkers --walkers 1 2 --workers 8 --model ring_model:$O/$1/model.pt > $O/$1_$kind.log 2>&1
  echo "[$(date '+%d %b %H:%M')] $1 $kind exit $? :: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/$1_$kind | cut -c1-160)" >> $O/score.log
done
