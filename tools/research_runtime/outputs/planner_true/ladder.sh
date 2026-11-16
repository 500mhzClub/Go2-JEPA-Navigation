#!/bin/bash
# P17 item 5: the ladder of models and the handicaps, from the pinned worktree ../Go2-JEPA-Navigation-planner-dev (CPU only).
# Resumable: a finished mission is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-planner-dev; O=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/planner_true
cd $W
go() {  # name, kind, horizon, walker counts (quoted), extra arguments...
  name=$1; kind=$2; horizon=$3; counts=$4; shift 4
  echo "[$(date '+%d %b %H:%M')] $name $(git rev-parse --short HEAD)" >> $O/ladder.log
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$name \
      --kind $kind --horizon $horizon --terminal walkers --walkers $counts --workers ${WORKERS:-16} "$@" > $O/$name.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $? $name" >> $O/ladder.log
}
suite() {  # prefix, horizon, extra arguments...
  prefix=$1; horizon=$2; shift 2
  go ${prefix}_rude_h$horizon rude $horizon "1 2" "$@"
  go ${prefix}_polite_h$horizon polite $horizon "1 2" "$@"
}
for h in 6.4 25.6; do
  for m in frozen_seen frozen velocity fresh; do
    suite ladder_$m $h --model $m
    go ladder_${m}_four_h$h polite $h 4 --model $m
  done
  suite slow $h --max-speed 0.2
  suite nocells $h --no-cells
done
suite slow_frozen_seen 6.4 --model frozen_seen --max-speed 0.2
