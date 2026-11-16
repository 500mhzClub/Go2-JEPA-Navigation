#!/bin/bash
# P17: the sampling planner on the true model, in the kinematic world, on the reference suites (CPU only).
# Resumable: a finished mission is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/planner_true
cd $W
go() {  # name, kind, horizon, terminal, walker counts...
  name=$1; kind=$2; horizon=$3; terminal=$4; shift 4
  echo "[$(date '+%d %b %H:%M')] $name $(git rev-parse --short HEAD)" >> $O/runs.log
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$name \
      --kind $kind --horizon $horizon --terminal $terminal --walkers "$@" --workers 28 > $O/$name.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $? $name" >> $O/runs.log
}
for h in 1.6 6.4 25.6; do for t in static walkers; do go rude_h${h}_$t rude $h $t 1 2; done; done
go still_h6.4_static none 6.4 static 0
for c in "1.6 static" "6.4 walkers" "25.6 static" "25.6 walkers"; do set -- $c
  go polite_h$1_$2 polite $1 $2 1 2
  go four_h$1_$2 polite $1 $2 4
done
go reactive_h6.4_walkers reactive 6.4 walkers 1 2
