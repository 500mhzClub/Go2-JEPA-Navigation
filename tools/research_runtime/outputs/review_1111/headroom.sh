#!/bin/bash
# Review of 11 November (second): where is the headroom for a learned walker model? P17's planner (6.4 s, walkers
# terminal) on suites not yet laddered. CPU only. Resumable.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/review_1111
cd $W
go() {  # name, set, kind, walkers (quoted), model
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions run --out $O/$1 --set $2 \
      --kind $3 --horizon 6.4 --terminal walkers --walkers $4 --workers 16 --model $5 > $O/$1.log 2>&1
  echo "$1 :: $(PYTHONPATH=. $PY -W ignore -m go2nav.eval.kinematic_missions summary $O/$1 | cut -c1-170)" >> $O/headroom.log
}
# the quick rows first (the ring models), then the slow ones (the simulator's rules stepped for every tape)
for m in ring_frozen ring_velocity; do go loops_rude_$m loops_rooms_v1 rude "1 2" $m; done
go rude4_ring_velocity validation rude 4 ring_velocity
go reactive4_ring_velocity validation reactive 4 ring_velocity
go loops_reactive_ring_velocity loops_rooms_v1 reactive "1 2" ring_velocity
go loops_rude_true loops_rooms_v1 rude "1 2" true
go rude4_true validation rude 4 true
go reactive12_fresh validation reactive "1 2" fresh
go reactive12_true validation reactive "1 2" true
go reactive4_true validation reactive 4 true
go loops_reactive_true loops_rooms_v1 reactive "1 2" true
echo done >> $O/headroom.log
