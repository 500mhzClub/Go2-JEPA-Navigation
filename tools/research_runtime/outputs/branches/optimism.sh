#!/bin/bash
# P22 item 10's last measure: tape optimism on four walkers that do not yield (seed 0), N3's expected-cost planner and
# the ring tracker. CPU, 6 workers beside the branch build.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; O=$W/outputs/branches
cd $W
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $O/optimism.log; }
for pair in n3expect:expect:outputs/object_dynamics/n3/model.pt ring_velocity:ring_velocity true:true; do
  name=${pair%%:*}; spec=${pair#*:}
  HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.eval.tape_optimism run --model $spec \
      --out $O/optimism_$name --every 5 --limit 4 --workers 6 > $O/optimism_$name.log 2>&1
  log "$name: $(tr -d '\n ' < $O/optimism_$name/summary.json)"
done
log done
