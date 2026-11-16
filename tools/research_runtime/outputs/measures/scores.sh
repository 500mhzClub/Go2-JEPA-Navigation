#!/bin/bash
# P16: the measures on go2_3level_v2, the control and go2_3level_c's 40k checkpoint, on the R9700, once the tapes exist.
# Resumable: each model's encoded features are cached in its folder; a finished model (measures.json with its verdicts) is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation; C=$W/outputs/phase_c; M=$W/outputs/measures; H=$W/outputs/hjepa
cd $W
while systemctl --user is-active -q measures-tapes; do sleep 20; done
score() {  # name, run folder, gates folder, extra arguments
  name=$1; run=$2; gates=$3; shift 3
  grep -q '"measures"' $M/$name/measures.json 2>/dev/null && return
  echo "[$(date '+%d %b %H:%M')] $name $(git rev-parse --short HEAD)" >> $M/scores.log
  PYTHONPATH=. $PY -W ignore -m go2nav.eval.measures run $run --root /mnt/workspace_drive/go2nav_lewm_v2 --gates $gates --out $M/$name \
      --train-labels $C/give_way_labels_train.npz --eval-labels $C/give_way_labels_val.npz \
      --train-tapes $M/side_tapes_train.npy --eval-tapes $M/side_tapes_val.npy \
      --heads-root /mnt/workspace_drive/go2nav_lewm_v2_heads --device cuda:0 "$@" >> $M/scores.log 2>&1
  echo "[$(date '+%d %b %H:%M')] exit $? $name" >> $M/scores.log
}
score control $H/go2_3level_p14_control $C/gates_control
score v2 $H/go2_3level_v2 $C/gates_v2
score c40k $C/interim_c40k/run $C/interim_c40k --obs-root /mnt/workspace_drive/go2nav_lewm_v2_obs
