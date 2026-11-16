#!/bin/bash
# After H-JEPA v2 (the corpus-v2 service) frees the GPU: the mover probe on its level 1 and level 2 (CPU, in the
# background), then the GPU check (gpu_check.py) and, if the grid model passes it, the grid LeWM (PLAN P4b) on the
# grid bench worktree's pinned commit. Training resumes from its checkpoint on a restart.
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-grid-bench
H=$W/outputs/hjepa/go2_3level_v2
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $W
until [ -f $H/hjepa.pt ] && ! systemctl --user is-active --quiet corpus-v2; do sleep 60; done
echo "[$(date +%H:%M)] H-JEPA v2 done"
probe() {  # level, out
  OMP_NUM_THREADS=8 PYTHONPATH=$W $PY -W ignore -m go2nav.eval.mover_probe probe /mnt/workspace_drive/go2nav_lewm_v2 \
    --world-model $1 --level $2 --train outputs/mover_probe/labels_train.npz \
    --eval outputs/mover_probe/labels_val.npz --out $3 --device cpu > $3.log 2>&1
}
( [ -f outputs/mover_probe/hjepa_v2_l1/results.json ] || probe $H/lewm.pt 1 outputs/mover_probe/hjepa_v2_l1
  [ -f outputs/mover_probe/hjepa_v2_l2/results.json ] || probe $H/hjepa.pt 2 outputs/mover_probe/hjepa_v2_l2
  echo "[$(date +%H:%M)] H-JEPA probes done" ) &
cd $B
export PYTHONPATH=$B
if [ ! -f $W/outputs/grid/gpu_check.passed ]; then
  $PY -W ignore $W/outputs/grid/gpu_check.py | tee $W/outputs/grid/gpu_check.jsonl || {
    echo "[$(date +%H:%M)] GPU check failed: grid training not started"; wait; exit 1; }
  touch $W/outputs/grid/gpu_check.passed
fi
echo "[$(date +%H:%M)] grid LeWM at $(git -C $B rev-parse --short HEAD)"
$PY -W ignore -m go2nav.training.lewm train --corpus /mnt/workspace_drive/go2nav_lewm_v2 \
  --out $W/outputs/lewm/go2_v2_grid --config configs/lewm/go2_v2_grid.yaml
echo "[$(date +%H:%M)] grid LeWM done"
wait
