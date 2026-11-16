#!/bin/bash
# After the grid LeWM (grid-lewm): its mover probe (CPU, in the background), then LeWM v2 retrained with the LayerNorm
# fix (go2_v2's config, the grid bench's pinned commit) and its mover probe. Training resumes from its checkpoint.
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-grid-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $W
until [ -f outputs/lewm/go2_v2_grid/lewm.pt ] && ! systemctl --user is-active --quiet grid-lewm; do sleep 120; done
echo "[$(date +%H:%M)] grid LeWM done"
probe() {  # world model, out
  OMP_NUM_THREADS=12 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.mover_probe probe /mnt/workspace_drive/go2nav_lewm_v2 \
    --world-model $1 --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz \
    --out $2 --device cpu > $2.log 2>&1
}
( [ -f outputs/mover_probe/go2_v2_grid/results.json ] || probe outputs/lewm/go2_v2_grid/lewm.pt outputs/mover_probe/go2_v2_grid
  echo "[$(date +%H:%M)] grid probe done" ) &
echo "[$(date +%H:%M)] LeWM v2 (LayerNorm fix) at $(git -C $B rev-parse --short HEAD)"
cd $B
PYTHONPATH=$B $PY -W ignore -m go2nav.training.lewm train --corpus /mnt/workspace_drive/go2nav_lewm_v2 \
  --out $W/outputs/lewm/go2_v2_lnfix --config configs/lewm/go2_v2.yaml
cd $W
echo "[$(date +%H:%M)] LeWM v2 (LayerNorm fix) done"
[ -f outputs/mover_probe/go2_v2_lnfix/results.json ] || probe outputs/lewm/go2_v2_lnfix/lewm.pt outputs/mover_probe/go2_v2_lnfix
echo "[$(date +%H:%M)] LeWM v2 (LayerNorm fix) probe done"
wait
