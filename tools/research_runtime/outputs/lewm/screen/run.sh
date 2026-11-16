#!/bin/bash
# LeWM screening (docs/lewm/STATUS.md): three 15,000-step variants, each then scored with the corpus probe.
# Rerunning resumes: training continues from its checkpoint, finished probes are skipped.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
CORPUS=/mnt/workspace_drive/go2nav_lewm
OUT=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/lewm/screen
variant() {
  name=$1; shift
  echo "[$(date +%H:%M)] $name: train"
  $PY -m go2nav.training.lewm train --corpus $CORPUS --out $OUT/$name "$@" --set optimiser.steps=15000 >> $OUT/$name.train.log 2>&1
  if [ ! -f $OUT/$name.probe.json ]; then
    echo "[$(date +%H:%M)] $name: probe"
    $PY -W ignore -m go2nav.eval.lewm_corpus_probe $OUT/$name/lewm.pt --corpus $CORPUS --out $OUT/$name.probe.json >> $OUT/$name.probe.log 2>&1
  fi
  echo "[$(date +%H:%M)] $name: done"
}
variant inverse --config configs/lewm/go2_inverse.yaml
variant batch4 --config configs/lewm/go2.yaml --set data.mazes_per_batch=4 --set data.windows_per_maze=32
variant paper --config configs/lewm/paper.yaml
echo "[$(date +%H:%M)] screening done"
