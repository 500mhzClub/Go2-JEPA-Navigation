#!/bin/bash
# LeWM screening, round 2 (docs/lewm/STATUS.md): inverse dynamics at H-JEPA's level-1 weight (100, SIGReg 0.08) and
# at 10, 15,000 steps each, each then scored with the corpus probe. Rerunning resumes.
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
variant idm100 --config configs/lewm/go2_idm100.yaml
variant idm10 --config configs/lewm/go2_idm100.yaml --set objective.inverse_weight=10.
echo "[$(date +%H:%M)] screening round 2 done"
