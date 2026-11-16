#!/bin/bash
# span8w: corpus base readout and T7's fit grown from it, three seeds (as run.sh). Rerunning resumes.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
CORPUS=/mnt/workspace_drive/go2nav_lewm
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
export PYTHONPATH=$R
out=outputs/lewm/readout/chain_span8w_corpus
echo "[$(date +%H:%M)] span8w corpus readout"
$PY -W ignore -m go2nav.training.lewm_readout corpus --world-model outputs/lewm/chain_span8w/world/lewm.pt --corpus $CORPUS --out $out/base
for seed in 2026092205 2026093011 2026093012; do
  $PY -W ignore -m go2nav.training.lewm_decoder fit --cache outputs/lewm/chain_span8w/cache --out $out/fit_s$seed --seed $seed \
    --base-readout $out/base/readout.pt
  $PY -c "import json; e=json.load(open('$out/fit_s$seed/evaluation.json')); print('span8w corpus $seed C5', round(e['score_mm'],2), 'C6', round(e['score_c6_mm'],2), 'probe', e['probe'])"
done
echo "[$(date +%H:%M)] done"
