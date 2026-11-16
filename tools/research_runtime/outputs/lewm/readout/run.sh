#!/bin/bash
# Base readouts for C5 (go2nav.training.lewm_readout) and T7's fit grown from them, three seeds, for the idm100 and Go2
# chains, from the corpus and from T7's own contexts. Rerunning resumes: finished readouts and fits are skipped.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
CORPUS=/mnt/workspace_drive/go2nav_lewm
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R
export PYTHONPATH=$R
for model in chain_idm chain; do
  for source in corpus contexts; do
    out=outputs/lewm/readout/${model}_$source
    echo "[$(date +%H:%M)] $model $source: readout"
    if [ $source = corpus ]; then src="--corpus $CORPUS"; else src="--cache outputs/lewm/$model/cache"; fi
    $PY -W ignore -m go2nav.training.lewm_readout $source --world-model outputs/lewm/$model/world/lewm.pt $src --out $out/base
    for seed in 2026092205 2026093011 2026093012; do
      $PY -W ignore -m go2nav.training.lewm_decoder fit --cache outputs/lewm/$model/cache --out $out/fit_s$seed --seed $seed \
        --base-readout $out/base/readout.pt
      $PY -c "import json; e=json.load(open('$out/fit_s$seed/evaluation.json')); print('$model $source $seed C5', round(e['score_mm'],2), 'C6', round(e['score_c6_mm'],2), 'probe', e['probe'])"
    done
  done
done
echo "[$(date +%H:%M)] done"
