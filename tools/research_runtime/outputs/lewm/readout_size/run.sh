#!/bin/bash
# Readout size study (span8w): corpus readout from U units at window stride S; L4 and the turn gate of each, read alone.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
CORPUS=/mnt/workspace_drive/go2nav_lewm
R=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $R; export PYTHONPATH=$R
W=outputs/lewm/chain_span8w/world/lewm.pt
for us in 12:1 50:4 37:1 150:4 123:1 500:4; do
  u=${us%:*}; s=${us#*:}; out=outputs/lewm/readout_size/u${u}_s${s}
  $PY -W ignore -m go2nav.training.lewm_readout corpus --world-model $W --corpus $CORPUS --units $u --stride $s --out $out/base
  $PY - <<PYEOF
import torch
fit = torch.load('outputs/lewm/chain_span8w/fit_contexts_s2026092205/fit_decoder_fix_s2026092205.pt', map_location='cpu', weights_only=False)
base = torch.load('$out/base/readout.pt', map_location='cpu', weights_only=False)
torch.save(dict(fit, c5=base['rollout']), '$out/fit_base_only.pt')
PYEOF
  [ -f $out/l4.json ] || $PY -W ignore -m go2nav.training.lewm_decoder evaluate --cache outputs/lewm/chain_span8w/cache --checkpoint $out/fit_base_only.pt > $out/l4.json
  [ -f $out/turn_gate.json ] || $PY -W ignore -m go2nav.eval.lewm_turn_gate --world-model $W --corpus $CORPUS --decoder c5=$out/base/readout.pt:rollout --out $out/turn_gate.json > /dev/null
  echo "[$(date +%H:%M)] u$u s$s done"
done
