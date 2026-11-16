#!/bin/bash
# Retrain T7 (C3 motion decoder + matched C4, decoder-fix recipe) on the retrained T4 and T5a: a new feature cache
# (predicted features depend on T4), the three original seeds, then the median-seed choice.
set -euo pipefail
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/training
$PY -m go2nav.training.decoder cache --predictor $O/t4/t4.pt \
  --recordings-root /home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 \
  --recordings-root /mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 \
  --out outputs/features/t7_cache > $O/t7_cache.log 2>&1
for seed in 2026092205 2026093011 2026093012; do
  $PY -m go2nav.training.decoder fit --cache outputs/features/t7_cache --base-readout $O/t5a/readout.pt \
    --out $O/t7_s$seed --seed $seed > $O/t7_s$seed.log 2>&1
done
$PY -m go2nav.training.decoder choose $O/t7_s2026092205 $O/t7_s2026093011 $O/t7_s2026093012 > $O/t7_choice.json
echo "T7 done $(date)"
