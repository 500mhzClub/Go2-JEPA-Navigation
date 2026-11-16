#!/bin/bash
# Retrain the T5 readouts with the original seeds, after the T1-T4 chain.
set -euo pipefail
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/training
until grep -q "T1-T4 done" $O/run_t1_t4.log 2>/dev/null; do sleep 60; done
$PY -m go2nav.training.readouts features --recordings-root /home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 --recordings-root /mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 --out outputs/features/readouts > $O/readout_features.log 2>&1
$PY -m go2nav.training.readouts t5-0 --features outputs/features/readouts --out $O/t5_0 > $O/t5_0.log 2>&1
$PY -m go2nav.training.readouts t5a --parent $O/t5_0/readout.pt --features outputs/features/readouts --out $O/t5a > $O/t5a.log 2>&1
$PY -m go2nav.training.readouts t5b --parent $O/t5a/readout.pt --features outputs/features/readouts --out $O/t5b > $O/t5b.log 2>&1
echo "T5 done $(date)"
