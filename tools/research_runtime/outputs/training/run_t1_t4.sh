#!/bin/bash
# Retrain the predictor lineage T1-T4 with the original seeds (development check of the port).
set -euo pipefail
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/training
until grep -q "features done" outputs/features/t1_build.log; do sleep 30; done
$PY -m go2nav.training.native_predictor features --recordings-root /home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 --recordings-root /mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1 --out outputs/features/native > $O/native_features.log 2>&1
$PY -m go2nav.training.t1_temporal train --features outputs/features/t1 --out $O/t1_seed2026080901 > $O/t1.log 2>&1
$PY -m go2nav.training.native_predictor t2 --parent $O/t1_seed2026080901/t1_seed2026080901_epoch21.pt --features outputs/features/native --out $O/t2 > $O/t2.log 2>&1
$PY -m go2nav.training.native_predictor t3 --parent $O/t2/t2.pt --features outputs/features/native --out $O/t3 > $O/t3.log 2>&1
$PY -m go2nav.training.native_predictor t4 --parent $O/t3/t3.pt --features outputs/features/native --out $O/t4 > $O/t4.log 2>&1
echo "T1-T4 done $(date)"
