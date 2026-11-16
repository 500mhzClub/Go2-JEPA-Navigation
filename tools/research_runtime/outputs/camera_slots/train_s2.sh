#!/bin/bash
# P23's third run (s2): s0's settings on DINOv2 features of frames upsampled twice (448 x 336: 32 x 24 patches, the
# same units), GPU, full precision; then its binding. The features first (resumable: a finished unit is skipped), then
# training (resumable from its checkpoint), with PyTorch's TunableOp choosing the GEMM kernels (3.4 s a step at 768
# patches on the default kernels, 0.53 s tuned; its choices kept in the run's folder). The decoder runs in blocks of
# frames (models.slots DECODE_ELEMENTS): in one block its gradients came back wrong on the GPU at 768 patches (norm
# 2.5e5 against 0.27 on the CPU; right at 192, so s0 and s1 stand); the first launch, stopped at step 200, is in
# s2_wrong_gradients.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
SL=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-slots; C=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/camera_slots
F=/mnt/workspace_drive/go2nav_dino/features_vits14_x2
cd $SL
echo "[$(date '+%d %b %H:%M')] s2 features start $(git rev-parse --short HEAD) (uncommitted: $(git status --short go2nav configs | wc -l) files)" >> $C/train.log
PYTHONPATH=$SL $PY -W ignore -m go2nav.data.dino_features extract --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $F --scale 2 >> $C/features_x2.log 2>&1
code=$?; echo "[$(date '+%d %b %H:%M')] s2 features exit $code: $(tail -1 $C/features_x2.log | cut -c1-120)" >> $C/train.log
[ $code -eq 0 ] || exit $code
mkdir -p $C/s2
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$C/s2/tunableop.csv
echo "[$(date '+%d %b %H:%M')] s2 start" >> $C/train.log
PYTHONPATH=$SL $PY -W ignore -m go2nav.training.slots train --features $F --out $C/s2 >> $C/s2.log 2>&1
echo "[$(date '+%d %b %H:%M')] s2 exit $?" >> $C/train.log
[ -f $C/s2/model.pt ] && PYTHONPATH=$SL $PY -W ignore -m go2nav.eval.slot_measures score --model $C/s2/model.pt --features $F --corpus /mnt/workspace_drive/go2nav_lewm_v2 --out $C/s2_binding.json > $C/s2_binding.log 2>&1
echo "[$(date '+%d %b %H:%M')] s2 binding: $(tr -d '\n ' < $C/s2_binding.json | cut -c1-400)" >> $C/train.log
