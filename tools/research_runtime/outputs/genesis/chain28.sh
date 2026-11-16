#!/bin/bash
# P24 2c (approved 15 November): the smoke from the pinned checkout ../Go2-JEPA-Navigation-p24 (d2880ac): six still
# missions with M5 (mazes 10-12, two episodes each). 6 at once on the R9700; the frozen rule in the environment.
# Resumable (--resume). The one-walker smoke follows only once the still smoke's gate is read.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-p24; G=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/genesis
L=$G/chain.log
log() { echo "[$(date '+%d %b %H:%M')] $*" >> $L; }
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1 PYTORCH_TUNABLEOP_ENABLED=0
export GO2NAV_EXACT=1 GO2NAV_RECOVER=1 GO2NAV_W_STILL=0 GO2NAV_WALKER_PRICE=300
log "chain28 start $(git rev-parse --short HEAD) (the pinned checkout)"
$PY -W ignore -m go2nav.eval.cohort configs/experiments/p24_tapes.yaml --cohort ${1:-tapes_smoke_still_m5} --workers 6 \
  --lewm-lanes 6 --resume >> $G/missions28.log 2>&1
log "chain28 ${1:-tapes_smoke_still_m5} done, exit $?"
