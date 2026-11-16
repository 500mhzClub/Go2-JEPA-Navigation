#!/bin/bash
# P14 item 4 (approved 7 November): the stand-off top-up into corpus v2, on the phase-b worktree (driver of 2de1826,
# gate amended at ed0cbb5; pilot 4's 48 units, mazes 0-23, are its first).
# 1,200 training units (mazes 0-599, family A `standoff` and family B `standoff_b`) and 100 validation (1200-1249).
# Resumable: rerun. 16 workers beside the phase A reference missions; rendering on the R9700.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b
export EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl
L=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/phase_b/standoff_collect.log
echo "[$(date +%H:%M)] start $(git rev-parse --short HEAD)" >> $L
$PY -W ignore -m go2nav.data.lewm_corpus_v2 collect /mnt/workspace_drive/go2nav_lewm_v2 --roles lewm_train --mazes 0-599 \
    --kinds standoff standoff_b --workers 16 --egl-devices 1 >> $L 2>&1
$PY -W ignore -m go2nav.data.lewm_corpus_v2 collect /mnt/workspace_drive/go2nav_lewm_v2 --roles lewm_val --mazes 1200-1249 \
    --kinds standoff standoff_b --workers 16 --egl-devices 1 >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
