#!/bin/bash
# P22 item 1: the ring tracker's objects over P20's corpus (CPU). Resumable: a finished episode is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $W
echo "[$(date '+%d %b %H:%M')] start $(git rev-parse --short HEAD)" >> outputs/object_dynamics/tracks.log
HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.data.object_tracks build \
    /mnt/workspace_drive/go2nav_ring/corpus_v1 --mazes-root /mnt/workspace_drive/go2nav_lewm_v2/mazes --workers 26 \
    >> outputs/object_dynamics/tracks.log 2>&1
echo "[$(date '+%d %b %H:%M')] exit $?" >> outputs/object_dynamics/tracks.log
