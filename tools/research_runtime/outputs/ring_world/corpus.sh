#!/bin/bash
# P20 item 4: the ring corpus (CPU only). Resumable: a finished episode is skipped.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
cd $W
echo "[$(date '+%d %b %H:%M:%S')] start $(git rev-parse --short HEAD)" >> outputs/ring_world/corpus.log
HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. $PY -W ignore -m go2nav.data.ring_corpus build \
    --mazes-root /mnt/workspace_drive/go2nav_lewm_v2/mazes --out /mnt/workspace_drive/go2nav_ring/corpus_v1 --workers 28 \
    >> outputs/ring_world/corpus.log 2>&1
echo "[$(date '+%d %b %H:%M:%S')] exit $?" >> outputs/ring_world/corpus.log
