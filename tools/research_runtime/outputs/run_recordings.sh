#!/bin/bash
# Regenerate every collection case, then compare each with its original (worktree at the recordings-driver commit).
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-data
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
echo "recordings start $(git rev-parse --short HEAD) $(date)"
$PY -m go2nav.data.recordings outputs/recordings --workers 6
echo "recordings done $(date)"
$PY tools/porting/collection_check.py outputs/recordings --workers 12
echo "check done $(date)"
