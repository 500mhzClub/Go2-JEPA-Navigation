#!/bin/bash
# record.sh NAME [recorder options] -- mission args : one replay bundle from the snapshot (porting).
S=~/Workspace/LeWMQuad-v3-snapshot-2026-10-06
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
G=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
name=$1; shift
cd $S && PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 PYTHONPATH=.:lewm_genesis:lewm_worlds $PY $G/tools/porting/replay_bundle.py record --out $G/outputs/porting/bundles/$name "$@" > $G/outputs/porting/bundles/$name.log 2>&1
echo "$name exit $?" >> $G/outputs/porting/bundles/record_done.log
