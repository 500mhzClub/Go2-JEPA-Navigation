#!/bin/bash
# check_all.sh BUNDLE... : run every perception replay check on each bundle
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
for b in "$@"; do for c in pose registration obstacles mapping; do
  echo "== $b $c"; $PY tools/porting/replay_check.py $c outputs/porting/bundles/$b 2>&1 | grep -v Warning | grep -v ": ok (" | tail -3
done; done
