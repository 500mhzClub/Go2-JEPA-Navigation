#!/bin/bash
# mission_check of each decoder-source mission (C3-v3 on-policy round, C3-v2 fresh checks) against its original.
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
CAP=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001
export PYTHONPATH=.
for run in outputs/cohorts/c3v3_onpolicy/runs/c3v3_onpolicy_*_ep[0-9] outputs/cohorts/c3v2_check/runs/c3v2_check_*_ep[0-9]; do
  [ -f $run/result.json ] || continue
  name=$(basename $run)
  if [[ $name == c3v3_onpolicy_* ]]; then
    arm=$(echo $name | sed -E 's/c3v3_onpolicy_([A-Z0-9]+)_.*/\1/'); maze=$(echo $name | sed -E 's/.*_([0-9]+)_ep([0-9]+)$/\1/'); ep=$(echo $name | sed -E 's/.*_ep([0-9]+)$/\1/')
    original=$CAP/runs/c3v3_onpolicy_${arm}_m${maze}_ep${ep}_attempt001
  else
    arm=$(echo $name | sed -E 's/c3v2_check_([A-Z0-9]+)_.*/\1/'); maze=$(echo $name | sed -E 's/.*_([0-9]+)_ep([0-9]+)$/\1/'); ep=$(echo $name | sed -E 's/.*_ep([0-9]+)$/\1/')
    original=$CAP/runs/c3v2_check_${arm}_chk${maze}_ep${ep}_attempt001
  fi
  echo "$name: $($PY tools/porting/mission_check.py $original $run 2>&1 | tr '\n' ';')"
done
