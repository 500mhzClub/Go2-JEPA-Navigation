#!/bin/bash
# The module oracles on the oracle worktree (its own commit): map_static at once, the rest after the density curve.
set -euo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
C=configs/experiments/module_oracles.yaml
launch() {
  systemd-run --user --unit=or-${1//_/-} -p LimitNOFILE=1048576 -E BENCH=$W-oracle \
    -p StandardOutput=append:$W/outputs/dynamic/or-$1.log -p StandardError=append:$W/outputs/dynamic/or-$1.log \
    bash $W/outputs/dynamic_cohorts.sh $C 7 --cohort or_${1}_obst1_c1 --cohort or_${1}_obst2_c1
}
echo "[$(date +%H:%M:%S)] oracle worktree at $(git -C $W-oracle rev-parse --short HEAD)"
systemctl --user is-active -q or-map-static || launch map_static
while systemctl --user is-active -q dn-obst4 dn-fast1 dn-fast2 dn-fast4; do sleep 60; done
for o in pose pred map_full; do launch $o; done
