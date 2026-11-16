#!/bin/bash
# P9 after stage A, on the pinned worktree: stage B, the forecast heads on its world model, the memory's scores against its
# bar (PLAN P9), the missions' threshold, then the missions if the bar passes. Each stage is skipped once its product
# exists, so a rerun resumes after a crash or reboot.
set -uo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-memory-bench
C=/mnt/workspace_drive/go2nav_lewm_v2
M=outputs/memory
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
log() { echo "[$(date +%H:%M)] $*" >> $M/pipeline.log; }
while systemctl --user is-active --quiet memory-train-fixed; do sleep 60; done
[ -f $M/fixed/memory.pt ] || { log "stage A left no memory.pt"; exit 1; }
[ -f $M/fixed/evaluation.json ] || $PY -W ignore -m go2nav.training.memory evaluate $C --states $M/states --memory $M/fixed/memory.pt >> $M/fixed.log 2>&1
log "stage A scored"
if [ ! -f $M/joint/memory.pt ]; then
  log "stage B start"
  $PY -W ignore -m go2nav.training.memory_joint --corpus $C --config configs/memory/joint.yaml --out $M/joint >> $M/joint.log 2>&1 \
    || { log "stage B failed"; exit 1; }
fi
log "stage B done"
if [ ! -f $M/joint_heads/heads.pt ]; then
  $PY -W ignore -m go2nav.training.lewm_heads_v2 $M/joint/lewm.pt --corpus $C --out $M/joint_heads --device cuda:0 >> $M/heads.log 2>&1 \
    || { log "heads failed"; exit 1; }
fi
log "heads done"
$PY -W ignore -m go2nav.training.memory cache $C --world-model $M/joint/lewm.pt --out $M/states_joint >> $M/cache_joint.log 2>&1 \
  || { log "cache failed"; exit 1; }
[ -f $M/joint/evaluation_memory.json ] || $PY -W ignore -m go2nav.training.memory evaluate $C --states $M/states_joint \
  --memory $M/joint/memory.pt --out $M/joint/evaluation_memory.json >> $M/joint.log 2>&1
verdict=$($PY - <<'P'
import json
m = json.load(open('outputs/memory/joint/evaluation_memory.json'))
probe = json.load(open('outputs/memory/probe.json'))
out = m['out_of_view']['all']
best = max(out[b] for b in ('forget', 'hold', 'constant_velocity', 'prior'))
inside, now = m['in_view_probe_frames']['all']['memory'], probe['in_view']['all']['now']
print('PASS' if out['memory'] > best and inside >= now else 'FAIL',
      f"out of view {out['memory']} against {best}; in view {inside} against {now}")
P
)
log "memory bar: $verdict"
case "$verdict" in PASS*) ;; *) exit 0;; esac
$PY -W ignore -m go2nav.training.memory calibrate $C --states $M/states_joint --memory $M/joint/memory.pt >> $M/joint.log 2>&1
log "threshold: $(tail -1 $M/joint.log)"
log "missions start"
$PY -W ignore -m go2nav.eval.cohort configs/experiments/walker_memory.yaml --workers 6 --lewm-lanes 6 --resume >> $M/missions.log 2>&1
log "missions ended: $?"
