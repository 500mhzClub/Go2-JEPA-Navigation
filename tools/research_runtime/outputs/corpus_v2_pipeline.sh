#!/bin/bash
# Corpus v2 then the v2 world models (docs/dynamic/PLAN.md, P3-P4), on the bench worktree's pinned commit. Resumable:
# collection skips finished units and retries failed ones; each later stage is skipped once its product exists;
# training resumes from its last checkpoint. Stops (exit 1) if a data gate fails, before any training.
set -euo pipefail
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-lewm-bench
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
R=/mnt/workspace_drive/go2nav_lewm_v2
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
cd $B
export PYTHONPATH=$B PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1
echo "[$(date +%H:%M)] corpus v2 pipeline at $(git -C $B rev-parse --short HEAD)"
mkdir -p $R
[ -d $R/mazes ] || cp -r /mnt/workspace_drive/go2nav_lewm/mazes $R/
if [ ! -f $R/.collected ]; then
  for attempt in 1 2 3; do
    if $PY -W ignore -m go2nav.data.lewm_corpus_v2 collect $R --workers 30 --egl-devices 1 >> $R/collect.log 2>&1; then
      touch $R/.collected; break
    fi
    echo "[$(date +%H:%M)] collection attempt $attempt left failures; retrying"
  done
  [ -f $R/.collected ] || { echo "collection failed three times"; exit 1; }
fi
echo "[$(date +%H:%M)] collected"
for L in 12 257; do
  [ -f $R/index_$L.npz ] || $PY -W ignore -m go2nav.data.lewm_corpus_v2 index $R --length $L
done
[ -f $R/audit.json ] && grep -q '"dynamic"' $R/audit.json || $PY -W ignore -m go2nav.data.lewm_corpus_v2 audit $R > /dev/null
# gates_accepted.json: {gate: reason} for a gate missed and accepted on review (logged with its reason)
$PY - $R/audit.json $R/gates_accepted.json <<'EOF'
import json, os, sys
report = json.load(open(sys.argv[1]))
gates = dict(report['gate'], **report['dynamic']['gate'])
accepted = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
print(json.dumps(gates))
for gate in accepted:
    if not gates.get(gate, True):
        print(f'accepted {gate}: {accepted[gate]}')
sys.exit(0 if all(ok or gate in accepted for gate, ok in gates.items()) else 1)
EOF
$PY -W ignore -m go2nav.data.lewm_corpus_v2 stats $R
cp $B/data/training/control_norm_stats_corpus_v2.json $W/data/training/
echo "[$(date +%H:%M)] LeWM v2"
$PY -W ignore -m go2nav.training.lewm train --corpus $R --out $W/outputs/lewm/go2_v2 --config configs/lewm/go2_v2.yaml
echo "[$(date +%H:%M)] H-JEPA v2"
$PY -W ignore -m go2nav.training.hjepa train --corpus $R --out $W/outputs/hjepa/go2_3level_v2 \
  --config configs/hjepa/go2_3level_v2.yaml
echo "[$(date +%H:%M)] corpus v2 pipeline done"
