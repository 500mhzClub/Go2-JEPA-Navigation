#!/bin/bash
# The full Go2 chain for the inverse-dynamics variant that passed screening (docs/lewm/STATUS.md): waits for
# screening round 2, takes idm100 or idm10, whichever reads lower on the probe's c5 head (validation mazes, 800 ms),
# and trains it for 60,000 steps into outputs/lewm/chain_idm. Rerunning resumes (the chain skips finished stages).
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
M=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
S=$M/outputs/lewm/screen
while systemctl --user is-active -q lewm-screen2; do sleep 300; done
choice=$M/outputs/lewm/chain_idm.choice
if [ ! -f $choice ]; then
  $PY - $S > $choice <<'PYEOF'
import json, sys
from pathlib import Path
scores = {}
for name in ('idm100', 'idm10'):
    f = Path(sys.argv[1])/f'{name}.probe.json'
    if f.exists():
        scores[name] = json.loads(f.read_text())['results']['c5']['val@800ms']['median_xy_mm']
best = min(scores, key=scores.get)
print(best if scores[best] <= 9.5 else 'none')
PYEOF
fi
pick=$(cat $choice)
echo "[$(date +%H:%M)] chosen: $pick"
case $pick in
  idm100) extra=() ;;
  idm10) extra=(--set objective.inverse_weight=10.) ;;
  *) echo "no variant under 9.5 mm"; exit 0 ;;
esac
exec $PY -m go2nav.training.lewm_chain $M/outputs/lewm/chain_idm --corpus /mnt/workspace_drive/go2nav_lewm \
  --recordings-root $M/outputs/recordings --config configs/lewm/go2_idm100.yaml "${extra[@]}"
