#!/bin/bash
# PLAN P4f: once the give-way missions have left the GPU, collect the give-way top-up of corpus v2 beside P4d's training
# (go2_v2_grid_movers; 1,200 units, 4-6 h; ten renderers fit beside it under the 85% VRAM cap). After that training, rebuild
# the window index, which a resuming trainer rereads, check the data gate (a quarter of the waiting windows before a
# retreat), then once P4d's probe has left the GPU train go2_v2_frozen_giveway, and probe and read it out.
set -uo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-giveway-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
C=/mnt/workspace_drive/go2nav_lewm_v2
O=outputs/lewm/go2_v2_frozen_giveway
P=outputs/mover_probe/go2_v2_frozen_giveway
L=$W/outputs/grid/giveway.log
cd $B
until ! systemctl --user is-active --quiet give-way-probe; do sleep 60; done
echo "[$(date +%H:%M)] missions done; collecting the give-way units" >> $L
EGL_DEVICE_ID=1 PYTHONPATH=$B $PY -W ignore -m go2nav.data.lewm_corpus_v2 collect $C --roles lewm_train --kinds giveway \
  --workers 10 --egl-devices 1 >> $W/outputs/grid/giveway_collect.log 2>&1 || { echo "collection failed" >> $L; exit 1; }
echo "[$(date +%H:%M)] collected; waiting for P4d's training" >> $L
until [ -f outputs/lewm/go2_v2_grid_movers/lewm.pt ]; do sleep 120; done
PYTHONPATH=$B $PY -W ignore -m go2nav.data.lewm_corpus_v2 index $C --length 12 > $W/outputs/grid/giveway_index.json 2>&1 \
  || { echo "index failed" >> $L; exit 1; }
PYTHONPATH=$B $PY -W ignore - >> $L 2>&1 <<'EOF' || { echo "data gate failed" >> $L; exit 1; }
import glob, json, numpy as np
from go2nav.eval import mover_probe as mp
n = back = ahead = 0
for f in sorted(glob.glob('/mnt/workspace_drive/go2nav_lewm_v2/units/lewm_train/*_giveway')):
    s, d = np.load(f+'/steps.npz'), np.load(f+'/dynamic.npz')
    pose, ex, xy, w = s['pose'], s['executed'], d['obstacles_xy'], d['waiting']
    for t in range(len(pose)-16):
        for k in np.flatnonzero(w[t]):
            b = mp.body_frame(pose[t], xy[t, k])
            if 0 < b[0] and abs(b[1]) < .75 and np.hypot(*b) < 1.5:
                v = ex[t:t+16, :, 0].mean()
                n, back, ahead = n+1, back+(v < -.03), ahead+(v > .03)
print(json.dumps(dict(waiting_windows=n, retreat_share=round(back/max(n, 1), 3), advance_share=round(ahead/max(n, 1), 3))))
raise SystemExit(0 if back >= .25*n else 1)
EOF
until [ -f outputs/mover_probe/go2_v2_grid_movers/results_grid.json ] || ! systemctl --user is-active --quiet p4d-after-frozen; do sleep 120; done   # P4d's GPU probe done
echo "[$(date +%H:%M)] training go2_v2_frozen_giveway" >> $L
PYTHONPATH=$B $PY -W ignore -m go2nav.training.lewm train --corpus $C --out $O --config configs/lewm/go2_v2_frozen_giveway.yaml \
  >> $W/outputs/grid/giveway_train.log 2>&1 || { echo "training failed" >> $L; exit 1; }
echo "[$(date +%H:%M)] probe" >> $L
[ -f $P/results.json ] || OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.mover_probe probe $C \
  --world-model $O/lewm.pt --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz \
  --out $P --device cuda:0 > $P.log 2>&1
[ -f $P/results_grid.json ] || OMP_NUM_THREADS=16 PYTHONPATH=$B $PY -W ignore -m go2nav.eval.grid_readout $P \
  --world-model $O/lewm.pt --train outputs/mover_probe/labels_train.npz --eval outputs/mover_probe/labels_val.npz \
  --threads 16 > ${P}_readout.log 2>&1
echo "[$(date +%H:%M)] probe and readout done" >> $L
