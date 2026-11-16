"""Gate 2's label (contact at t+0.8 s): how well does the present alone score, with no prediction at all?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b')
from go2nav.eval import mover_probe as mp
ROOT = Path('/mnt/workspace_drive/go2nav_lewm_v2'); HEADS = Path('/mnt/workspace_drive/go2nav_lewm_v2_heads')
H, STRIDE, n = 8, 5, 4
rows = {'val': [], 'transfer': []}
for folder in sorted((ROOT/'units/lewm_val').iterdir()):
    path = f'units/lewm_val/{folder.name}'
    hf = HEADS/path/'heads.npz'
    if not hf.exists():
        continue
    c = np.load(hf)['clearance'].astype(np.float32)
    with np.load(folder/'steps.npz') as s:
        valid = s['block_valid']
    t = np.array([k for k in range(n-1, len(valid)-H, STRIDE) if valid[k-n+1:k+H].all()], int)
    if not len(t):
        continue
    split = 'transfer' if folder.name.endswith('_transfer') else 'val'
    rows[split].append(np.stack([c[t], c[t+H]], 1))
for split, r in rows.items():
    r = np.concatenate(r)
    now, later = r[:, 0], r[:, 1]
    label = later < .10
    free_now = now >= .10
    print(split, 'windows', len(r), 'contact share', round(float(label.mean()), 4),
          '| AUC of true clearance at t for contact at t+0.8s:', round(mp.auc(-now, label), 3),
          '| AUC of the binary contact-now flag:', round(mp.auc((now < .10).astype(float), label), 3),
          '| share of future contacts already in contact at t:', round(float((~free_now[label]).mean()), 3),
          '| new contacts (not in contact at t) share of windows:', round(float((label & free_now).mean()), 4))
