"""Where the waiting walker sits in the robot's body frame during stand-offs, and whether it is in the forward view."""
import json, sys
from pathlib import Path
import numpy as np
from go2nav.eval.walker_compare import planar_yaw, in_reach
ROOT = Path('outputs/cohorts'); DT = .5
bear, inview, dist, first_seen = [], [], [], []
for c in sys.argv[1:]:
    for run in sorted((ROOT/c/'runs').iterdir()):
        if 'attempt' in run.name or not (run/'episode_evaluation.json').exists(): continue
        tr = np.load(run/'native/physics_trace.npz'); o = np.load(run/'native/obstacle_trace.npz')
        walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
        ts = np.arange(1., tr['timestamp_s'][-1], DT)
        i = np.minimum(np.searchsorted(tr['timestamp_s'], ts), len(tr['timestamp_s'])-1)
        k = np.minimum(np.searchsorted(o['timestamp_s'], ts), len(o['timestamp_s'])-1)
        base = tr['base_pose_world'][i, :2]; yaw = planar_yaw(tr['base_pose_world'][i]); wx = o['xy'][k]
        for j in range(wx.shape[1]):
            heading = np.zeros(2); still = 0; spell = None
            for t in range(1, len(ts)):
                d = wx[t, j]-wx[t-1, j]
                if np.linalg.norm(d) > 1e-3: heading = d/np.linalg.norm(d); still = 0
                else: still += 1
                r = base[t]-wx[t, j]
                b = still >= 2 and np.linalg.norm(r) < .70 and heading@r > 0
                if b:
                    rel = wx[t, j]-base[t]; cy, sy = np.cos(yaw[t]), np.sin(yaw[t])
                    bx, by = cy*rel[0]+sy*rel[1], -sy*rel[0]+cy*rel[1]
                    bear.append(np.degrees(np.arctan2(by, bx))); dist.append(np.hypot(bx, by))
                    iv = in_reach(base[t], float(yaw[t]), wx[t, j], walls); inview.append(iv)
                    if spell is None: spell = []; 
                    spell.append(iv)
                elif spell is not None:
                    if len(spell) >= 8: first_seen.append(spell[0])
                    spell = None
bear, inview, dist = np.abs(np.array(bear)), np.array(inview), np.array(dist)
print('samples', len(bear), 'in view', inview.mean().round(2), 'dist median', np.median(dist).round(2))
for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 135), (135, 181)):
    m = (bear >= lo) & (bear < hi); print(f'|bearing| {lo}-{hi}: {m.mean():.2f} of time, in view {inview[m].mean() if m.any() else 0:.2f}')
print('spells >= 4 s:', len(first_seen), 'walker in view at spell start', np.mean(first_seen).round(2))
