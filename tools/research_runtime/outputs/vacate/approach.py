"""Before each stand-off (>= 4 s, as standoff.py): how long the walker was in the forward view and approaching from 2.5 m."""
import json, sys
from pathlib import Path
import numpy as np
from go2nav.eval.walker_compare import planar_yaw, in_reach
ROOT = Path('outputs/cohorts'); DT = .25
seen_s, approach_s, seen_far = [], [], []
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
            d = np.linalg.norm(wx[:, j]-base, axis=1)
            still = np.r_[False, np.linalg.norm(np.diff(wx[:, j], axis=0), axis=1) < 1e-3]
            parked = still & (d < .70)
            t = 0
            while t < len(ts):
                if parked[t:t+16].all() and len(parked[t:t+16]) == 16 and (t == 0 or not parked[t-1]):
                    a = t
                    while a > 0 and d[a-1] < 2.5 and d[a-1] >= d[a]-.02: a -= 1      # back to where it started closing in
                    window = range(a, t)
                    vis = [in_reach(base[u], float(yaw[u]), wx[u, j], walls) for u in window]
                    approach_s.append(len(window)*DT); seen_s.append(sum(vis)*DT)
                    seen_far.append(sum(v for u, v in zip(window, vis) if d[u] >= 1.0)*DT)
                    t += 16
                else:
                    t += 1
a, s, f = map(np.array, (approach_s, seen_s, seen_far))
print('stand-offs', len(a), '| approach from <=2.5 m, s: median', np.median(a), '| seen during approach, s: median', np.median(s),
      '| seen while >= 1 m away, s: median', np.median(f), '| share with >= 2 s seen at >= 1 m', round(float((f >= 2).mean()), 2),
      '| never seen', round(float((s == 0).mean()), 2))
