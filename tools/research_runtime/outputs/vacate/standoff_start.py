"""For each mission: the first stand-off of >= 4 s (as standoff.py), its start time and the robot's path length so far."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('outputs/cohorts'); DT = .5
rows = []
for c in sys.argv[1:]:
    for run in sorted((ROOT/c/'runs').iterdir()):
        if 'attempt' in run.name or not (run/'episode_evaluation.json').exists(): continue
        tr = np.load(run/'native/physics_trace.npz'); o = np.load(run/'native/obstacle_trace.npz')
        ts = np.arange(1., tr['timestamp_s'][-1], DT)
        i = np.minimum(np.searchsorted(tr['timestamp_s'], ts), len(ts)*0+len(tr['timestamp_s'])-1)
        k = np.minimum(np.searchsorted(o['timestamp_s'], ts), len(o['timestamp_s'])-1)
        base = tr['base_pose_world'][i, :2]; wx = o['xy'][k]
        path = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))))
        first = None
        for j in range(wx.shape[1]):
            heading = np.zeros(2); still = 0; run_len = 0
            for t in range(1, len(ts)):
                d = wx[t, j]-wx[t-1, j]
                if np.linalg.norm(d) > 1e-3: heading = d/np.linalg.norm(d); still = 0
                else: still += 1
                r = base[t]-wx[t, j]
                b = still >= 2 and np.linalg.norm(r) < .70 and heading@r > 0
                run_len = run_len+1 if b else 0
                if run_len == 8 and (first is None or t < first): first = t
        if first is not None:
            rows.append((ts[first], path[first], np.linalg.norm(base[first]-base[0])))
r = np.array(rows)
print('missions with a stand-off', len(r), '| start s median', np.median(r[:, 0]).round(0),
      '| path so far < 0.3 m:', int((r[:, 1] < .3).sum()), '| within 1 m of start:', int((r[:, 2] < 1).sum()))
print('path so far percentiles 10/25/50:', np.percentile(r[:, 1], [10, 25, 50]).round(1))
