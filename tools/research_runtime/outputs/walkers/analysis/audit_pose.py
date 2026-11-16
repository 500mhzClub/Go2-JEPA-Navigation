"""What the robot was doing when visual odometry ended a mission, and whether a walker was the reason."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('outputs/cohorts')
COHORTS = ['ws_obst1_c5', 'ws_obst2_c5', 'wh_obst1_c5', 'wh_obst2_c5', 'wo_obst1_c5', 'wo_obst2_c5', 'wf_ref_c5', 'wf_held_c5',
           'cufh_heldout_obst1_c1', 'cufh_heldout_obst2_c1', 'cufh_heldout_static_c1']
rows = []
for c in COHORTS:
    for run in sorted((ROOT/c/'runs').iterdir()):
        res = json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
        if 'PoseLost' not in (res.get('error') or ''):
            continue
        trace = np.load(run/'native/physics_trace.npz')
        keys = list(trace.keys())
        t = trace['timestamp_s'] if 'timestamp_s' in keys else trace[[k for k in keys if 'time' in k][0]]
        pos_key = 'base_pose_world'
        if False:
            print(keys)
            sys.exit()
        p = trace[pos_key][:, :2]
        out = dict(cohort=c, run=run.name[-14:], t=round(float(t[-1]), 1))
        ob = run/'native/obstacle_trace.npz'
        if ob.exists():
            o = np.load(ob)
            k = list(o.keys())
            xy = o[next(x for x in k if 'pos' in x or 'xy' in x)]
            tt = o[next(x for x in k if 'time' in x or x == 't')] if any('time' in x or x == 't' for x in k) else None
            last = xy[-1].reshape(-1, xy.shape[-1])[:, :2]
            d = np.linalg.norm(last-p[-1], axis=1)
            out['nearest_walker_m'] = round(float(d.min()), 2)
        req = json.loads((run/'requests.json').read_text())[-20:]
        c_ = np.array([r['requested_command'] for r in req])
        out['cmd_last_0.4s'] = np.round(c_[-20:].mean(0), 2).tolist()
        reasons = collections.Counter(r['reason'] for r in req)
        out['reason'] = reasons.most_common(1)[0][0][:30]
        plan = [q for q in json.loads((run/'planning.json').read_text()) if 'action' in q]
        out['route'] = (plan[-1]['route_status'] if plan else '')[:28]
        poses = json.loads((run/'poses.json').read_text())
        v = [q['visual'] for q in poses[-30:] if q.get('visual')]
        if v:
            out['visual_keys'] = None
            fc = [x.get('feature_counts') for x in v[-6:]]
            out['features_last'] = fc[-1] if fc else None
            out['bridge_frames'] = [x.get('bridge_frames') for x in v[-4:]]
        rows.append(out)
for r in rows:
    print(r)
print()
print('n', len(rows), 'walker within 1 m', sum(r.get('nearest_walker_m', 9) < 1 for r in rows), 'within 1.5 m', sum(r.get('nearest_walker_m', 9) < 1.5 for r in rows),
      'turning', sum(abs(r['cmd_last_0.4s'][2]) > .1 for r in rows), 'reversing', sum(r['cmd_last_0.4s'][0] < -.02 for r in rows))
