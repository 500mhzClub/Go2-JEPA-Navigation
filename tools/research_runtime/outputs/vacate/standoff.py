"""Stand-offs with non-yielding walkers: per mission, seconds a walker waits on the robot (still >= 1 s, within 0.70 m of
the base, robot ahead of the walker's last heading), the longest spell, the robot's motion in spells, and the last 2 s
before a contact stop."""
import json, sys
from pathlib import Path
import numpy as np
from go2nav.eval.walker_compare import planar_yaw, in_reach

ROOT = Path('outputs/cohorts')
DT = .1
rows = []
for c in sys.argv[1:]:
    for run in sorted((ROOT/c/'runs').iterdir()):
        if 'attempt' in run.name or not (run/'episode_evaluation.json').exists():
            continue
        res = json.loads((run/'result.json').read_text())
        rt = json.loads((run/'episode_evaluation.json').read_text())['round_trip_success']
        err = str(res.get('error') or '')
        end = 'RT' if rt else ('contact' if 'DISALLOWED' in err else 'poselost' if 'PoseLost' in err else 'timeout')
        tr = np.load(run/'native/physics_trace.npz'); o = np.load(run/'native/obstacle_trace.npz')
        T = tr['timestamp_s'][-1]
        ts = np.arange(1., T, DT)
        i = np.minimum(np.searchsorted(tr['timestamp_s'], ts), len(tr['timestamp_s'])-1)
        k = np.minimum(np.searchsorted(o['timestamp_s'], ts), len(o['timestamp_s'])-1)
        base = tr['base_pose_world'][i, :2]; yaw = planar_yaw(tr['base_pose_world'][i])
        cmd = tr['applied_command'][i]
        wx = o['xy'][k]                                       # (n, W, 2)
        n, W = wx.shape[:2]
        blocked = np.zeros((n, W), bool)
        for j in range(W):
            heading = np.zeros(2); still = 0
            for t in range(1, n):
                d = wx[t, j]-wx[t-1, j]
                if np.linalg.norm(d) > 1e-3:
                    heading = d/np.linalg.norm(d); still = 0
                else:
                    still += 1
                r = base[t]-wx[t, j]
                blocked[t, j] = still >= 10 and np.linalg.norm(r) < .70 and heading@r > 0
        any_b = blocked.any(1)
        spells, run_len = [], 0
        for b in any_b:
            if b: run_len += 1
            elif run_len: spells.append(run_len*DT); run_len = 0
        if run_len: spells.append(run_len*DT)
        row = dict(c=c, m=run.name.rsplit('_C5_', 1)[1], end=end, T=T, blocked_s=any_b.sum()*DT,
                   longest_s=max(spells, default=0.), spells=len(spells),
                   reverse_in_b=float((cmd[any_b, 0] < -.02).mean()) if any_b.any() else np.nan,
                   still_in_b=float((np.abs(cmd[any_b]).max(1) < .02).mean()) if any_b.any() else np.nan,
                   blocked_at_end=bool(any_b[-20:].any()))
        if end == 'contact':
            last = slice(max(0, n-20), n)
            row['pre_vx'] = float(cmd[last, 0].mean()); row['pre_wz'] = float(np.abs(cmd[last, 2]).mean())
            hit = next(x for x in json.loads((run/'native/contact_events.json').read_text()) if x['disallowed_contacts'])
            raw = hit['disallowed_contacts'][0]; oid = raw['environment_object_id']
            row['link'] = raw['robot_link_name']; row['obj'] = 'walker' if oid.startswith('obstacle') else 'wall'
            if oid.startswith('obstacle'):
                j = int(oid.split('_')[-1])
                row['walker_moving'] = bool(np.linalg.norm(wx[-1, j]-wx[-6, j]) > .01)
                row['walker_was_blocked'] = bool(blocked[-30:, j].any())
                walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
                row['in_view_last_1s'] = any(in_reach(base[t], float(yaw[t]), wx[t, j], walls) for t in range(n-10, n))
        rows.append(row)
        print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
json.dump(rows, open(sys.argv[0].replace('.py', f'_{sys.argv[1]}.json'), 'w'), default=float)
