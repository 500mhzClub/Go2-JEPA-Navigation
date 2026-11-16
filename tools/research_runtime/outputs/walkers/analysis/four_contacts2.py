import json, sys, math
from pathlib import Path
import numpy as np
from go2nav.eval.walker_compare import body, planar_yaw, in_reach, map_turn
for name in sys.argv[1:]:
    run = Path('outputs/cohorts')/name.split('_C5')[0]/'runs'/name
    hit = next(x for x in json.loads((run/'native/contact_events.json').read_text()) if x['disallowed_contacts'])
    t = hit['timestamp_s']; raw = hit['disallowed_contacts'][0]
    j = int(raw['environment_object_id'].split('_')[-1])
    tr = np.load(run/'native/physics_trace.npz'); o = np.load(run/'native/obstacle_trace.npz')
    walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
    print(name, 't', round(t, 2), raw['robot_link_name'], raw['environment_object_id'], 'mission frac', round(t/tr['timestamp_s'][-1], 2))
    rows = [p for p in json.loads((run/'planning.json').read_text()) if 'selection' in p]
    registered = {p['frame']: p['registered_pose'] for p in json.loads((run/'poses.json').read_text())}
    theta = map_turn(run, rows); c, s = math.cos(theta), math.sin(theta)
    for dt in (-5, -4, -3, -2, -1.5, -1, -.5, -.2, 0):
        now = t+dt
        i = min(np.searchsorted(tr['timestamp_s'], now), len(tr['timestamp_s'])-1)
        k = min(np.searchsorted(o['timestamp_s'], now), len(o['timestamp_s'])-1); b = max(np.searchsorted(o['timestamp_s'], now-.4), 0)
        robot, h = tr['base_pose_world'][i, :2], float(planar_yaw(tr['base_pose_world'][i]))
        w = o['xy'][k, j]; v = (o['xy'][k, j]-o['xy'][b, j])/max(o['timestamp_s'][k]-o['timestamp_s'][b], 1e-3)
        bx, by = body(w, robot, h)
        tw = tr['base_twist_world'][i, :2]
        fwd = math.cos(h)*tw[0]+math.sin(h)*tw[1]
        # closest confirmed track at the latest plan
        pl = [p for p in rows if p['measured_ns']/1e9 <= now]
        gap = None; n = 0
        if pl:
            p = pl[-1]; est = registered.get(p['frame'])
            if est is not None:
                q = np.asarray(est['position'], float); q = np.array([c*q[0]-s*q[1], s*q[0]+c*q[1]])
                head = math.atan2(np.asarray(est['rotation'])[1, 0], np.asarray(est['rotation'])[0, 0])+theta
                ts = [x for x in p['selection'].get('obstacle_tracks') or [] if x['motion_updates'] >= 2]
                n = len(ts)
                if ts:
                    f = body(np.array([x['centroid_m'] for x in ts], float).reshape(-1, 2), q, head)
                    gap = float(np.linalg.norm(f-np.array([bx, by]), axis=1).min())
            act = p['selection'].get('action')
        else:
            act = None
        print(f'  dt {dt:5.1f} walker body x {bx:5.2f} y {by:5.2f} dist {math.hypot(bx, by):4.2f} speed {np.linalg.norm(v):4.2f} in_view {in_reach(robot, h, w, walls)} robot fwd {fwd:5.2f} act {act} tracks {n} gap_to_nearest_track {None if gap is None else round(gap, 2)}')
