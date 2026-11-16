"""Why the model's velocity error is 2.4x the hand tracker's: same walkers, split by track age and true speed."""
import json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from go2nav.eval import walker_compare as wc

def run(path):
    run = Path(path)
    rows, shadow, has = wc.plan_rows(run)
    out = []
    if not rows or not has: return out
    tr = np.load(run/'native/physics_trace.npz'); times, pose = tr['timestamp_s'], tr['base_pose_world']
    ob = run/'native/obstacle_trace.npz'
    if not ob.exists(): return out
    with np.load(ob) as o: wt, wxy = o['timestamp_s'], o['xy']
    walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
    reg = {p['frame']: p['registered_pose'] for p in json.loads((run/'poses.json').read_text())}
    theta = wc.map_turn(run, rows); c, s = math.cos(theta), math.sin(theta)
    for row in rows:
        est = reg.get(row['frame'])
        if est is None: continue
        now = row['measured_ns']/1e9
        i = min(np.searchsorted(times, now), len(times)-1)
        robot, heading = pose[i, :2], float(wc.planar_yaw(pose[i]))
        k = min(np.searchsorted(wt, now), len(wt)-1); b = min(np.searchsorted(wt, now-.4), len(wt)-1)
        truth = wxy[k]; vel = (wxy[k]-wxy[b])/max(wt[k]-wt[b], 1e-3)
        seen = [j for j, w in enumerate(truth) if wc.in_reach(robot, heading, w, walls)]
        if not seen: continue
        tb = wc.body(truth, robot, heading)
        p, R = np.asarray(est['position'], float), np.asarray(est['rotation'], float)
        q = np.array([c*p[0]-s*p[1], s*p[0]+c*p[1]]); mh = math.atan2(R[1, 0], R[0, 0])+theta
        tracks = dict(hand=[t for t in row['selection'].get('obstacle_tracks') or [] if t['motion_updates'] >= 2],
                      model=[t for t in shadow.get(row['frame']) or [] if t.get('confirmed')])
        got = {}
        for name, ts in tracks.items():
            f = np.array([t['centroid_m'] for t in ts], float).reshape(-1, 2)
            fb = wc.body(f, q, mh)
            for a, bb in wc.match(fb, tb[seen]):
                v = wc.turn(np.array(ts[a]['velocity_m_s']), mh)
                got.setdefault(seen[bb], {})[name] = (float(np.linalg.norm(fb[a]-tb[seen[bb]])), float(np.linalg.norm(v-wc.turn(vel[seen[bb]], heading))), float(np.linalg.norm(v)), ts[a].get('motion_updates', 0))
        for j in seen:
            sp = float(np.linalg.norm(vel[j])); g = got.get(j, {})
            out.append((sp, *(g.get('hand', (np.nan,)*4)), *(g.get('model', (np.nan,)*4)), float(np.linalg.norm(tb[j]))))
    return out

if __name__ == '__main__':
    runs = sorted(str(r) for c in ('ws_obst1_c5', 'ws_obst2_c5') for r in (Path('outputs/cohorts')/c/'runs').iterdir())
    with ProcessPoolExecutor(10) as pool: rows = [x for r in pool.map(run, runs, chunksize=2) for x in r]
    a = np.array(rows, float); np.save(sys.argv[1], a)
    sp, hp, hv, hs, hn, mp, mv, ms, mn, dist = a.T
    print('walker-plans', len(a), 'true moving share', (sp > .05).mean(), 'median moving speed', np.median(sp[sp > .05]))
    both = ~np.isnan(hv)&~np.isnan(mv)
    print('both found', both.sum(), 'hand only', (~np.isnan(hv)&np.isnan(mv)).sum(), 'model only', (np.isnan(hv)&~np.isnan(mv)).sum())
    mov = sp > .05
    for nm, m in (('moving, both', both&mov), ('still, both', both&~mov), ('moving, model only', ~np.isnan(mv)&np.isnan(hv)&mov), ('still, model only', ~np.isnan(mv)&np.isnan(hv)&~mov)):
        print(nm, m.sum(), 'hand vel', np.nanmedian(hv[m]) if m.any() else None, 'model vel', np.nanmedian(mv[m]), 'hand pos', np.nanmedian(hp[m]) if m.any() else None, 'model pos', np.nanmedian(mp[m]),
              'model called moving', np.nanmean(ms[m] >= .05), 'hand called moving', np.nanmean((hs[m] >= .05)[~np.isnan(hs[m])]) if (~np.isnan(hs[m])).any() else None)
    print('model zero velocity share (moving walkers)', np.mean(ms[mov&~np.isnan(ms)] == 0))
    for lo, hi in ((3, 8), (8, 12), (12, 20), (20, 40), (40, 1e9)):
        m = mov&(mn >= lo)&(mn < hi)
        print('detections', lo, hi, m.sum(), 'vel err', np.median(mv[m]) if m.any() else None)
