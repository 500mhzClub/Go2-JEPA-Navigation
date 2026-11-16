"""Does a confirmed model track's disk reach the wall the robot later hit? The same test on the shadow's tracks, which never touched the map."""
import json, math, glob, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from go2nav.eval import walker_compare as wc
from go2nav.sim.obstacles import RADIUS_M
CELL = .05
def one(path):
    run = Path(path)
    rows = [p for p in json.loads((run/'planning.json').read_text()) if 'selection' in p]
    if not rows or not (run/'walker_shadow.json').exists(): return None
    tr = np.load(run/'native/physics_trace.npz'); times, pose = tr['timestamp_s'], tr['base_pose_world']
    reg = {p['frame']: p['registered_pose'] for p in json.loads((run/'poses.json').read_text())}
    theta = wc.map_turn(run, rows); c_, s_ = math.cos(theta), math.sin(theta)
    walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
    sh = json.loads((run/'walker_shadow.json').read_text())
    pts, ts = [], []
    for x in sh:
        e = reg.get(x['frame'])
        if e is None: continue
        i = min(np.searchsorted(times, x['measured_ns']/1e9), len(times)-1)
        rob, head = pose[i, :2], float(wc.planar_yaw(pose[i]))
        p_, R = np.asarray(e['position'], float), np.asarray(e['rotation'], float)
        q = np.array([c_*p_[0]-s_*p_[1], s_*p_[0]+c_*p_[1]]); mh = math.atan2(R[1, 0], R[0, 0])+theta
        for t in x['tracks']:
            if t['confirmed']:
                b = wc.body(np.array(t['centroid_m']), q, mh)
                cc, ss = math.cos(head), math.sin(head)
                pts.append(rob+np.array([cc*b[0]-ss*b[1], ss*b[0]+cc*b[1]])); ts.append(x['measured_ns']/1e9)
    pts = np.array(pts).reshape(-1, 2); ts = np.array(ts)
    # disk cells of every track position
    g = np.arange(-RADIUS_M, RADIUS_M+CELL, CELL); gx, gy = np.meshgrid(g, g); off = np.stack((gx.ravel(), gy.ravel()), 1); off = off[np.linalg.norm(off, axis=1) <= RADIUS_M]
    cells = np.unique(np.round((pts[:, None]+off[None]).reshape(-1, 2)/CELL).astype(int), axis=0) if len(pts) else np.zeros((0, 2), int)
    in_wall = wc.inside_walls(cells*CELL, walls) if len(cells) else np.zeros(0, bool)
    out = dict(name=run.name.split('validation_')[1], cells=len(cells), wall_cells=int(in_wall.sum()))
    hit = next((x for x in json.loads((run/'native/contact_events.json').read_text()) if x['disallowed_contacts']), None)
    if hit:
        d = hit['disallowed_contacts'][0]; t = hit['timestamp_s']
        oid = d['environment_object_id']
        w = [b for b in walls if b.get('name') == oid or b.get('id') == oid]
        cp = None
        for k in ('position_world_m', 'contact_position_world_m', 'position_m'):
            if k in d: cp = np.array(d[k][:2], float)
        out['object'] = oid
        if cp is not None and len(pts):
            early = ts <= t
            gaps = np.linalg.norm(pts[early]-cp, axis=1) if early.any() else np.array([np.inf])
            out['track_to_contact_m'] = round(float(gaps.min()), 2)
            out['seconds_before'] = round(float(t-ts[early][int(gaps.argmin())]), 1) if early.any() else None
    return out
if __name__ == '__main__':
    print('radius', RADIUS_M)
    for c in sys.argv[1:]:
        runs = sorted(glob.glob(f'outputs/cohorts/{c}/runs/*'))
        with ProcessPoolExecutor(10) as p: rs = [r for r in p.map(one, runs) if r]
        print(c, 'missions', len(rs), 'trail cells', sum(r['cells'] for r in rs), 'inside walls', sum(r['wall_cells'] for r in rs), 'missions with any', sum(r['wall_cells'] > 0 for r in rs), 'median per mission', np.median([r['wall_cells'] for r in rs]))
        for r in rs:
            if 'object' in r: print('  ', r['name'], r['object'][-12:], 'nearest earlier track to the contact', r.get('track_to_contact_m'), 'm,', r.get('seconds_before'), 's before; wall cells in trail', r['wall_cells'])
