"""Tracker settings on the head's 10-Hz detections: true poses, true walkers; model tracks scored as walker_compare scores them."""
import glob, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path('/mnt/workspace_drive/go2nav_lewm_v2')
PRES, CONF = .5, 3

class T:
    def __init__(s, i, c, now): s.id, s.h, s.seen, s.n, s.v = i, [(now, c)], now, 1, np.zeros(2)

def merge(points, r):
    points = [np.asarray(p) for p in points]
    if r <= 0 or len(points) < 2: return points
    out, used = [], set()
    for i, p in enumerate(points):
        if i in used: continue
        group = [p]
        for j in range(i+1, len(points)):
            if j not in used and np.linalg.norm(points[j]-p) <= r:
                group.append(points[j]); used.add(j)
        out.append(np.mean(group, 0))
    return out

def run(tracks_cfg, data):
    gate, predict, merge_r, span, window, keep = tracks_cfg
    d, poses, xy = data
    tracks, nid, rows, born = [], 0, [], 0
    for f in range(len(d)):
        now = f*.1
        c, s = np.cos(poses[f, 2]), np.sin(poses[f, 2])
        pts = [poses[f, :2]+np.array([c*x-s*y, s*x+c*y]) for p, x, y in d[f] if p > PRES]
        pts = merge(pts, merge_r)
        tracks = [t for t in tracks if now-t.seen <= keep]
        def at(t): return t.h[-1][1]+(t.v*(now-t.seen) if predict else 0)
        pairs = sorted((float(np.linalg.norm(p-at(t))), k, t.id) for k, p in enumerate(pts) for t in tracks
                       if np.linalg.norm(p-at(t)) <= gate+(0 if predict else .6)*(now-t.seen))
        used, taken, by = set(), set(), {t.id: t for t in tracks}
        new = []
        for _, k, i in pairs:
            if k not in used and i not in taken:
                used.add(k); taken.add(i); new.append((by[i], pts[k]))
        for k, p in enumerate(pts):
            if k not in used:
                t = T(nid, p, now); nid += 1; tracks.append(t); born += 1
        for t, p in new:
            t.seen, t.n = now, t.n+1
            t.h = [h for h in t.h if now-h[0] <= window]+[(now, p)]
            ts = np.array([h[0]-now for h in t.h])
            if len(ts) < 3 or ts[-1]-ts[0] < span: t.v = np.zeros(2); continue
            q = np.array([h[1] for h in t.h]); dt = ts-ts.mean()
            v = (dt[:, None]*(q-q.mean(0))).sum(0)/(dt*dt).sum(); sp = np.linalg.norm(v)
            t.v = v*min(1, .6/sp) if sp > 0 else v
        if f < 8 or f+1 >= len(d): continue
        truth = xy[f][~np.isnan(xy[f, :, 0])] if xy is not None else np.zeros((0, 2))
        vel = (xy[f]-xy[f-4])/.4 if xy is not None else None
        conf = [t for t in tracks if t.n >= CONF]
        if conf:
            q = np.array([t.h[-1][1] for t in conf])
            near = np.linalg.norm(q-poses[f, :2], axis=1) <= 2
            for a in q[near]:
                rows.append((0, np.linalg.norm(truth-a, axis=1).min() > .5 if len(truth) else True, 0, 0, 0))
        # walkers in view: the frame's labels, as world points
        rows_view = []
        return_view = d[f]  # placeholder for clarity
    return rows, born

def score(cfg, args):
    path = args
    z = np.load(path)
    unit = json.loads(str(z['unit']))
    folder = ROOT/unit['path']
    xy = None
    if (folder/'dynamic.npz').exists():
        with np.load(folder/'dynamic.npz') as o: xy = o['obstacles_xy'].astype(float)
    if xy is None or not xy.size: return None
    d, labels, poses = z['detected'], z['labels'], z['poses'].astype(float)
    return scored(cfg, d, labels, poses, xy)

def scored(cfg, d, labels, poses, xy):
    gate, predict, merge_r, span, window, keep = cfg
    tracks, nid = [], 0
    reach = found = false = plans = born = 0
    verr, vstill, called, zero = [], [], [], 0
    for f in range(len(d)):
        now = f*.1
        c, s = np.cos(poses[f, 2]), np.sin(poses[f, 2])
        pts = merge([poses[f, :2]+np.array([c*x-s*y, s*x+c*y]) for p, x, y in d[f] if p > PRES], merge_r)
        tracks = [t for t in tracks if now-t.seen <= keep]
        at = lambda t: t.h[-1][1]+(t.v*(now-t.seen) if predict else 0)
        pairs = sorted((float(np.linalg.norm(p-at(t))), k, t.id) for k, p in enumerate(pts) for t in tracks
                       if np.linalg.norm(p-at(t)) <= gate+(0 if predict else .6)*(now-t.seen))
        used, taken, by, new = set(), set(), {t.id: t for t in tracks}, []
        for _, k, i in pairs:
            if k not in used and i not in taken:
                used.add(k); taken.add(i); new.append((by[i], pts[k]))
        for k, p in enumerate(pts):
            if k not in used:
                tracks.append(T(nid, p, now)); nid += 1; born += 1
        for t, p in new:
            t.seen, t.n = now, t.n+1
            t.h = [h for h in t.h if now-h[0] <= window]+[(now, p)]
            ts = np.array([h[0]-now for h in t.h])
            if len(ts) < 3 or ts[-1]-ts[0] < span: t.v = np.zeros(2); continue
            q = np.array([h[1] for h in t.h]); dt = ts-ts.mean()
            v = (dt[:, None]*(q-q.mean(0))).sum(0)/(dt*dt).sum(); sp = np.linalg.norm(v)
            t.v = v*min(1, .6/sp) if sp > 0 else v
        if f < 4: continue
        plans += 1
        conf = [t for t in tracks if t.n >= CONF]
        q = np.array([t.h[-1][1] for t in conf]).reshape(-1, 2)
        ok = ~np.isnan(xy[f, :, 0]); truth, vel = xy[f][ok], ((xy[f]-xy[f-4])/.4)[ok]
        near = q[np.linalg.norm(q-poses[f, :2], axis=1) <= 2]
        for a in near:
            if not len(truth) or np.linalg.norm(truth-a, axis=1).min() > .5: false += 1
        view = labels[f][labels[f, :, 0] > 0, 1:]
        cc, ss = np.cos(poses[f, 2]), np.sin(poses[f, 2])
        for b in view:
            if np.hypot(*b) > 2: continue
            w = poses[f, :2]+np.array([cc*b[0]-ss*b[1], ss*b[0]+cc*b[1]])
            reach += 1
            if not len(truth): continue
            j = int(np.linalg.norm(truth-w, axis=1).argmin())
            if not len(q): continue
            g = np.linalg.norm(q-truth[j], axis=1); a = int(g.argmin())
            if g[a] > .5: continue
            found += 1
            sp = np.linalg.norm(vel[j])
            if sp > .05:
                verr.append(float(np.linalg.norm(conf[a].v-vel[j]))); called.append(np.linalg.norm(conf[a].v) >= .05); zero += float(np.all(conf[a].v == 0))
            else:
                vstill.append(float(np.linalg.norm(conf[a].v)))
    return dict(reach=reach, found=found, false=false, plans=plans, born=born, verr=verr, vstill=vstill, called=called, zero=zero)

def one(a): return score(*a)

if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1]+'/*.npz'))
    cfgs = {'current': (.15, False, 0, .7, 1.2, 1.2), 'A': (.25, False, .3, .5, 1.6, 1.2)}
    with ProcessPoolExecutor(10) as pool:
        for name, cfg in cfgs.items():
            rs = [r for r in pool.map(one, [(cfg, f) for f in files], chunksize=4) if r]
            sum_ = lambda k: sum(r[k] for r in rs)
            v, st, ca = (np.concatenate([np.array(r[k], float) for r in rs]) for k in ('verr', 'vstill', 'called'))
            print(f"{name:44s} found {sum_('found')/max(sum_('reach'),1):.3f} false/100 {100*sum_('false')/sum_('plans'):.2f} tracks/unit {sum_('born')/len(rs):.1f} "
                  f"vel err {np.median(v):.3f} zero {sum_('zero')/len(v):.3f} called moving {ca.mean():.3f} still speed {np.median(st) if len(st) else float('nan'):.3f} n {len(rs)}/{sum_('reach')}", flush=True)
