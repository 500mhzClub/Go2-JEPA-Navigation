"""The rollout's 0.8-s walker forecast against the model tracker's own straight line and a held position, on moving walkers."""
import glob, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from go2nav.data.lewm_dataset import Corpus
from go2nav.models.lewm import MOVER_SCALE_M, load_world_model, normalise_frames
from go2nav.controllers import walkers as W
torch.set_num_threads(10)
root = Path('/mnt/workspace_drive/go2nav_lewm_v2')
model = load_world_model(sys.argv[1], torch.device('cpu')).eval()
c = Corpus(root, length=12, roles=('lewm_val',), kinds=('explore', 'mission'), command_dim=model.config.command_dim, movers=True)
units = sorted(int(Path(f).stem) for f in glob.glob(sys.argv[2]+'/*.npz'))
I = np.eye(3)
def Rz(a): s, c_ = np.sin(a), np.cos(a); return np.array([[c_, -s, 0], [s, c_, 0], [0, 0, 1.]])
rows = []
for u in units:
    z = np.load(f'{sys.argv[2]}/{u}.npz'); folder = root/c.units[u]['path']
    if not (folder/'dynamic.npz').exists(): continue
    with np.load(folder/'dynamic.npz') as o: xy = o['obstacles_xy'].astype(float)
    if not xy.size: continue
    det, labels, poses = z['detected'], z['labels'], z['poses'].astype(float)
    tr = W.ModelTracker()
    states = {}
    for f in range(len(det)):
        tr.update([det[f]], [int(f*1e8)], [(np.array([*poses[f, :2], 0.]), Rz(poses[f, 2]))], I)
        states[f] = [(t.history[-1][1].copy(), t.velocity.copy(), (f*1e8-t.seen_ns)/1e9) for t in tr.tracks if t.confirmed]
    w = np.flatnonzero(c.unit == u); w = w[c.start[w] % 4 == 0]
    if not len(w): continue
    for s in range(0, len(w), 48):
        ids = w[s:s+48]
        f_, a, k, m, *_ = c.batch(ids)
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            zz = model.encode(normalise_frames(f_[:, :4])); rolled = model.rollout(zz, a, 8)
            out = model.movers(rolled[:, -1]).float().numpy()
        for i, (_, st) in enumerate(k.numpy()):
            t, e = int(st)+3, int(st)+11
            if e >= len(poses): continue
            ce, se = np.cos(poses[e, 2]), np.sin(poses[e, 2])
            guess = out[i][out[i, :, 0] > 0, 1:]*MOVER_SCALE_M
            guess_w = poses[e, :2]+np.stack((ce*guess[:, 0]-se*guess[:, 1], se*guess[:, 0]+ce*guess[:, 1]), -1) if len(guess) else np.zeros((0, 2))
            vc, vs = np.cos(poses[t, 2]), np.sin(poses[t, 2])
            for b in labels[t][labels[t, :, 0] > 0, 1:]:
                here = poses[t, :2]+np.array([vc*b[0]-vs*b[1], vs*b[0]+vc*b[1]])
                ok = ~np.isnan(xy[t, :, 0])
                if not ok.any(): continue
                j = np.flatnonzero(ok)[np.linalg.norm(xy[t][ok]-here, axis=1).argmin()]
                if np.isnan(xy[e, j, 0]): continue
                speed = np.linalg.norm(xy[t, j]-xy[max(t-4, 0), j])/.4
                truth = xy[e, j]
                walked = float(np.linalg.norm(truth-xy[t, j]))
                in_end = bool(len(labels[e][labels[e, :, 0] > 0])) and np.linalg.norm(((np.array([[ce, se], [-se, ce]])@(truth-poses[e, :2]))-labels[e][labels[e, :, 0] > 0, 1:]), axis=1).min() < .3
                tracks = states.get(t, [])
                if not tracks: tracker, held = np.inf, np.inf
                else:
                    a_ = int(np.argmin([np.linalg.norm(p-xy[t, j]) for p, _, _ in tracks])); p, v, age = tracks[a_]
                    if np.linalg.norm(p-xy[t, j]) > .5: tracker, held = np.inf, np.inf
                    else: tracker, held = float(np.linalg.norm(p+v*(age+.8)-truth)), float(np.linalg.norm(p-truth))
                rollout = float(np.linalg.norm(guess_w-truth, axis=1).min()) if len(guess_w) else np.inf
                rows.append((speed, walked, tracker, held, rollout, in_end, u))
a = np.array(rows, float); np.save(sys.argv[3], a)
sp, wk, tk, hd, ro, ie, un = a.T
def line(n, m):
    both = m&np.isfinite(tk)&np.isfinite(ro)&(tk <= .75)&(ro <= .75)
    rng = np.random.default_rng(0); us = np.unique(un[both])
    d = []
    for _ in range(1000):
        pick = rng.choice(us, len(us)); mask = np.concatenate([np.flatnonzero((un == x)&both) for x in pick])
        d.append(np.median(ro[mask])-np.median(tk[mask]))
    print(f'{n:34s} n {m.sum():6d} scored {both.sum():6d} units {len(us)} walked {np.median(wk[m]):.3f} | rollout {np.median(ro[both]):.3f} tracker {np.median(tk[both]):.3f} held {np.median(hd[both]):.3f} | rollout-tracker 95% [{np.quantile(d,.025):+.3f}, {np.quantile(d,.975):+.3f}] | rollout within .5 {np.mean(ro[m]<=.5):.3f} tracker within .5 {np.mean(tk[m]<=.5):.3f}')
line('all walkers', sp >= 0); line('moving (>.05 m/s)', sp > .05); line('walking (>.15 m/s)', sp > .15); line('still', sp <= .05); line('moving, visible at the end', (sp > .05)&(ie > 0)); line('moving, out of view at the end', (sp > .05)&(ie == 0))
