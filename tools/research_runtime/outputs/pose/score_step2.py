"""P10 step 2 bar on windows shaped like step 1's outages: for each closed outage (frames d, path p), the error after
min(d, 100) carried steps on every window whose true path there lies in p's bin (PATH_EDGES_M), each outage weighted
equally. Bar: the model's 90th percentile at most 5 cm and its median no worse than C1's."""
import glob, json, numpy as np
from go2nav.eval.pose_carry import PATH_EDGES_M
t = np.load('outputs/pose/carriers/trajectories.npz')
path = t['path']
O = [o for f in glob.glob('outputs/cohorts/wm_pose_rude_obst*_c5/runs/*/pose_shadow.json') for o in json.load(open(f))['outages']]
closed = [o for o in O if o['closed']]
pooled = {k: [] for k in ('model', 'c1', 'command')}
used = 0
for o in closed:
    d = min(max(int(round(o['duration_s']*10)), 1), path.shape[1])
    b = np.clip(np.searchsorted(PATH_EDGES_M, o['path_m'], side='right')-1, 0, len(PATH_EDGES_M)-2)
    at = path[:, d-1]
    ok = (at >= PATH_EDGES_M[b]) & (at < PATH_EDGES_M[b+1])
    if ok.sum() < 20:
        continue
    used += 1
    w = np.full(ok.sum(), 1./ok.sum())
    for k in pooled:
        pooled[k].append((t[k][ok, d-1], w))
def wq(pairs, q):
    v = np.concatenate([p[0] for p in pairs]); w = np.concatenate([p[1] for p in pairs])
    i = np.argsort(v); c = np.cumsum(w[i])/w.sum()
    return float(v[i][np.searchsorted(c, q)])
res = {k: dict(median_m=round(wq(v, .5), 4), p90_m=round(wq(v, .9), 4)) for k, v in pooled.items()}
dur = np.array([o['duration_s'] for o in O]); pth = np.array([o['path_m'] for o in O])
print(json.dumps(dict(outages=len(O), closed=len(closed), scored=used, duration_s=dict(median=float(np.median(dur)), p90=float(np.percentile(dur, 90)), max=float(dur.max())),
                      path_m=dict(median=float(np.median(pth)), p90=float(np.percentile(pth, 90)), max=float(pth.max()), over_5cm=int((pth > .05).sum())),
                      carriers=res,
                      model_passes=res['model']['p90_m'] <= .05 and res['model']['median_m'] <= res['c1']['median_m'],
                      c1_passes=res['c1']['p90_m'] <= .05), indent=1))
