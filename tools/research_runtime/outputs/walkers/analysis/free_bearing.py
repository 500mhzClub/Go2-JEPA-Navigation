"""The free-space head's error by ray bearing, and against simple references, on encoded states, CPU."""
import json, sys
from pathlib import Path
import numpy as np, torch
from go2nav.data.lewm_dataset import Corpus
from go2nav.models.lewm import FREE_SCALE_M, load_world_model, normalise_frames
from go2nav.eval.mover_probe import HALF_FOV_RAD
torch.set_num_threads(10)
model = load_world_model('outputs/lewm/go2_v2_walkers/lewm.pt', torch.device('cpu')).eval()
c = Corpus(Path('/mnt/workspace_drive/go2nav_lewm_v2'), length=12, roles=('lewm_val',), kinds=('explore', 'mission', 'free'), command_dim=model.config.command_dim, movers=True, free=True)
rng = np.random.default_rng(2026110102); idx = rng.choice(len(c), 3000, replace=False)
P, T = [], []
for s in range(0, len(idx), 48):
    f, a, k, m, r = c.batch(idx[s:s+48])
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        z = model.encode(normalise_frames(f[:, :4]))
        P.append((model.free_rays(z[:, 3]).float()*FREE_SCALE_M).numpy()); T.append(r[:, 3].numpy())
p, t = np.concatenate(P), np.concatenate(T)
print('half field of view deg', np.degrees(HALF_FOV_RAD))
print('ray  deg  mean_truth  err  near_share near_err false_clear  mean_pred pred_std truth_std corr')
for i in range(16):
    n = t[:, i] < .6
    print(i, round(i*22.5), round(float(t[:, i].mean()), 2), round(float(np.median(abs(p[:, i]-t[:, i]))), 3), round(float(n.mean()), 2), round(float(np.median(abs(p[n, i]-t[n, i]))), 3), round(float(((p[n, i]-t[n, i]) > .15).mean()), 2),
          round(float(p[:, i].mean()), 2), round(float(p[:, i].std()), 2), round(float(t[:, i].std()), 2), round(float(np.corrcoef(p[:, i], t[:, i])[0, 1]), 2))
