"""Is gate 1 achievable? Fit the gate's label (the walker responds to a 0.1 m/s retreat against holding by > 5 cm at
t+h) from the simulator's own state at t, not from a latent. Train on the training units' give-way windows, score AUC
on val and transfer, as the gate does."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b')
from go2nav.eval import mover_probe as mp

C = '/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/phase_c'
torch.manual_seed(0)
torch.set_num_threads(24)


def load(name):
    d = dict(np.load(f'{C}/give_way_labels_{name}.npz'))
    rows = np.arange(len(d['t']))
    keep = mp.give_way_windows(d, rows) & ~np.isnan(d['cf_hold'][:, 0, 0])
    xy = d['xy_t']
    vel = (d['constant_velocity'][:, 0]-xy)/.4
    base = np.column_stack([xy, np.linalg.norm(xy, axis=-1), np.arctan2(xy[:, 1], xy[:, 0]), vel,
                            np.linalg.norm(vel, axis=-1), d['waiting'], d['moving'], d['previous']])
    feats = dict(observable=base, with_type=np.column_stack([base, d['polite']]))
    split = mp.kinds(d) if name == 'val' else np.full(len(rows), 'train')
    labels = {j: np.linalg.norm(d['cf_retreat'][:, j]-d['cf_hold'][:, j], axis=-1) > mp.EVENT_M for j in (2, 3)}
    return {k: v[keep] for k, v in feats.items()}, {j: l[keep] for j, l in labels.items()}, split[keep], d['polite'][keep]


def fit(x, y, mlp):
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    mean, std = x.mean(0), x.std(0)+1e-6
    net = (torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                               torch.nn.GELU(), torch.nn.Linear(128, 1)) if mlp else torch.nn.Linear(x.shape[1], 1))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    pos = y.mean().clamp(1e-3, 1-1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1-pos)/pos)
    for step in range(3000 if mlp else 1500):
        i = torch.randint(len(x), (4096,))
        loss = loss_fn(net((x[i]-mean)/std).squeeze(-1), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda z: net((torch.tensor(z, dtype=torch.float32)-mean)/std).squeeze(-1).detach().numpy()


train_x, train_y, _, _ = load('train')
eval_x, eval_y, split, polite = load('val')
out = dict(windows=dict(train=int(len(train_y[2])), **{s: int((split == s).sum()) for s in ('val', 'transfer')}),
           positive_share={f'{h}s': dict(train=round(float(train_y[j].mean()), 3),
                                         **{s: round(float(eval_y[j][split == s].mean()), 3) for s in ('val', 'transfer')})
                           for j, h in ((2, 1.6), (3, 3.2))},
           auc={})
for feats in ('observable', 'with_type'):
    for mlp in (False, True):
        key = f'{feats}_{"mlp" if mlp else "logistic"}'
        out['auc'][key] = {}
        for j, h in ((2, 1.6), (3, 3.2)):
            f = fit(train_x[feats], train_y[j], mlp)
            score = f(eval_x[feats])
            out['auc'][key][f'{h}s'] = {s: round(mp.auc(score[split == s], eval_y[j][split == s]), 3)
                                        for s in ('val', 'transfer')}
        print(key, out['auc'][key], flush=True)
print(json.dumps(out, indent=1))
json.dump(out, open(f'{C}/give_way_ceiling.json', 'w'), indent=1)
