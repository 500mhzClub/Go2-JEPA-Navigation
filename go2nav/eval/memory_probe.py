"""Does a world model already remember the walkers it no longer sees? (docs/dynamic/PLAN.md, P9, step 2)

    python -m go2nav.eval.memory_probe ROOT --hjepa outputs/hjepa/go2_3level_v2/hjepa.pt --out outputs/memory/probe.json

H-JEPA v2's upper levels encode each level-1 latent on its own (`HJEPA.states`), so a state at t holds only frame t;
what a level knows of the past is the snapshots its predictor attends over (level 2: t-32, t-16, t, 1.6 s apart;
level 3: t-128, t). The probe reads those snapshots, with the robot's pose at each relative to t (as visual
odometry gives it), into the body-frame walker grid of `data.memory_labels`, against what level 1 sees now:
- `now`: level-1 z at t and t-3 (the 0.3 s the deployed model sees);
- `l1_snapshots`: level-1 z at t, t-16, t-32, t-128 and the poses;
- `l2`: level-2 states at t, t-16, t-32 and the poses;
- `l3`: level-3 states at t, t-128, the level-2 state at t, and the poses.
Each is an MLP over the 256 cells (`fit`), trained on the training units' frames and scored on the validation units'
as `eval.memory_baselines` scores the baselines (average precision on the cells out of view, the rear zone, walkers
that do not yield, and on the cells in view), with the baselines on the same frames. A frame needs 128 frames of
history (12.8 s), so the first 12.8 s of each unit are not scored.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from go2nav.data import lewm_corpus as v1
from go2nav.data import memory_labels as ml
from go2nav.eval import memory_baselines as mb

OFFSETS = (16, 32, 128)
EVERY = 2                          # every second labelled frame (0.4 s)
SETS = ('now', 'l1_snapshots', 'l2', 'l3')


def relative_poses(pose, t):
    """(len(t), 4 x len(OFFSETS)): the pose at t-o in the body frame at t (x, y, sin, cos of the yaw change)."""
    out = []
    for o in OFFSETS:
        rows = []
        for s in t:
            yaw, yaw0 = float(v1.planar_yaw(pose[s][3:7])), float(v1.planar_yaw(pose[s-o][3:7]))
            c, si = math.cos(yaw), math.sin(yaw)
            d = pose[s-o][:2]-pose[s][:2]
            rows.append((c*d[0]+si*d[1], -si*d[0]+c*d[1], math.sin(yaw0-yaw), math.cos(yaw0-yaw)))
        out.append(np.asarray(rows))
    return np.concatenate(out, 1).astype(np.float32)


def unit_features(model, folder, prior, device):
    """Features of each set, labels and baseline scores at a unit's probed frames (None if it has none)."""
    import torch
    from go2nav.models.lewm import normalise_frames
    grids = mb.unit_grids(folder, prior)
    rows = np.flatnonzero(grids['t'] >= max(OFFSETS))[::EVERY]
    if not len(rows):
        return None
    t = grids['t'][rows]
    frames = np.load(folder/'frames.npy', mmap_mode='r')
    needed = np.unique(np.concatenate([t, t-3, *(t-o for o in OFFSETS)]))
    z = {}
    with torch.inference_mode():
        for s in range(0, len(needed), 512):
            chunk = needed[s:s+512]
            x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                encoded = model.level1.encode(normalise_frames(x)[:, None])[:, 0].float()
            s2, s3 = model.states(encoded[None], 2)[0], model.states(encoded[None], 3)[0]
            for k, f in enumerate(chunk):
                z[int(f)] = (encoded[k].cpu().numpy(), s2[k].float().cpu().numpy(), s3[k].float().cpu().numpy())
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose']
    rel = relative_poses(pose, t)
    get = lambda level, offsets: np.concatenate([np.stack([z[int(s-o)][level] for s in t]) for o in offsets], 1)
    features = dict(now=get(0, (0, 3)), l1_snapshots=np.concatenate([get(0, (0, *OFFSETS)), rel], 1),
                    l2=np.concatenate([get(1, (0, 16, 32)), rel], 1), l3=np.concatenate([get(2, (0, 128)), get(1, (0,)), rel], 1))
    return dict(features=features, grids=grids, rows=rows)


def fit(x, y, device, epochs=30, seed=0):
    """An MLP (two hidden layers of 512) from standardised features to the 256 cells' logits."""
    import torch
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0)+1e-6
    X = torch.from_numpy((x-mean)/std).to(device)
    Y = torch.from_numpy(y.astype(np.float32)).to(device)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(.1),
                              torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    steps = epochs*math.ceil(len(X)/1024)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=device).split(1024):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def predict(x):
        with torch.inference_mode():
            return torch.sigmoid(net(torch.from_numpy((x-mean)/std).to(device))).cpu().numpy()
    return predict


def main():
    import torch
    from go2nav.models.hjepa import load_hjepa
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('root', type=Path)
    p.add_argument('--hjepa', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-units', type=int, default=1600)
    p.add_argument('--val-units', type=int, help='the first N validation units only (a smoke test)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    model = load_hjepa(a.hjepa, a.device)
    prior = mb.prior_rates(a.root, None)
    rng = np.random.default_rng(0)
    train = sorted(f for f in (a.root/'units'/'lewm_train').iterdir() if (f/ml.FILE).exists())
    train = [train[i] for i in sorted(rng.choice(len(train), min(a.train_units, len(train)), replace=False))]
    val = sorted(f for f in (a.root/'units'/'lewm_val').iterdir() if (f/ml.FILE).exists())[:a.val_units]
    parts = {}
    for split, folders in (('train', train), ('val', val)):
        parts[split] = [u for u in (unit_features(model, f, prior, a.device) for f in folders) if u is not None]
        print(split, len(parts[split]), 'units', flush=True)
    y = np.concatenate([u['grids']['occupied'][u['rows']] for u in parts['train']])
    result = dict(train_units=len(parts['train']), val_units=len(parts['val']), frames=dict(train=len(y)))
    predicted = {}
    for name in SETS:
        x = np.concatenate([u['features'][name] for u in parts['train']])
        predict = fit(x, y, a.device)
        predicted[name] = [predict(u['features'][name]) for u in parts['val']]
        print(name, 'fitted', flush=True)
    for view, out in (('out_of_view', True), ('in_view', False)):
        data = [mb.cells(u['grids'], u['rows'], out) for u in parts['val']]
        for name in SETS:
            mask = [~u['grids']['visible'][u['rows']] if out else u['grids']['visible'][u['rows']] for u in parts['val']]
            for d, m, s in zip(data, mask, predicted[name]):
                d[name] = s[m]
        merged = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
        zones = dict(all=np.ones(len(merged['label']), bool), rear=merged['rear'], do_not_yield=merged['rude'],
                     rear_do_not_yield=merged['rear'] & merged['rude'])
        result[view] = {zone: {name: round(mb.average_precision(merged[name][m], merged['label'][m]), 4)
                               for name in (*SETS, *mb.BASELINES)} | dict(occupied=int(merged['label'][m].sum()))
                        for zone, m in zones.items()}
        age = merged['age']
        result[view]['ap_by_age'] = {label: {name: round(mb.average_precision(
            merged[name][~merged['label'] | sel], merged['label'][~merged['label'] | sel]), 4) for name in (*SETS, 'constant_velocity')}
            for label, lo, hi in mb.AGE_BINS
            for sel in [np.isinf(age) if label == 'never seen' else (age > lo) & (age <= hi)]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
