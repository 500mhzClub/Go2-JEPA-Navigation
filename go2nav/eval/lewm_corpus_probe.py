"""How much body-frame motion LeWM latents hold, read with plenty of in-domain data: the screening measure for C5.

    python -m go2nav.eval.lewm_corpus_probe WORLD.pt --corpus ROOT --out FILE.json [--train-units 300]
                                            [--val-units 60] [--transfer-units 20] [--updates 20000]

T7's fit (`go2nav.training.lewm_decoder`) reads the decoder contexts, 43k decisions; this probe reads the corpus's own
windows, whose poses are labels: units drawn with a fixed seed from the training mazes (fit) and the validation mazes
and transfer units (evaluation). Per window, frames t-3..t and the futures t+500 and t+800 ms; targets the body-frame
motion (x, y, yaw) from the poses. Four heads, each the same MLP (two 1,024 layers; AdamW 3e-4, cosine, batch 512):
- `probe`: C5's decoder input, [z_t, z_(t+h) - z_t, z_t - z_(t-1), z_t - z_(t-3)], with the true future z;
- `c5`: the same with the world model's rollout under the executed actions;
- `cmds`: the applied-command history and the tape up to h (no vision);
- `c6`: the four z with the commands.
Reported per split and horizon: median XY error and RMS yaw, and on windows moving more than 50 mm the median XY error
and the median projected displacement ratio. On the 17 October Go2 chain (docs/lewm/STATUS.md) `c5` on validation
mazes at 800 ms read 13.6 mm (T7 score 21.5 mm); the screening bar is 9.5 mm.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from go2nav.models.lewm import action_blocks, control_stats, load_world_model, normalise_frames

HS = (5, 8)
HEADS = ('probe', 'c5', 'cmds', 'c6')


def yaw_of(q):  # (n, 4) x, y, z, w
    x, y, z, w = q.T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


@torch.no_grad()
def unit_data(model, root, units, index, u, dev, cmean, cstd):
    folder = root/units[u]['path']
    frames = np.load(folder/'frames.npy', mmap_mode='r')
    steps = np.load(folder/'steps.npz')
    zs = []
    for s in range(0, len(frames), 1024):
        x = torch.from_numpy(np.ascontiguousarray(frames[s:s+1024])).to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zs.append(model.encode(normalise_frames(x)[:, None])[:, 0].float())
    z = torch.cat(zs)
    starts = index['start'][index['unit'] == u]
    t = starts+3
    pose = steps['pose']
    yaw = yaw_of(pose[:, 3:7])
    targets = []
    for h in HS:
        d = pose[t+h, :2]-pose[t, :2]
        c, s_ = np.cos(yaw[t]), np.sin(yaw[t])
        dx, dy = c*d[:, 0]+s_*d[:, 1], -s_*d[:, 0]+c*d[:, 1]
        dyaw = np.angle(np.exp(1j*(yaw[t+h]-yaw[t])))
        targets.append(np.stack([dx, dy, dyaw], 1))
    blocks = action_blocks(steps['executed'], cmean, cstd).to(dev)          # (n, 10)
    hist = torch.from_numpy(((steps['applied_history'][:, :, [0, 2]]-cmean)/cstd).astype(np.float32)).to(dev)  # (n,15,2)
    tape = torch.from_numpy(((steps['executed'][:, 0, [0, 2]]-cmean)/cstd).astype(np.float32)).to(dev)       # (n,2) first cmd of tick
    tt = torch.from_numpy(t).to(dev)
    zh = torch.stack([z[tt+k] for k in (-3, -2, -1, 0)], 1)                  # (w, 4, D)
    zf = torch.stack([z[tt+h] for h in HS], 1)                                # (w, 2, D)
    acts = torch.stack([blocks[tt-3+k] for k in range(11)], 1)                # (w, 11, 10)
    rolled = []
    for s in range(0, len(tt), 2048):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rolled.append(model.rollout(zh[s:s+2048], acts[s:s+2048], 8).float())
    rolled = torch.cat(rolled)[:, [h-1 for h in HS]]                         # (w, 2, D)
    tapes = torch.stack([tape[tt+k] for k in range(8)], 1)                    # (w, 8, 2)
    return dict(zh=zh.cpu(), zf=zf.cpu(), zr=rolled.cpu(), past=hist[tt].cpu(), tape=tapes.cpu(),
                y=torch.from_numpy(np.stack(targets, 1).astype(np.float32)))   # y (w, 2, 3)


def inputs(d, kind, hi, dev):
    zh = d['zh'].to(dev)
    cur = zh[:, -1]
    h = HS[hi]
    if kind in ('probe', 'c5'):
        fut = (d['zf'] if kind == 'probe' else d['zr'])[:, hi].to(dev)
        return torch.cat((cur, fut-cur, cur-zh[:, -2], cur-zh[:, 0]), 1)
    steps = torch.arange(8, device=dev)[None]
    tape = d['tape'].to(dev)*(steps < h)[:, :, None]
    cmds = torch.cat((d['past'].to(dev).flatten(1), tape.flatten(1), torch.full((len(zh), 1), h/8, device=dev)), 1)
    return cmds if kind == 'cmds' else torch.cat((zh.flatten(1), cmds), 1)

def fit_eval(data, kind, dev, updates, batch=512):
    X = torch.cat([inputs(data['train'], kind, hi, dev) for hi in range(len(HS))])
    Y = torch.cat([data['train']['y'][:, hi] for hi in range(len(HS))]).to(dev)
    mu, sd = Y.mean(0), Y.std(0)
    xm, xs = X.mean(0), X.std(0)+1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates)
    g = torch.Generator(device=dev).manual_seed(1)
    for step in range(updates):
        i = torch.randint(len(X), (batch,), device=dev, generator=g)
        loss = (net((X[i]-xm)/xs)-(Y[i]-mu)/sd).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        sched.step()
    res = {}
    with torch.no_grad():
        for split in ('train', 'val', 'transfer'):
            for hi, h in enumerate(HS):
                x = inputs(data[split], kind, hi, dev)
                y = data[split]['y'][:, hi].to(dev)
                p = net((x-xm)/xs)*sd+mu
                xy = (p[:, :2]-y[:, :2]).norm(dim=1)*1000
                yaw = torch.rad2deg(p[:, 2]-y[:, 2])
                true = y[:, :2].norm(dim=1)*1000
                moving = true > 50
                res[f'{split}@{h*100}ms'] = dict(median_xy_mm=xy.median().item(), rmse_yaw_deg=yaw.pow(2).mean().sqrt().item(),
                                                 median_true_mm=true.median().item(),
                                                 moving_share=moving.float().mean().item(),
                                                 moving_median_xy_mm=xy[moving].median().item() if moving.any() else float('nan'),
                                                 moving_median_ratio=((p[moving, :2]*y[moving, :2]).sum(1)/y[moving, :2].pow(2).sum(1)).median().item() if moving.any() else float('nan'))
    return res


def run(world, corpus, *, train_units=300, val_units=60, transfer_units=20, updates=20000, seed=0, dev='cuda:0'):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(corpus)
    units = json.loads((root/'index_12.json').read_text())['units']
    index = np.load(root/'index_12.npz')

    def pick(role, kinds, n):
        ids = [i for i, u in enumerate(units) if u['role'] == role and u['kind'] in kinds]
        return sorted(rng.choice(ids, min(n, len(ids)), replace=False).tolist())
    chosen = dict(train=pick('lewm_train', ('explore', 'mission', 'free'), train_units),
                  val=pick('lewm_val', ('explore', 'mission', 'free'), val_units),
                  transfer=pick('lewm_val', ('transfer',), transfer_units))
    model = load_world_model(world, dev)
    cmean, cstd = control_stats()
    started, data = time.time(), {}
    for split, ids in chosen.items():
        parts = [unit_data(model, root, units, index, u, dev, cmean, cstd) for u in ids]
        data[split] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        print(split, len(ids), 'units', len(data[split]['y']), 'windows', f'{time.time()-started:.0f} s', flush=True)
    results = {}
    for kind in HEADS:
        results[kind] = fit_eval(data, kind, dev, updates)
        print(kind, json.dumps({k: round(v['median_xy_mm'], 2) for k, v in results[kind].items()}), flush=True)
    return dict(world_model=str(world), units={k: len(v) for k, v in chosen.items()}, updates=updates, results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('world', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--train-units', type=int, default=300)
    p.add_argument('--val-units', type=int, default=60)
    p.add_argument('--transfer-units', type=int, default=20)
    p.add_argument('--updates', type=int, default=20000)
    a = p.parse_args()
    result = run(a.world, a.corpus, train_units=a.train_units, val_units=a.val_units, transfer_units=a.transfer_units,
                 updates=a.updates)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1)+'\n')


if __name__ == '__main__':
    main()
