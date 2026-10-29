"""How well can anything predict a mover's response to the robot's give-way choice? (docs/dynamic/PLAN.md, P4e)

    python -m go2nav.eval.give_way_ceiling ROOT --eval LABELS_VAL.npz --out DIR [--workers N]

The give-way gate (`mover_probe.give_way`) asks a world model which movers near and ahead respond when the robot backs
off rather than holds. Walkers look alike (one cylinder, a colour per index): whether one responds depends on state
the camera shows only through its motion (stopped for the robot or pausing at a cell, its heading while stopped, its
politeness, which is per unit), so the gate's attainable score depends on how much of a walker's past a predictor
sees. **Inputs** (privileged: the simulator's records, never a model's), on the probe's give-way windows with a
counterfactual, in the robot's body frame at t:
- `track_0.3s`: the mover's and the robot's positions at frames t-3..t (a world model's history), the command
  executed before t, and the wall clearance at, and 0.3 m behind, the robot and at the mover;
- `track_3s`: the same over t-30..t (frames before the unit's start repeat its first, with the share that exists);
- `oracle`: `track_3s` plus the walker's internal state at t from the replay (`mover_probe.replay`): polite, stopped
  for the robot and for how long, pausing and for how long, its heading, its next cell, its speed.
**Fits:** per input set and horizon (1.6 s and 3.2 s), an MLP classifier of whether the retreat (and advance)
contrast exceeds `mover_probe.EVENT_M`, and an MLP regressor of the contrast vector; `val` scored by 5-fold
cross-validation over units, `transfer` by the fit on every `val` window. **Scores:** as `mover_probe.give_way`:
the AUC with a 95% interval over units, and the contrast's R2. Also the share that responds by condition at t.
Writes DIR/results.json.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.data import lewm_corpus as v1
from go2nav.data import lewm_corpus_v2 as v2
from go2nav.eval import mover_probe as mp
from go2nav.mazes.grid import PITCH_M

SETS = ('track_0.3s', 'track_3s', 'oracle')
SHORT, LONG = 4, 31                            # frames of track: t-3..t, t-30..t
BEHIND_M = .3
HORIZON_INDICES = (2, 3)                       # 1.6 s, 3.2 s
FOLDS, SEED = 5, 2026102901
EPOCHS, HIDDEN, DROPOUT, WEIGHT_DECAY, LR = 300, 128, .2, 1e-2, 3e-3


def track(pose, recorded, t, k, frames):
    """(frames*4 + 1,) the mover's and the robot's positions at t-frames+1..t in the body frame at t, and the share of
    those frames inside the unit."""
    index = np.arange(t-frames+1, t+1)
    inside = float((index >= 0).mean())
    index = np.maximum(index, 0)
    return np.concatenate([mp.body_frame(pose[t], recorded[index, k]).ravel(),
                           mp.body_frame(pose[t], pose[index, :2]).ravel(), [inside]])


def unit_inputs(args):
    """The input sets (see the module docstring) of one unit's windows `ts`, `ks`; {set: (n, d)}."""
    root, folder, ts, ks = Path(args[0]), Path(args[1]), args[2], args[3]
    meta = json.loads((folder/'meta.json').read_text())
    placement = meta['obstacles']['placement']
    with np.load(folder/'steps.npz') as steps:
        pose, executed = steps['pose'], steps['executed']
    with np.load(folder/'dynamic.npz') as dynamic:
        recorded = dynamic['obstacles_xy'].astype(float)
    spec = mp.unit_spec(root, meta)
    clearance = v2.FreeDriverV2(spec, np.random.default_rng(0)).wall_clearance
    _, states = mp.replay(spec, placement, pose, recorded, keep=set(int(t) for t in ts))
    out = {s: [] for s in SETS}
    for t, k in zip(ts, ks):
        p = pose[t]
        yaw = float(v1.planar_yaw(np.asarray(p)[3:7]))
        back = p[:2]-BEHIND_M*np.array([np.cos(yaw), np.sin(yaw)])
        common = np.concatenate([executed[t-1, -1], [clearance(p[:2]), clearance(back), clearance(recorded[t, k])]])
        short = np.concatenate([track(pose, recorded, t, k, SHORT), common])
        long = np.concatenate([track(pose, recorded, t, k, LONG), common])
        w = states[t][k]
        rotate = lambda v: mp.body_frame(p, np.asarray(p[:2])+v)
        state = np.concatenate([[float(w.polite), float(w.blocked_for > 0), w.blocked_for, float(w.pause_left > 0),
                                 w.pause_left, w.speed], rotate(w.heading()),
                                mp.body_frame(p, np.asarray(w.target, float)*PITCH_M)])
        out['track_0.3s'].append(short)
        out['track_3s'].append(long)
        out['oracle'].append(np.concatenate([long, state]))
    return {s: np.asarray(v, np.float32) for s, v in out.items()}


def inputs(root, data, rows, workers=None):
    """{set: (len(rows), d)} for the windows `rows` of the labels `data`, one job per unit."""
    units = data['unit'][rows]
    jobs, order = [], []
    for u in np.unique(units):
        mine = rows[units == u]
        jobs.append((str(root), str(Path(root)/str(data['units'][u])), data['t'][mine], data['mover'][mine]))
        order.append(np.flatnonzero(units == u))
    out = {s: np.zeros((len(rows), 0), np.float32) for s in SETS}
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2)//4),
                             mp_context=multiprocessing.get_context('forkserver')) as pool:
        for where, part in zip(order, pool.map(unit_inputs, jobs, chunksize=1)):
            for s in SETS:
                if out[s].shape[1] == 0:
                    out[s] = np.zeros((len(rows), part[s].shape[1]), np.float32)
                out[s][where] = part[s]
    return out


def fit(x, y, groups, *, classify, seed=SEED):
    """An MLP on standardised x -> y (n,) bool (classify) or (n, 2), with the epoch of the lowest loss on a tenth of
    the units held out; returns a predictor of (m, d) -> (m,) scores or (m, 2)."""
    import torch
    rng = np.random.default_rng(seed)
    held = rng.permutation(np.unique(groups))[:max(1, len(np.unique(groups))//10)]
    select, rfit = np.isin(groups, held), ~np.isin(groups, held)
    mean, scale = x[rfit].mean(0), x[rfit].std(0)+1e-6
    xt = torch.from_numpy((x-mean)/scale).float()
    yt = torch.from_numpy(np.asarray(y, np.float32)).reshape(len(y), -1)
    ymean, yscale = (yt[rfit].mean(0), yt[rfit].std(0).clamp_min(1e-6)) if not classify else (0., 1.)
    yn = (yt-ymean)/yscale
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                              torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                              torch.nn.Linear(HIDDEN, yt.shape[1]))
    loss_of = torch.nn.functional.binary_cross_entropy_with_logits if classify else torch.nn.functional.mse_loss
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rfit, select = torch.from_numpy(np.flatnonzero(rfit)), torch.from_numpy(np.flatnonzero(select))
    best = (np.inf, None)
    for _ in range(EPOCHS):
        net.train()
        for batch in rfit[torch.randperm(len(rfit))].split(256):
            loss = loss_of(net(xt[batch]), yn[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            held_loss = float(loss_of(net(xt[select]), yn[select]))
        if held_loss < best[0]:
            best = (held_loss, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    net.eval()

    def predict(x):
        with torch.no_grad():
            out = net(torch.from_numpy((x-mean)/scale).float())*yscale+ymean
        return out[:, 0].numpy() if classify else out.numpy()
    return predict


def scores(score, label, contrast, true, groups):
    out = dict(windows=int(len(label)), responsive=int(label.sum()))
    out['r2'] = round(1-float(((contrast-true)**2).sum())/max(float((true**2).sum()), 1e-12), 3)
    if label.sum() >= 10 and (~label).sum() >= 10:
        out['auc'] = round(mp.auc(score, label), 3)
        out['auc_95'] = mp.bootstrap(groups, lambda i: mp.auc(score[i], label[i]))
    return out


def conditions(data, rows, label):
    """The share of windows that respond, by the mover's condition at t."""
    polite, waiting, moving = (data[k][rows] for k in ('polite', 'waiting', 'moving'))
    groups = dict(polite_waiting=polite & waiting, polite_moving=polite & moving & ~waiting,
                  polite_still=polite & ~moving & ~waiting, impolite_waiting=~polite & waiting,
                  impolite_moving=~polite & moving & ~waiting, impolite_still=~polite & ~moving & ~waiting)
    return {name: dict(windows=int(m.sum()), responds=round(float(label[m].mean()), 3) if m.any() else None)
            for name, m in groups.items()}


def ceiling(root, evaluation, out, *, workers=None, threads=4):
    import torch
    torch.set_num_threads(threads)
    data = dict(np.load(evaluation))
    rows = np.flatnonzero(mp.give_way_windows(data, np.arange(len(data['t']))) & ~np.isnan(data['cf_hold'][:, 0, 0]))
    split = mp.kinds(data)[rows]
    x = inputs(root, data, rows, workers)
    groups = data['unit'][rows]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = dict(labels=str(evaluation), windows={s: int((split == s).sum()) for s in ('val', 'transfer')},
                   dims={s: int(v.shape[1]) for s, v in x.items()}, epochs=EPOCHS, folds=FOLDS, conditions={},
                   heads={})
    val, transfer = np.flatnonzero(split == 'val'), np.flatnonzero(split == 'transfer')
    folds = np.random.default_rng(SEED).permutation(np.unique(groups[val]))
    folds = {u: i % FOLDS for i, u in enumerate(folds)}
    fold = np.asarray([folds[u] for u in groups[val]])
    for c in mp.GIVE_WAY_CONTRASTS:
        for j in HORIZON_INDICES:
            true = (data[f'cf_{c}'][rows, j]-data['cf_hold'][rows, j]).astype(np.float32)
            label = np.linalg.norm(true, axis=-1) > mp.EVENT_M
            horizon = f'{mp.HORIZONS[j]/10:g}s'
            results['conditions'][f'{c}_{horizon}'] = {s: conditions(data, rows[m], label[m])
                                                       for s, m in (('val', split == 'val'),
                                                                    ('transfer', split == 'transfer'))}
            for s in SETS:
                score, contrast = np.zeros(len(rows)), np.zeros((len(rows), 2), np.float32)
                for f in range(FOLDS):
                    test, train = val[fold == f], val[fold != f]
                    score[test] = fit(x[s][train], label[train], groups[train], classify=True)(x[s][test])
                    contrast[test] = fit(x[s][train], true[train], groups[train], classify=False)(x[s][test])
                score[transfer] = fit(x[s][val], label[val], groups[val], classify=True)(x[s][transfer])
                contrast[transfer] = fit(x[s][val], true[val], groups[val], classify=False)(x[s][transfer])
                entry = {name: scores(score[m], label[m], contrast[m], true[m], groups[m])
                         for name, m in (('val', val), ('transfer', transfer))}
                results['heads'].setdefault(s, {}).setdefault(c, {})[horizon] = entry
                (out/'results.json').write_text(json.dumps(results, indent=1)+'\n')
                print(json.dumps(dict(contrast=c, horizon=horizon, inputs=s,
                                      val=entry['val'].get('auc'), transfer=entry['transfer'].get('auc'),
                                      r2=[entry['val']['r2'], entry['transfer']['r2']])), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root')
    parser.add_argument('--eval', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    ceiling(args.root, args.eval, args.out, workers=args.workers, threads=args.threads)


if __name__ == '__main__':
    main()
