"""What the robot can know about the walkers it does not see, without learning (docs/dynamic/PLAN.md, P9, step 1).

    python -m go2nav.eval.memory_baselines ROOT --out outputs/memory/baselines.json [--workers N]

On corpus v2's validation units, from `data.memory_labels`: each baseline scores every cell of the body-frame grid
at each labelled frame, from what the robot saw before (a walker's true position at the frames it was in view, and the
robot's true pose, so these are upper bounds on what a tracker of each kind could do):
- `forget`: the hand tracker's rule (`perception.tracks.KEEP_UNSEEN_S`): a walker unseen for up to 1.2 s is where its
  last seen velocity takes it, then forgotten;
- `hold`: a walker stays where it was last seen, until the cameras see that place empty;
- `constant_velocity`: it walks on at its last seen velocity (over the last 0.4 s of sightings, at most
  `MAX_SPEED_M_S`), until the cameras see the place it should be empty;
- `prior`: how often each cell is occupied while out of view on the training units, the same at every frame (no memory).
`hold` and `constant_velocity` rank cells by 1/(1 + seconds since the sighting), `forget` by 1.

**Scores**, on the cells out of the camera's view: average precision (`ap`), and for each baseline its precision and
recall at any positive score. Recall is also given by the age of the sighting of the walker in the cell (never seen, up
to 1.2 s, 1.2-5 s, 5-20 s, older), and every score for the rear zone (`REAR`: behind the body origin, within 0.8 m of
it, where reversing and turning in place reach) and for units with walkers that do not yield.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.data import memory_labels as ml
from go2nav.eval import mover_probe as mp
from go2nav.perception.tracks import KEEP_UNSEEN_S, MAX_SPEED_M_S

BASELINES = ('forget', 'hold', 'constant_velocity', 'prior')
VELOCITY_FRAMES = 4
AGE_BINS = (('never seen', np.inf, np.inf), ('up to 1.2 s', 0, KEEP_UNSEEN_S), ('1.2-5 s', KEEP_UNSEEN_S, 5.),
            ('5-20 s', 5., 20.), ('older', 20., np.inf))
REAR = (ml.GRID[:, 0] < 0) & (np.linalg.norm(ml.GRID, axis=1) < .8)


def stamp(body):
    """Cells within `OCCUPIED_M` of each body-frame point (P, 2): (P, cells) bool."""
    return np.linalg.norm(ml.GRID[None]-np.asarray(body).reshape(-1, 1, 2), axis=2) < ml.OCCUPIED_M


def cell_of(body):
    """Flat cell index of a body-frame point, or -1 off the grid."""
    i, j = np.floor((np.asarray(body)+ml.HALF_M)/ml.CELL_M).astype(int)
    return i*ml.CELLS+j if 0 <= i < ml.CELLS and 0 <= j < ml.CELLS else -1


def unit_grids(folder, prior):
    """One unit's labelled frames `t`, labels (`occupied`, `visible`, (T, cells)), each cell's sighting age (`age`,
    of the walker nearest the cell), whether its walkers all yield (`polite`), and each baseline's scores (T, cells)."""
    folder = Path(folder)
    with np.load(folder/ml.FILE) as d:
        ts, occupied, visible, seen, age, polite = (d[k] for k in ('t', 'occupied', 'visible', 'seen', 'age_s', 'polite'))
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose']
    with np.load(folder/'dynamic.npz') as dynamic:
        movers = dynamic['obstacles_xy'].astype(float)
    T, K = len(ts), movers.shape[1]
    occupied, visible = occupied.reshape(T, -1), visible.reshape(T, -1)
    scores = {b: np.zeros((T, ml.CELLS**2), np.float32) for b in BASELINES}
    scores['prior'][:] = prior
    last, velocity = np.full(K, -1), np.zeros((K, 2))
    cleared = np.zeros((2, K), bool)                    # hold, constant velocity
    t_prev = -1
    for i, t in enumerate(ts):
        for u in range(t_prev+1, t+1):                  # sightings since the last labelled frame
            for k in np.flatnonzero(seen[u]):
                last[k] = u
                cleared[:, k] = False
                back = u-VELOCITY_FRAMES
                velocity[k] = ((movers[u, k]-movers[back, k])/(VELOCITY_FRAMES*ml.FRAME_S)
                               if back >= 0 and seen[back, k] else 0.)
                speed = np.linalg.norm(velocity[k])
                if speed > MAX_SPEED_M_S:
                    velocity[k] *= MAX_SPEED_M_S/speed
        t_prev = t
        for k in np.flatnonzero(last >= 0):
            if seen[t, k]:
                continue
            a = (t-last[k])*ml.FRAME_S
            held, walked = movers[last[k], k], movers[last[k], k]+velocity[k]*a
            if a <= KEEP_UNSEEN_S:
                scores['forget'][i] = np.maximum(scores['forget'][i], stamp(mp.body_frame(pose[t], walked))[0])
            for n, (name, point) in enumerate((('hold', held), ('constant_velocity', walked))):
                body = mp.body_frame(pose[t], point)
                c = cell_of(body)
                if c >= 0 and visible[i, c]:            # the cameras see the place, and the walker is not seen there
                    cleared[n, k] = True
                if not cleared[n, k]:
                    scores[name][i] = np.maximum(scores[name][i], stamp(body)[0]/(1.+a))
    nearest = np.zeros((T, ml.CELLS**2), int)
    for i, t in enumerate(ts):
        nearest[i] = np.linalg.norm(ml.GRID[:, None]-mp.body_frame(pose[t], movers[t])[None], axis=2).argmin(1)
    return dict(t=ts, occupied=occupied, visible=visible, age=np.take_along_axis(age, nearest, 1).astype(np.float32),
                polite=bool(polite.all()), scores=scores)


def cells(grids, rows=None, out=True):
    """The flat arrays `report` scores, from `unit_grids` at frames `rows` (all by default), on the cells out of
    the camera's view (`out`) or in it."""
    rows = slice(None) if rows is None else rows
    mask = ~grids['visible'][rows] if out else grids['visible'][rows]
    return dict(label=grids['occupied'][rows][mask], age=grids['age'][rows][mask],
                rear=np.broadcast_to(REAR, mask.shape)[mask], rude=np.full(int(mask.sum()), not grids['polite']),
                **{b: grids['scores'][b][rows][mask] for b in BASELINES})


def unit_scores(args):
    """Baseline scores and labels on one unit's out-of-view cells (None for a unit without labels)."""
    folder, prior = Path(args[0]), args[1]
    if not (folder/ml.FILE).exists():
        return None
    return cells(unit_grids(folder, prior))


def average_precision(score, label):
    if not label.any():
        return float('nan')
    order = np.argsort(-score, kind='stable')
    hit = label[order]
    precision = np.cumsum(hit)/np.arange(1, len(hit)+1)
    # ties share their group's precision: score each run of equal scores at its end
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(hit)-1]
    gained = np.diff(np.r_[0, np.cumsum(hit)[ends]])
    return float((precision[ends]*gained).sum()/hit.sum())


def report(data):
    label = data['label']
    out = {}
    for b in BASELINES:
        s = data[b]
        claim = s > 0
        row = dict(ap=round(average_precision(s, label), 4))
        if b != 'prior':
            row.update(precision=round(float(label[claim].mean()), 4) if claim.any() else None,
                       recall=round(float(claim[label].mean()), 4))
            row['recall_by_age'] = {name: round(float(claim[label & (data['age'] > lo) & (data['age'] <= hi)].mean()), 4)
                                    if name != 'never seen' else round(float(claim[label & np.isinf(data['age'])].mean()), 4)
                                    for name, lo, hi in AGE_BINS}
        out[b] = row
    out['occupied_cells'] = int(label.sum())
    out['cells'] = len(label)
    out['occupied_by_age'] = {name: int((label & (np.isinf(data['age']) if name == 'never seen' else
                                                  (data['age'] > lo) & (data['age'] <= hi))).sum()) for name, lo, hi in AGE_BINS}
    return out


def prior_rates(root, workers):
    folders = sorted((Path(root)/'units'/'lewm_train').iterdir())
    hits, counts = np.zeros(ml.CELLS**2), np.zeros(ml.CELLS**2)
    for f in folders:
        if not (f/ml.FILE).exists():
            continue
        with np.load(f/ml.FILE) as d:
            occ, vis = d['occupied'].reshape(len(d['t']), -1), d['visible'].reshape(len(d['t']), -1)
        hits += (occ & ~vis).sum(0)
        counts += (~vis).sum(0)
    return (hits/np.maximum(counts, 1)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    prior = prior_rates(a.root, a.workers)
    folders = sorted((a.root/'units'/'lewm_val').iterdir())
    with ProcessPoolExecutor(a.workers or os.cpu_count(), mp_context=multiprocessing.get_context('spawn')) as pool:
        parts = [r for r in pool.map(unit_scores, [(str(f), prior) for f in folders], chunksize=2) if r is not None]
    data = {k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
    result = dict(units=len(parts), all=report(data))
    for name, mask in (('rear', data['rear']), ('do_not_yield', data['rude']), ('rear_do_not_yield', data['rear'] & data['rude'])):
        result[name] = report({k: v[mask] for k, v in data.items()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
