"""P22's measures of the object-level predictor (docs/dynamic/PLAN.md P22 item 5 (a) and (b)), on P20's val split.

    python -m go2nav.eval.object_measures offline --model PATH --out FILE [--every 8] [--workers N]
    python -m go2nav.eval.object_measures reaction --model PATH --out FILE [--windows 600] [--workers N]

The model runs as the planner runs it (`planning.object_model.ObjectModel`, on the CPU here): the tracker over the
recorded scans, a rollout under a tape. The walkers' true positions, sight and waiting are read only to choose what is
scored and to score it.

**offline** (a): every `every`-th frame t0 of every val episode, the recorded requests of the next 6.4 s as the tape
and the recorded poses as its path. Each true walker in sight at t0 is matched to the nearest slot seen at t0 within
0.3 m; at 1.6, 3.2 and 6.4 s it is scored if it is in sight then and within 3 m of the robot: the error of the
model's position, of the ring tracker's walker carried on at its velocity (`planning.sampling.RingModel`,
`ring_velocity`, the same tracker), and of it frozen. Medians over all such walkers and over those waiting on the
robot at t0.

**reaction** (b): val windows with a walker in sight within 1.5 m of the robot (at most 2 an episode). The episode
is replayed exactly to the window (`data.ring_corpus.episode`, `stop_at`) and branched under two tapes: hold for
6.4 s, and back away at 0.2 m/s for 2 s then hold. The difference in that walker's true position between the
branches, against the model's predicted difference, at 1.6, 3.2 and 6.4 s: the slope with its 95% interval over
episodes (`eval.measures.slope`). Windows with a contact in either branch are left out. Carrying a walker on at its
velocity predicts no difference (slope 0).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

CORPUS = Path('/mnt/workspace_drive/go2nav_ring/corpus_v1')
MAZES = Path('/mnt/workspace_drive/go2nav_lewm_v2/mazes')
HORIZONS = (16, 32, 64)                       # frames: 1.6, 3.2, 6.4 s
MATCH_M, NEAR_M, CLOSE_M = .3, 3., 1.5
BACK_M_S, BACK_FRAMES, TAPE_FRAMES = .2, 20, 64


def _setup(meta, model_path):
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.fields import Fields
    from go2nav.planning.object_model import ObjectModel
    from go2nav.planning.sampling import RingModel
    from go2nav.sim import kinematic as kin
    spec, _ = episode_spec('lewm_sets_v1', meta['maze'], meta['home_episode'], root=MAZES)
    geometry = kin.Geometry(spec)
    fields = Fields(geometry)
    model, ring = ObjectModel(model_path, device='cpu'), RingModel('ring_velocity')
    model.bind(fields), ring.bind(fields)
    return geometry, fields, model, ring


def _matched(slots_xy, seen, walker):
    if not seen.any():
        return None
    d = np.where(seen, np.linalg.norm(np.nan_to_num(slots_xy, nan=1e3)-walker, axis=1), np.inf)
    k = int(d.argmin())
    return k if d[k] <= MATCH_M else None


def _offline_episode(args):
    index, model_path, every = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)
    from go2nav.planning.sampling import carried
    meta = json.loads((CORPUS/'val'/f'{index:05d}.json').read_text())
    if meta['walkers'] == 0:
        return []
    with np.load(CORPUS/'val'/f'{index:05d}.npz') as d:
        ranges, pose, requested = d['ranges'].astype(np.float32), d['pose'], d['requested']
        truth, sight, waiting = d['walkers_xy'], d['in_sight'], d['waiting']
    geometry, fields, model, ring = _setup(meta, model_path)
    rows, frames = [], len(pose)
    for t in range(frames):
        scan = (ranges[t], ranges[t] > 0)
        model.observe(t*.1, scan, pose[t, :2], float(pose[t, 2]), geometry)
        ring.observe(t*.1, scan, pose[t, :2], float(pose[t, 2]), geometry)
        if t >= 8 and t % every == 0 and t+TAPE_FRAMES < frames and sight[t].any():
            tape = requested[t:t+TAPE_FRAMES][None]
            xy, yaw = pose[t+1:t+1+TAPE_FRAMES, :2][None], pose[t+1:t+1+TAPE_FRAMES, 2][None]
            predicted = model.walkers(None, xy, yaw, None, geometry, requested=tape)[0]
            now = model.frames[-1]
            at, velocity = ring.tracker.state()
            for j in np.flatnonzero(sight[t]):
                k = _matched(now['xy'], now['seen'], truth[t, j])
                if k is None or not len(at):
                    continue
                r = int(np.linalg.norm(at-truth[t, j], axis=1).argmin())
                if np.linalg.norm(at[r]-truth[t, j]) > MATCH_M:
                    continue
                cv = carried(at[r:r+1], velocity[r:r+1], 1, TAPE_FRAMES, geometry)[0, :, 0]
                for h in HORIZONS:
                    target = truth[t+h, j]
                    if not sight[t+h, j] or np.linalg.norm(target-pose[t+h, :2]) > NEAR_M:
                        continue
                    rows.append(dict(episode=index, kind=meta['kind'], horizon=h, waiting=bool(waiting[t, j]),
                                     model=float(np.linalg.norm(predicted[h-1, k]-target)),
                                     velocity=float(np.linalg.norm(cv[h-1]-target)),
                                     frozen=float(np.linalg.norm(at[r]-target))))
        model.record(requested[t])
    return rows


def offline(model_path, every=8, workers=None):
    indices = sorted(int(p.stem) for p in (CORPUS/'val').glob('*.json'))
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = [r for part in pool.map(_offline_episode, [(i, str(model_path), every) for i in indices]) for r in part]
    report = {}
    for h in HORIZONS:
        for name, keep in (('all', lambda r: True), ('waiting', lambda r: r['waiting'])):
            sel = [r for r in rows if r['horizon'] == h and keep(r)]
            if sel:
                report[f'{h/10:.1f}s_{name}'] = dict(count=len(sel), **{m: round(float(np.median([r[m] for r in sel])), 4)
                                                                         for m in ('model', 'velocity', 'frozen')})
    return report


def _reaction_window(args):
    index, t0, model_path = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)
    from go2nav.data import ring_corpus as rc
    from go2nav.planning.sampling import integrate
    meta = json.loads((CORPUS/'val'/f'{index:05d}.json').read_text())
    with np.load(CORPUS/'val'/f'{index:05d}.npz') as d:
        ranges, pose, requested, waiting = d['ranges'].astype(np.float32), d['pose'], d['requested'], d['waiting']
    world, _ = rc.episode(MAZES, index, 'val', stop_at=t0)
    geometry, fields, model, _ = _setup(meta, model_path)
    for t in range(t0+1):
        model.observe(t*.1, (ranges[t], ranges[t] > 0), pose[t, :2], float(pose[t, 2]), geometry)
        if t < t0:
            model.record(requested[t])
    walkers = world.walkers_xy()
    distance = np.linalg.norm(walkers-world.xy, axis=1)
    j = int(distance.argmin())
    now = model.frames[-1]
    k = _matched(now['xy'], now['seen'], walkers[j])
    if k is None:
        return None
    tapes = np.zeros((2, TAPE_FRAMES, 3), np.float32)
    tapes[1, :BACK_FRAMES, 0] = -BACK_M_S
    true = []
    for tape in tapes:
        branch = copy.deepcopy(world)
        path = []
        for f in range(TAPE_FRAMES):
            branch.step(tape[f])
            path.append(branch.walkers_xy()[j])
        if branch.contact is not None:
            return None
        true.append(np.array(path))
    xy, yaw = integrate(dict(xy=world.xy, yaw=world.yaw, executed=world.executed), tapes)
    predicted = model.walkers(None, xy, yaw, None, geometry, requested=tapes)[:, :, k]
    return dict(episode=index, t0=t0, kind=meta['kind'], waiting=bool(waiting[t0, j]),
                true=[(true[1][h-1]-true[0][h-1]).tolist() for h in HORIZONS],
                predicted=[(predicted[1, h-1]-predicted[0, h-1]).tolist() for h in HORIZONS])


def reaction(model_path, windows=600, workers=None, seed=2026111105):
    from go2nav.eval.measures import slope
    rng = np.random.default_rng(seed)
    jobs = []
    for p in sorted((CORPUS/'val').glob('*.json')):
        meta = json.loads(p.read_text())
        if meta['walkers'] == 0:
            continue
        with np.load(p.with_suffix('.npz')) as d:
            pose, truth, sight = d['pose'], d['walkers_xy'], d['in_sight']
        close = np.nanmin(np.where(sight, np.linalg.norm(truth-pose[:, None, :2], axis=-1), np.inf), axis=1)
        frames = [t for t in range(8, len(pose)-TAPE_FRAMES) if close[t] <= CLOSE_M]
        for t in rng.choice(frames, min(2, len(frames)), replace=False) if frames else []:
            jobs.append((int(p.stem), int(t), str(model_path)))
    rng.shuffle(jobs)
    jobs = jobs[:windows]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = [r for r in pool.map(_reaction_window, jobs) if r is not None]
    true, predicted = np.array([r['true'] for r in rows]), np.array([r['predicted'] for r in rows])
    groups = np.array([r['episode'] for r in rows])
    report = dict(windows=len(rows), asked=len(jobs))
    for i, h in enumerate(HORIZONS):
        moved = np.linalg.norm(true[:, i], axis=1) > .05
        report[f'{h/10:.1f}s'] = dict(slope(predicted[:, i], true[:, i], groups), true_moved_share=round(float(moved.mean()), 3),
                                      true_mean_m=round(float(np.linalg.norm(true[:, i], axis=1).mean()), 3),
                                      predicted_mean_m=round(float(np.linalg.norm(predicted[:, i], axis=1).mean()), 3))
    report['rows'] = rows
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('offline', 'reaction'):
        s = sub.add_parser(name)
        s.add_argument('--model', required=True, type=Path)
        s.add_argument('--out', required=True, type=Path)
        s.add_argument('--workers', type=int)
    sub.choices['offline'].add_argument('--every', type=int, default=8)
    sub.choices['reaction'].add_argument('--windows', type=int, default=600)
    a = p.parse_args()
    report = offline(a.model, a.every, a.workers) if a.command == 'offline' else reaction(a.model, a.windows, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
