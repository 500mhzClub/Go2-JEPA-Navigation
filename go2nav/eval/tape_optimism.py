"""P22 item 10's last measure (docs/dynamic/PLAN.md): among the planner's tapes, how often does its model have a walker
keep clear that truly would not, and does the planner choose those tapes?

    python -m go2nav.eval.tape_optimism run --model SPEC --out DIR [--kind rude] [--walkers 4] [--seed 0]
        [--every 5] [--limit 8] [--workers N]
    python -m go2nav.eval.tape_optimism summary DIR [DIR ...]

The suite's missions (`eval.kinematic_missions`: the planner of 6.4 s with the walkers' terminal distance) run as they
do; the probe does not change them. At every `every`-th plan at which a walker is held up by the robot
(`sim.kinematic.waiting_on`, a label used only to choose), at most `limit` a mission, each of the plan's tapes that
keeps off the walls is scored along the robot's path under it (`planning.sampling.integrate`), against the floor the
planner's expected cost uses (`FLOOR_M`, or for a walker already under it, nearer than now):
- **the model's chance** of coming within it, as the expected cost reads it: one less the chance that no future of
  any walker does (`planning.sampling.Planner.costs`); for a model of one future, 0 or 1;
- **the truth:** whether the world's walkers, copied with their dice and stepped with the robot under that tape
  (`sim.kinematic.World.step`, exact), do.
Per mission, DIR/<maze>_<episode>_<walkers>.json: the mission's row and `probes` (per probe, per tape kept: the
model's chance and the truth; the chosen tape).

**summary:** over probes, the share whose chosen tape truly comes within the floor, and of those the share the model
gave a chance under `OPTIMISTIC_P`; over tapes, the mean chance on tapes that truly come within and on those that do
not, optimism (truly within, chance under `OPTIMISTIC_P`), pessimism (truly clear, chance `PESSIMISTIC_P` or more),
and the area under the curve of the chance against the truth.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

OPTIMISTIC_P, PESSIMISTIC_P = .1, .5


def _within(walkers, snapshot, xy, yaw):
    """(K, n): whether each walker's path (K, T, n, 2) comes within the floor of the robot's paths."""
    from go2nav.planning.sampling import FLOOR_M
    from go2nav.sim import kinematic as kin
    from go2nav.sim import obstacles as moving
    gaps = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None])-moving.RADIUS_M
    now = kin.rectangle_gap(walkers[:, 0], snapshot['xy'], snapshot['yaw'])-moving.RADIUS_M
    return (gaps < np.minimum(FLOOR_M, now-.005)[:, None]).any(1)


class Probe:
    def __init__(self, every, limit):
        self.every, self.limit, self.held, self.rows, self.last, self.attached = every, limit, 0, [], None, False

    def _attach(self, planner):
        costs = planner.costs

        def recorded(snapshot, target, requested):
            total, terms = costs(snapshot, target, requested)
            self.last = (requested.copy(), total.copy(), terms['breached'].copy())
            return total, terms
        planner.costs, self.attached = recorded, True

    def __call__(self, planner, world, snapshot, target):
        from go2nav.planning.sampling import integrate
        from go2nav.sim import kinematic as kin
        if not self.attached:
            self._attach(planner)
            return
        if not any(kin.waiting_on(w, world.xy) for w in world.walkers):
            return
        self.held += 1
        if (self.held-1) % self.every or len(self.rows) >= self.limit or self.last is None:
            return
        requested, total, breached = self.last
        kept = np.flatnonzero(~breached)
        if not len(kept):
            return
        tapes = requested[kept]
        xy, yaw = integrate(snapshot, tapes)
        count, frames = tapes.shape[:2]
        model = planner.model
        extra = dict(requested=tapes) if getattr(model, 'takes_tapes', False) else {}
        walkers = model.walkers(snapshot, xy, yaw, np.full(count, frames), planner.geometry, **extra)
        within = _within(walkers, snapshot, xy, yaw)
        weights = getattr(model, 'weights', None)
        if weights is not None:
            clear = np.ones(count)
            for g in np.unique(model.groups):
                clear = clear*(1.-np.clip((weights[:, model.groups == g]*within[:, model.groups == g]).sum(1), 0., 1.))
            chance = 1.-clear
        else:
            chance = within.any(1).astype(float)
        truth = np.zeros((count, frames, len(world.walkers), 2))
        contact = np.zeros(count, bool)
        for k in range(count):
            branch = copy.deepcopy(world)
            for f in range(frames):
                branch.step(tapes[k, f])
                truth[k, f] = branch.walkers_xy()
            contact[k] = branch.contact == 'walker'
        true_within = _within(truth, snapshot, xy, yaw).any(1)
        chosen = int(np.flatnonzero(kept == int(total.argmin()))[0]) if int(total.argmin()) in kept else -1
        self.rows.append(dict(frame=int(world.frame), chosen=chosen, chance=np.round(chance, 4).tolist(),
                              truth=true_within.tolist(), contact=contact.tolist()))


def _job(args):
    out, key, kwargs, every, limit = args
    from go2nav.eval.kinematic_missions import run_mission
    target = Path(out)/f'{key[1]:02d}_{key[2]}_{key[3]}.json'
    if target.exists():
        return json.loads(target.read_text())
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    probe = Probe(every, limit)
    row = run_mission(key[0], key[1], key[2], walkers=key[3], probe=probe, **kwargs)
    row['probes'] = probe.rows
    partial = target.with_suffix('.tmp')
    partial.write_text(json.dumps(row)+'\n')
    partial.rename(target)
    return row


def run(out, *, model, kind='rude', walkers=(4,), seed=0, every=5, limit=8, workers=None, set_name='validation',
        mazes=range(10, 30), episodes=(0, 1)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kind=kind, horizon_s=6.4, terminal='walkers', seed=seed, model=model)
    (out/'config.json').write_text(json.dumps(dict(kwargs, walkers=list(walkers), every=every, limit=limit))+'\n')
    jobs = [(str(out), (set_name, m, e, n), kwargs, every, limit) for n in walkers for m in mazes for e in episodes]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(_job, jobs))
    report = summary(rows)
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def _auc(score, label):
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate((pos, neg)), kind='mergesort')
    ranks = np.empty(len(order))
    values = np.concatenate((pos, neg))[order]
    i = 0
    while i < len(values):                                    # mid-ranks for ties
        j = i
        while j+1 < len(values) and values[j+1] == values[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j)/2+1
        i = j+1
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def summary(rows):
    probes = [p for r in rows for p in r.get('probes', [])]
    chance = np.array([c for p in probes for c in p['chance']])
    truth = np.array([t for p in probes for t in p['truth']], bool)
    chosen = [p for p in probes if p['chosen'] >= 0]
    picked_within = [p for p in chosen if p['truth'][p['chosen']]]
    return dict(
        missions=len(rows), round_trips=sum(r['round_trip'] for r in rows),
        contacts=sum(r['exit'] == 'contact_walker' for r in rows), probes=len(probes), tapes=int(len(chance)),
        chosen_truly_within=round(len(picked_within)/max(len(chosen), 1), 3),
        chosen_truly_within_model_under=round(sum(p['chance'][p['chosen']] < OPTIMISTIC_P for p in picked_within)
                                              / max(len(picked_within), 1), 3),
        probes_with_an_optimistic_tape=round(sum(any(t and c < OPTIMISTIC_P for c, t in zip(p['chance'], p['truth']))
                                                 for p in probes)/max(len(probes), 1), 3),
        truly_within_share=round(float(truth.mean()), 3) if len(truth) else None,
        mean_chance_truly_within=round(float(chance[truth].mean()), 3) if truth.any() else None,
        mean_chance_truly_clear=round(float(chance[~truth].mean()), 3) if (~truth).any() else None,
        optimism=round(float((chance[truth] < OPTIMISTIC_P).mean()), 3) if truth.any() else None,
        pessimism=round(float((chance[~truth] >= PESSIMISTIC_P).mean()), 3) if (~truth).any() else None,
        auc=None if not len(truth) else (lambda a: None if a is None else round(a, 3))(_auc(chance, truth)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--kind', default='rude')
    r.add_argument('--walkers', type=int, nargs='+', default=[4])
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--every', type=int, default=5)
    r.add_argument('--limit', type=int, default=8)
    r.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('folders', type=Path, nargs='+')
    a = p.parse_args()
    if a.command == 'run':
        report = run(a.out, model=a.model, kind=a.kind, walkers=a.walkers, seed=a.seed, every=a.every, limit=a.limit,
                     workers=a.workers)
    else:
        report = {str(f): summary([json.loads(x.read_text()) for x in sorted(f.glob('[0-9]*.json'))]) for f in a.folders}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
