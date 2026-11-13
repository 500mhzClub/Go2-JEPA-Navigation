"""P22 item 13 (docs/dynamic/PLAN.md): how well a model's chance of a walker contact matches what happens, at the plans
the planner makes near walkers rather than only at stand-offs, and whether its wrong choices come from false alarms
(the freezing) or from misses (the contacts).

    python -m go2nav.eval.contact_calibration run --model SPEC --out DIR [--kind rude] [--walkers 4] [--seed 0]
        [--near 3.0] [--every 5] [--limit 30] [--workers N]
    python -m go2nav.eval.contact_calibration summary DIR [DIR ...]

The suite's missions (`eval.kinematic_missions`: the planner of 6.4 s with the walkers' terminal distance, under the
rule the environment sets) run as they do: the probe reads the planner's own call of its model and does not call it
again. At every `every`-th plan with a walker centre within `near` metres of the robot's, at most `limit` a mission,
each of the plan's tapes that keeps off the walls (`FLOOR_M`) is scored along the robot's path under it
(`planning.sampling.integrate`):
- **the model's chance** that some walker comes within the floor by the horizon, as the worked-out rule reads it
  (`planning.sampling.Planner.expected_costs`: one less the product over walkers of one less the share of its futures
  within; with the way out, a walker already under the floor counts only when the tape goes nearer); for a model of
  one future, 0 or 1;
- **the truth:** whether the world's walkers, copied with their dice and stepped with the robot under that tape
  (`sim.kinematic.World.step`, exact), do;
- **the tape's cost under the truth:** the worked-out rule with that one future, the walkers' terminal distance from
  where the truth leaves them.
Per mission, DIR/<maze>_<episode>_<walkers>.json: the mission's row and `probes` (per probe: the frame, the nearest
walker, the chosen tape and its kind; per tape kept: the model's chance, the truth, the cost under the model and under
the truth).

**summary:** calibration (per band of the model's chance, the tapes and the share truly within), the area under the
curve of the chance against the truth, the Brier score against that of the base rate; on the chosen tapes, the share
truly within and the model's chance there; and the regret, the chosen tape's cost under the truth less the best kept
tape's. A probe with a regret of `REGRET_S` or more is a wrong choice, put down to:
- **a miss:** the chosen tape truly comes within the floor (avoidable when the best tape under the truth does not);
- **a false alarm:** it does not, and the model gave the best tape under the truth a chance of `ALARM_P` or more;
- **the rest:** neither (where the model leaves the walkers for the distance left, the time, the clearance).
For each wrong choice, the cost's terms (time, clearance, progress, contact; `split`) under the model and the truth
give each term's part in it: how much the model over-costed the best tape and under-costed the chosen one on that
term; the parts sum to at least the regret. Reported as their mean and as how often each term is the largest.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

MISS_P, ALARM_P, REGRET_S = .1, .5, 2.
BANDS = (0., .01, .05, .1, .2, .3, .5, .7, .9, 1.)


def chance_within(snapshot, xy, yaw, walkers, weights=None, groups=None, recover=True):
    """(K,): the chance that some walker's path (K, T, J, 2) comes within the floor of the robot's paths (K, T) by the
    end, futures weighed per walker (`groups`) as `Planner.expected_costs` does; with no weights, 0 or 1."""
    from go2nav.planning.sampling import FLOOR_M
    from go2nav.sim import kinematic as kin
    from go2nav.sim import obstacles as moving
    gaps = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None])-moving.RADIUS_M                   # (K, T, J)
    floor = FLOOR_M
    if recover:
        now = kin.rectangle_gap(walkers[:, 0], snapshot['xy'], snapshot['yaw'])-moving.RADIUS_M            # (K, J)
        floor = np.minimum(FLOOR_M, now-.005)[:, None]
    within = (gaps < floor).any(1)                                                                       # (K, J)
    if weights is None:
        return within.any(1).astype(float)
    w = np.clip(np.asarray(weights, float), 0., None)
    clear = np.ones(len(within))
    for g in np.unique(groups):
        members = groups == g
        total = w[:, members].sum(1, keepdims=True)
        share = np.where(total > 0, w[:, members]/np.where(total > 0, total, 1.), 0.)
        clear = clear*(1.-np.clip((share*within[:, members]).sum(1), 0., 1.))
    return 1.-clear


TERMS = ('time', 'clearance', 'progress', 'contact')


def split(total, terms):
    """A tape's cost (K,) in its parts (4, K): time, clearance, progress (the distance left, and a sealed way), and the
    rest (`contact`: the breach's cost, and under the priced rule the risk)."""
    parts = [np.asarray(terms[k], float) for k in TERMS[:3]]
    return np.stack(parts+[np.asarray(total, float)-sum(parts)])


def kind_of(name):
    """A proposal's kind from its name (`Planner.proposals`)."""
    if name.startswith('cell'):
        return 'cell_backing' if name.endswith('_backing') else 'cell'
    if name.startswith('gaussian'):
        return 'gaussian'
    return name.split('@')[0]


class Probe:
    def __init__(self, near, every, limit):
        self.near, self.every, self.limit = near, every, limit
        self.eligible, self.rows, self.attached, self.last = 0, [], False, {}

    def _attach(self, planner):
        """Record the plan's tapes, their costs and the model's futures as the planner makes them."""
        model, walkers, costs, proposals = planner.model, planner.model.walkers, planner.costs, planner.proposals

        def recorded_walkers(*args, **kwargs):
            out = walkers(*args, **kwargs)
            weights, groups = getattr(model, 'weights', None), getattr(model, 'groups', None)
            self.last.update(walkers=out.copy(), weights=None if weights is None else np.array(weights, float),
                             groups=None if groups is None else np.array(groups))
            return out

        def recorded_proposals(snapshot, target):
            requested, names = proposals(snapshot, target)
            self.last['names'] = list(names)
            return requested, names

        def recorded_costs(snapshot, target, requested):
            total, terms = costs(snapshot, target, requested)
            self.last.update(requested=requested.copy(), total=np.array(total, float),
                             terms=split(total, terms))
            return total, terms
        model.walkers, planner.costs, planner.proposals, self.attached = recorded_walkers, recorded_costs, \
            recorded_proposals, True

    def __call__(self, planner, world, snapshot, target):
        from go2nav.planning.sampling import FLOOR_M, integrate
        from go2nav.sim import kinematic as kin
        if not self.attached:
            self._attach(planner)
            return
        if not world.walkers:
            return
        nearest = float(np.linalg.norm(world.walkers_xy()-world.xy, axis=1).min())
        if nearest > self.near:
            return
        self.eligible += 1
        if (self.eligible-1) % self.every or len(self.rows) >= self.limit or 'total' not in self.last:
            return
        last, self.last = self.last, {}
        requested, total = last['requested'], last['total']
        xy, yaw = integrate(snapshot, requested)
        walls = planner.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1)                     # (K, T)
        kept = np.flatnonzero(walls.min(1) >= FLOOR_M)
        if not len(kept) or last['walkers'].shape[2] == 0:
            return
        recover = getattr(planner, 'recover', False)
        weights = None if last['weights'] is None else last['weights'][kept]
        chance = chance_within(snapshot, xy[kept], yaw[kept], last['walkers'][kept], weights, last['groups'], recover)
        frames, n = requested.shape[1], len(world.walkers)
        truth = np.zeros((len(kept), frames, n, 2))
        for i, k in enumerate(kept):
            branch = copy.deepcopy(world)
            for f in range(frames):
                branch.step(requested[k, f])
                truth[i, f] = branch.walkers_xy()
        truly = chance_within(snapshot, xy[kept], yaw[kept], truth, recover=recover) > .5
        true_cost, true_terms = planner.expected_costs(snapshot, target, xy[kept], yaw[kept], walls[kept], truth,
                                                       np.ones((len(kept), n)), np.arange(n), ends=truth[:, -1])
        k = int(total.argmin())
        chosen = int(np.flatnonzero(kept == k)[0]) if k in kept else -1
        self.rows.append(dict(frame=int(world.frame), nearest_m=round(nearest, 3), chosen=chosen,
                              chosen_kind=kind_of(last['names'][k]) if 'names' in last else None,
                              kinds=[kind_of(last['names'][j]) for j in kept] if 'names' in last else None,
                              chance=np.round(chance, 4).tolist(), truth=truly.tolist(),
                              cost=np.round(total[kept], 2).tolist(), true_cost=np.round(true_cost, 2).tolist(),
                              terms=np.round(last['terms'][:, kept], 2).tolist(),
                              true_terms=np.round(split(true_cost, true_terms), 2).tolist()))


def _job(args):
    out, key, kwargs, probe_kwargs = args
    from go2nav.eval.kinematic_missions import run_mission
    target = Path(out)/f'{key[1]:02d}_{key[2]}_{key[3]}.json'
    if target.exists():
        return json.loads(target.read_text())
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    probe = Probe(**probe_kwargs)
    row = run_mission(key[0], key[1], key[2], walkers=key[3], probe=probe, **kwargs)
    row['probes'] = probe.rows
    partial = target.with_suffix('.tmp')
    partial.write_text(json.dumps(row)+'\n')
    partial.rename(target)
    return row


def run(out, *, model, kind='rude', walkers=(4,), seed=0, near=3., every=5, limit=30, workers=None,
        set_name='validation', mazes=range(10, 30), episodes=(0, 1)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kind=kind, horizon_s=6.4, terminal='walkers', seed=seed, model=model)
    probe_kwargs = dict(near=near, every=every, limit=limit)
    rule = {k: v for k, v in os.environ.items() if k.startswith('GO2NAV_')}
    (out/'config.json').write_text(json.dumps(dict(kwargs, walkers=list(walkers), **probe_kwargs, rule=rule))+'\n')
    jobs = [(str(out), (set_name, m, e, n), kwargs, probe_kwargs) for n in walkers for m in mazes for e in episodes]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(_job, jobs))
    report = summary(rows)
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def summary(rows):
    from go2nav.eval.tape_optimism import _auc
    probes = [p for r in rows for p in r.get('probes', [])]
    chance = np.array([c for p in probes for c in p['chance']], float)
    truth = np.array([t for p in probes for t in p['truth']], bool)
    bands = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        inside = (chance >= lo) & ((chance < hi) if hi < 1. else (chance <= hi))
        bands.append(dict(band=f'{lo:g}-{hi:g}', tapes=int(inside.sum()),
                          mean_chance=round(float(chance[inside].mean()), 3) if inside.any() else None,
                          truly_within=round(float(truth[inside].mean()), 3) if inside.any() else None))
    chosen = [p for p in probes if p['chosen'] >= 0]
    picked_within = [p for p in chosen if p['truth'][p['chosen']]]
    causes, kinds, regrets, avoidable = dict(miss=0, false_alarm=0, rest=0), {}, [], 0
    errors, largest = {name: [] for name in TERMS}, dict.fromkeys(TERMS, 0)
    for p in chosen:
        true_cost = np.array(p['true_cost'])
        regret = float(true_cost[p['chosen']]-true_cost.min())
        regrets.append(regret)
        if regret < REGRET_S:
            continue
        best = int(true_cost.argmin())
        cause = 'miss' if p['truth'][p['chosen']] else ('false_alarm' if p['chance'][best] >= ALARM_P else 'rest')
        causes[cause] += 1
        avoidable += int(cause == 'miss' and not p['truth'][best])          # a miss with a tape truly clear to take
        if 'terms' in p:          # each term's error in the choice: the best tape over-costed, the chosen under-costed
            m, t = np.array(p['terms']), np.array(p['true_terms'])
            tipped = (m[:, best]-t[:, best])-(m[:, p['chosen']]-t[:, p['chosen']])
            for i, name in enumerate(TERMS):
                errors[name].append(float(tipped[i]))
            largest[TERMS[int(tipped.argmax())]] += 1
        kinds.setdefault(cause, {})
        kinds[cause][p['chosen_kind']] = kinds[cause].get(p['chosen_kind'], 0)+1
    regrets = np.array(regrets)
    base = float(truth.mean()) if len(truth) else None
    return dict(
        missions=len(rows), round_trips=sum(r['round_trip'] for r in rows),
        contacts=sum(r['exit'] == 'contact_walker' for r in rows), probes=len(probes), tapes=int(len(chance)),
        truly_within_share=None if base is None else round(base, 3),
        auc=None if not len(truth) else (lambda a: None if a is None else round(a, 3))(_auc(chance, truth)),
        brier=None if not len(truth) else round(float(((chance-truth)**2).mean()), 4),
        brier_of_base_rate=None if base is None else round(base*(1-base), 4),
        mean_chance_truly_within=round(float(chance[truth].mean()), 3) if truth.any() else None,
        mean_chance_truly_clear=round(float(chance[~truth].mean()), 3) if (~truth).any() else None,
        misses=round(float((chance[truth] < MISS_P).mean()), 3) if truth.any() else None,
        false_alarms=round(float((chance[~truth] >= ALARM_P).mean()), 3) if (~truth).any() else None,
        bands=bands,
        chosen_truly_within=round(len(picked_within)/max(len(chosen), 1), 3),
        chosen_truly_within_mean_chance=round(float(np.mean([p['chance'][p['chosen']] for p in picked_within])), 3)
        if picked_within else None,
        mean_regret_s=round(float(regrets.mean()), 2) if len(regrets) else None,
        wrong_choices=int((regrets >= REGRET_S).sum()), wrong_share=round(float((regrets >= REGRET_S).mean()), 3)
        if len(regrets) else None, causes=causes, avoidable_misses=avoidable, chosen_kinds_by_cause=kinds,
        **({} if not errors['time'] else dict(
            wrong_choice_error_by_term={k: round(float(np.mean(v)), 2) for k, v in errors.items()},
            wrong_choice_largest_term=largest)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--kind', default='rude')
    r.add_argument('--walkers', type=int, nargs='+', default=[4])
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--near', type=float, default=3.)
    r.add_argument('--every', type=int, default=5)
    r.add_argument('--limit', type=int, default=30)
    r.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('folders', type=Path, nargs='+')
    a = p.parse_args()
    if a.command == 'run':
        report = run(a.out, model=a.model, kind=a.kind, walkers=a.walkers, seed=a.seed, near=a.near, every=a.every,
                     limit=a.limit, workers=a.workers)
    else:
        report = {str(f): summary([json.loads(x.read_text()) for x in sorted(f.glob('[0-9]*.json'))]) for f in a.folders}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
