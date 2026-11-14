"""P22 item 14 (docs/dynamic/PLAN.md): a bank of planning moments, fixed once, at which every model chooses among the
same tapes and is scored against the truth. A paired measure: every model meets the same states, so two models can be
ranked in minutes, where missions take hours and each model's own probes see different states.

    python -m go2nav.eval.decision_bank build --out DIR [--sources NAME ...] [--workers N]
    python -m go2nav.eval.decision_bank score --bank DIR --model SPEC --out FILE [--workers N]
    python -m go2nav.eval.decision_bank summary FILE [FILE ...]

**build.** Each source (`SOURCES`: the planner on a model, on a suite of `eval.kinematic_missions`, under the rule the
environment sets) runs its missions exactly as a suite does, recording every frame's request. Plans with a walker
centre within `NEAR_M` of the robot's are eligible; up to `PER_MISSION` of them, evenly spaced over the mission, become
moments: the frame, the target, the plan's tapes and their names, and the tape the source chose. The mission is then
replayed by its requests (the world repeats a mission given them; the pose is checked at each moment), and at each
moment every tape that keeps off the walls (`FLOOR_M`) is stepped in a copy of the world, the walkers' dice included:
whether a walker comes within the floor, and the tape's cost under that truth (the worked-out rule with that one
future, the walkers' terminal distance from where it leaves them), as `eval.contact_calibration` scores its probes.
Per mission, DIR/<source>/<maze>_<episode>_<walkers>.npz (the requests; each moment's tapes) and .json (the mission's
row and the moments); a finished one is skipped on a rerun.

**score.** Each mission is replayed by its requests, the model told every frame's scan (`observe`) and request
(`record`) as in a mission, so at a moment it holds the history it would have held there. The planner costs the
moment's tapes on the model and takes the cheapest. Per moment: the chosen tape, whether it is the source's, the
model's chance of a contact on each kept tape (`contact_calibration.chance_within`), the costs and their terms. The
summary is `contact_calibration.summary`'s (regret under the truth, wrong choices by cause, the chosen tapes truly
within, calibration), in all and per source, with the share of choices that agree with the source's. A model that
held the truth would agree with the true model's choices and have a regret near nothing (its walkers step at 100 ms,
the world's at 2 ms).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

NEAR_M, PER_MISSION = 3., 4
HORIZON_S, TERMINAL = 6.4, 'walkers'
MAZES, EPISODES, SET = tuple(range(10, 30)), (0, 1), 'validation'
SOURCES = {           # name: (model, kind, walker counts, planner seed)
    'tracker_rude4': ('ring_velocity', 'rude', (4,), 0),
    'true_rude4': ('true', 'rude', (4,), 0),
    'tracker_rude12': ('ring_velocity', 'rude', (1, 2), 0),
    'true_rude12': ('true', 'rude', (1, 2), 0),
    'tracker_polite12': ('ring_velocity', 'polite', (1, 2), 0),
    'true_polite12': ('true', 'polite', (1, 2), 0),
    'tracker_reactive12': ('ring_velocity', 'reactive', (1, 2), 0),
    'true_reactive12': ('true', 'reactive', (1, 2), 0),
}


def rule():
    """The rule the environment sets (the planner's development switches)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith('GO2NAV_')}


def mission_world(set_name, maze, episode_index, walkers, kind):
    """The world a suite's mission starts from (`eval.kinematic_missions.run_mission`)."""
    from go2nav.eval.kinematic_missions import placement_for
    from go2nav.eval.mission import episode_spec
    from go2nav.sim.kinematic import World
    spec, episode = episode_spec(set_name, maze, episode_index)
    return World(spec, episode['home_se2_world'], placement_for(set_name, maze, episode_index, spec, episode, walkers,
                                                                kind))


class Recorder:
    """A mission probe (`run_mission(probe=...)`) that keeps every eligible plan's tapes and every frame's request,
    and changes nothing."""

    def __init__(self, near=NEAR_M):
        self.near, self.eligible, self.steps, self.attached, self.last = near, [], [], False, {}

    def _attach(self, planner, world):
        proposals, costs, step = planner.proposals, planner.costs, world.step

        def recorded_proposals(snapshot, target):
            requested, names = proposals(snapshot, target)
            self.last.update(requested=np.array(requested, float), names=list(names))
            return requested, names

        def recorded_costs(snapshot, target, requested):
            total, terms = costs(snapshot, target, requested)
            self.last['total'] = np.array(total, float)
            return total, terms

        def recorded_step(requested):
            self.steps.append(np.array(requested, np.float32))
            return step(requested)
        planner.proposals, planner.costs, world.step, self.attached = recorded_proposals, recorded_costs, \
            recorded_step, True

    def __call__(self, planner, world, snapshot, target):
        if not self.attached:                     # the first plan: from here on (before it, every request is a hold)
            self.steps = [np.zeros(3, np.float32)]*int(world.frame)
            self._attach(planner, world)
            return
        last, self.last = self.last, {}
        if not world.walkers or 'total' not in last:
            return
        nearest = float(np.linalg.norm(world.walkers_xy()-world.xy, axis=1).min())
        if nearest > self.near:
            return
        self.eligible.append(dict(frame=int(world.frame), target=np.asarray(target, float).copy(), tapes=last['requested'],
                                  names=last['names'], chosen=int(last['total'].argmin()), nearest_m=nearest,
                                  xy=snapshot['xy'].copy(), yaw=float(snapshot['yaw'])))


def spaced(count, limit):
    """Up to `limit` indices of `count`, evenly spaced, the first and the last included."""
    if count <= limit:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count-1, limit)})


def kept_tapes(planner, snapshot, tapes):
    """The robot's paths under the tapes and the indices of those that keep off the walls."""
    from go2nav.planning.sampling import FLOOR_M, integrate
    from go2nav.sim import kinematic as kin
    xy, yaw = integrate(snapshot, tapes)
    walls = planner.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1)
    return xy, yaw, walls, np.flatnonzero(walls.min(1) >= FLOOR_M)


def truth_at(world, planner, snapshot, target, tapes):
    """At a moment: the kept tapes, whether a walker truly comes within the floor under each, and each one's cost and
    terms under the truth (exact copies of the world stepped under the tape, the walkers' dice included)."""
    from go2nav.eval import contact_calibration as cc
    xy, yaw, walls, kept = kept_tapes(planner, snapshot, tapes)
    frames, n = tapes.shape[1], len(world.walkers)
    truth = np.zeros((len(kept), frames, n, 2))
    for i, k in enumerate(kept):
        branch = copy.deepcopy(world)
        for f in range(frames):
            branch.step(tapes[k, f])
            truth[i, f] = branch.walkers_xy()
    if not len(kept):
        return kept, np.zeros(0, bool), np.zeros(0), np.zeros((4, 0))
    truly = cc.chance_within(snapshot, xy[kept], yaw[kept], truth, recover=planner.recover) > .5
    cost, terms = planner.expected_costs(snapshot, target, xy[kept], yaw[kept], walls[kept], truth,
                                         np.ones((len(kept), n)), np.arange(n), ends=truth[:, -1])
    return kept, truly, cost, cc.split(cost, terms)


def build_mission(out, source, key, per_mission=PER_MISSION, near=NEAR_M, frames=None):
    """One source mission: run it, keep its moments, replay it for the truth at each; returns the row."""
    from go2nav.eval.contact_calibration import kind_of
    from go2nav.eval.kinematic_missions import FRAMES, run_mission
    from go2nav.planning.sampling import Planner
    model, kind, _, seed = SOURCES[source] if isinstance(source, str) else source
    set_name, maze, episode, walkers = key
    recorder = Recorder(near)
    row = run_mission(set_name, maze, episode, walkers=walkers, kind=kind, horizon_s=HORIZON_S, terminal=TERMINAL,
                      seed=seed, model=model, frames=frames or FRAMES, probe=recorder)
    moments = [recorder.eligible[i] for i in spaced(len(recorder.eligible), per_mission)]
    steps = np.array(recorder.steps, np.float32).reshape(-1, 3)
    world = mission_world(set_name, maze, episode, walkers, kind)
    planner = Planner(world.geometry, horizon_s=HORIZON_S, terminal=TERMINAL, seed=seed, model='true')
    rows, arrays = [], {'steps': steps}
    for j, m in enumerate(moments):
        while world.frame < m['frame']:
            world.step(steps[world.frame])
        if not (np.allclose(world.xy, m['xy'], atol=1e-9) and abs(world.yaw-m['yaw']) < 1e-9):
            raise RuntimeError(f'the replay left the mission at frame {m["frame"]}')
        snapshot = world.snapshot()
        kept, truly, cost, terms = truth_at(world, planner, snapshot, m['target'], m['tapes'])
        arrays[f'tapes_{j}'] = m['tapes']
        rows.append(dict(frame=m['frame'], target=m['target'].tolist(), nearest_m=round(m['nearest_m'], 3),
                         source_chosen=m['chosen'], names=m['names'], kinds=[kind_of(m['names'][k]) for k in kept],
                         kept=kept.tolist(), truth=truly.tolist(), true_cost=np.round(cost, 3).tolist(),
                         true_terms=np.round(terms, 3).tolist()))
    if out is not None:
        folder = Path(out)/(source if isinstance(source, str) else 'custom')
        folder.mkdir(parents=True, exist_ok=True)
        name = f'{maze:02d}_{episode}_{walkers}'
        np.savez_compressed(folder/f'{name}.npz', **arrays)
        partial = folder/f'{name}.tmp'
        partial.write_text(json.dumps(dict(row, source=source if isinstance(source, str) else None, key=list(key),
                                           eligible=len(recorder.eligible), moments=rows))+'\n')
        partial.rename(folder/f'{name}.json')
    return dict(row, eligible=len(recorder.eligible), moments=rows), arrays


def _build_job(args):
    out, source, key, per_mission, near = args
    target = Path(out)/source/f'{key[1]:02d}_{key[2]}_{key[3]}.json'
    if not target.exists():
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        build_mission(out, source, key, per_mission, near)
    row = json.loads(target.read_text())
    return source, row['exit'], len(row['moments'])


def build(out, sources=tuple(SOURCES), per_mission=PER_MISSION, near=NEAR_M, workers=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(sources={s: SOURCES[s] for s in sources}, per_mission=per_mission, near=near, horizon_s=HORIZON_S,
                  terminal=TERMINAL, set=SET, mazes=list(MAZES), episodes=list(EPISODES), rule=rule())
    if (out/'config.json').exists():
        before = json.loads((out/'config.json').read_text())
        if before['rule'] != config['rule']:
            raise RuntimeError(f'the bank was built under {before["rule"]}, not {config["rule"]}')
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    jobs = [(str(out), s, (SET, m, e, n), per_mission, near) for s in sources for n in SOURCES[s][2] for m in MAZES
            for e in EPISODES]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        done = list(pool.map(_build_job, jobs))
    report = {s: dict(missions=sum(d[0] == s for d in done), round_trips=sum(d[0] == s and d[1] == 'round_trip'
                                                                              for d in done),
                      moments=sum(d[2] for d in done if d[0] == s)) for s in sources}
    report['all'] = dict(missions=len(done), moments=sum(d[2] for d in done))
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def score_mission(json_file, model, recover=None):
    """A model's choice at each of a bank mission's moments (rows for `contact_calibration.summary`)."""
    from go2nav.eval import contact_calibration as cc
    from go2nav.planning.sampling import Planner
    from go2nav.sim.kinematic import FRAME_S
    json_file = Path(json_file)
    meta = json.loads(json_file.read_text())
    arrays = np.load(json_file.with_suffix('.npz'))
    steps, (set_name, maze, episode, walkers) = arrays['steps'], meta['key']
    _, kind, _, seed = SOURCES[meta['source']]
    world = mission_world(set_name, maze, episode, walkers, kind)
    planner = Planner(world.geometry, horizon_s=HORIZON_S, terminal=TERMINAL, seed=seed, model=model)
    recover = planner.recover if recover is None else recover
    captured, walkers_of = {}, planner.model.walkers

    def recorded_walkers(*args, **kwargs):
        out = walkers_of(*args, **kwargs)
        weights, groups = getattr(planner.model, 'weights', None), getattr(planner.model, 'groups', None)
        captured.update(walkers=out, weights=None if weights is None else np.array(weights, float),
                        groups=None if groups is None else np.array(groups))
        return out
    planner.model.walkers = recorded_walkers
    observes, records = hasattr(planner.model, 'observe'), hasattr(planner.model, 'record')
    at = {m['frame']: (j, m) for j, m in enumerate(meta['moments'])}
    rows, last = [], max(at, default=-1)
    for frame in range(last+1):
        if observes:
            planner.model.observe(frame*FRAME_S, world.scan(), world.xy, world.yaw, world.geometry)
        if frame in at:
            j, m = at[frame]
            tapes, snapshot, target = arrays[f'tapes_{j}'], world.snapshot(), np.asarray(m['target'], float)
            total, terms = planner.costs(snapshot, target, tapes)
            k = int(np.argmin(total))
            xy, yaw, _, kept = kept_tapes(planner, snapshot, tapes)
            if kept.tolist() != m['kept']:
                raise RuntimeError(f'{json_file.name} frame {frame}: the kept tapes differ from the bank\'s')
            row = dict(source=meta['source'], mission=json_file.stem, frame=frame, nearest_m=m['nearest_m'],
                       agrees=k == m['source_chosen'], chosen=int(np.flatnonzero(kept == k)[0]) if k in kept else -1,
                       chosen_kind=cc.kind_of(m['names'][k]), kinds=m['kinds'], truth=m['truth'],
                       true_cost=m['true_cost'], true_terms=m['true_terms'])
            if len(kept):
                weights = captured.get('weights')
                row.update(chance=np.round(cc.chance_within(snapshot, xy[kept], yaw[kept], captured['walkers'][kept],
                                                            None if weights is None else weights[kept],
                                                            captured.get('groups'), recover), 4).tolist(),
                           cost=np.round(total[kept], 3).tolist(),
                           terms=np.round(cc.split(total, terms)[:, kept], 3).tolist())
                rows.append(row)
        if records:
            planner.model.record(steps[frame])
        world.step(steps[frame])
    return rows


def _score_job(args):
    json_file, model = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    return score_mission(json_file, model)


def summarise(rows):
    """`contact_calibration.summary` over moments (not missions), with the agreement with the source's choices."""
    from go2nav.eval import contact_calibration as cc
    out = cc.summary([dict(round_trip=False, exit=None, probes=rows)])
    for key in ('missions', 'round_trips', 'contacts'):
        out.pop(key)
    return dict(moments=len(rows), agrees=round(float(np.mean([r['agrees'] for r in rows])), 3) if rows else None,
                off_walls=sum(r['chosen'] < 0 for r in rows), **out)


def score(bank, model, out, workers=None):
    bank = Path(bank)
    config = json.loads((bank/'config.json').read_text())
    if config['rule'] != rule():
        raise RuntimeError(f'the bank was built under {config["rule"]}; this environment sets {rule()}')
    files = sorted(f for s in config['sources'] for f in (bank/s).glob('[0-9]*.json'))
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = [r for part in pool.map(_score_job, [(str(f), model) for f in files]) for r in part]
    report = dict(model=model, bank=str(bank), rule=rule(), all=summarise(rows),
                  sources={s: summarise([r for r in rows if r['source'] == s]) for s in config['sources']})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report, rows=rows))+'\n')
    return report


HEADLINE = ('moments', 'agrees', 'mean_regret_s', 'wrong_share', 'causes', 'chosen_truly_within', 'auc', 'brier')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    b.add_argument('--per-mission', type=int, default=PER_MISSION)
    b.add_argument('--near', type=float, default=NEAR_M)
    b.add_argument('--workers', type=int)
    s = sub.add_parser('score')
    s.add_argument('--bank', required=True, type=Path)
    s.add_argument('--model', required=True)
    s.add_argument('--out', required=True, type=Path)
    s.add_argument('--workers', type=int)
    m = sub.add_parser('summary')
    m.add_argument('files', type=Path, nargs='+')
    a = p.parse_args()
    if a.command == 'build':
        print(json.dumps(build(a.out, a.sources, a.per_mission, a.near, a.workers), indent=1))
    elif a.command == 'score':
        report = score(a.bank, a.model, a.out, a.workers)
        print(json.dumps({k: report['all'].get(k) for k in HEADLINE}))
    else:
        for f in a.files:
            report = json.loads(f.read_text())
            print(f.stem, json.dumps({k: report['all'].get(k) for k in HEADLINE}))


if __name__ == '__main__':
    main()
