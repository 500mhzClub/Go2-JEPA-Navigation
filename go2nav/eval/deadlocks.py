"""Timeouts with walkers that do not yield: trapped, or a horizon problem? (STATUS, P22b's reason, 14 November)

A walker that does not yield waits for ever while the robot is within its reach ahead (`sim.obstacles`), and a passage
(1.3 m) leaves no room to pass one, so the robot can get by only from a junction, where it can step aside. For the
missions of a decision bank source (`eval.decision_bank`), replayed exactly from their recorded requests:
- `timeline`: every second, the walkers waiting on the robot and whether it is trapped (`trapped`: it cannot reach a
  junction, a cell of three or more passages, or its target without passing one of them); when the last stand-off
  began, whether the robot was trapped from its start, when it was last free in it, and the state at the end;
- `replan`: the same missions planned again with the source's model at other horizons, the rule as set.

    python -m go2nav.eval.deadlocks --bank BANK --source true_rude4 --horizons 12.8 25.6 --out FILE [--all]
"""
import argparse
import json
import os
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def trapped(world, target):
    """(walkers waiting on the robot, whether it is trapped) in the world's state, making for `target` (x, y)."""
    neighbours, pitch = world.geometry.neighbours, world.geometry.pitch
    near = lambda xy: min(neighbours, key=lambda c: float(np.linalg.norm(np.asarray(c, float)*pitch-xy)))
    waiting = [w for w in world.walkers if w.yields_to(world.xy)]
    cut = {frozenset((w.cell, w.target)) for w in waiting}
    start, goal = near(world.xy), near(np.asarray(target, float))
    seen, queue = {start}, deque([start])
    while queue:
        c = queue.popleft()
        for n in neighbours[c]:
            if n not in seen and frozenset((c, n)) not in cut:
                seen.add(n)
                queue.append(n)
    free = goal in seen or any(len(neighbours[c]) >= 3 for c in seen)
    return len(waiting), bool(waiting) and not free


def timeline(json_file):
    """A bank mission's stand-offs, second by second (see the module docstring)."""
    from go2nav.eval.decision_bank import SOURCES, mission_world
    from go2nav.eval.kinematic_missions import ARRIVAL_RADIUS_M as arrive_m
    from go2nav.eval.mission import episode_spec
    meta = json.loads(Path(json_file).read_text())
    steps = np.load(Path(json_file).with_suffix('.npz'))['steps']
    set_name, maze, episode, walkers = meta['key']
    world = mission_world(set_name, maze, episode, walkers, SOURCES[meta['source']][1])
    _, record = episode_spec(set_name, maze, episode)
    home, beacon = np.asarray(record['home_se2_world'][:2], float), np.asarray(record['beacon_xy_world'], float)
    reached, seconds = False, []
    for frame, s in enumerate(steps):
        world.step(s)
        reached = reached or float(np.linalg.norm(world.xy-beacon)) < arrive_m
        if frame % 10 == 0:
            seconds.append((frame/10, *trapped(world, home if reached else beacon)))
    waiting = [n > 0 for _, n, _ in seconds]
    began = len(seconds)-next((i for i, w in enumerate(reversed(waiting)) if not w), len(seconds))
    out = dict(mission=Path(json_file).stem, exit=meta['exit'], beacon_reached=reached,
               waiting_at_end=seconds[-1][1], trapped_at_end=seconds[-1][2])
    if began < len(seconds):
        free = [i for i in range(began, len(seconds)) if not seconds[i][2]]
        out.update(stand_off_began_s=seconds[began][0], waiting_then=seconds[began][1],
                   trapped_from_start=seconds[began][2], last_free_s=seconds[free[-1]][0] if free else None)
    return out


def replan(job):
    """One bank mission run again at another horizon: (mission, horizon, exit, simulated seconds)."""
    key, model, kind, seed, horizon = job
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from go2nav.eval.kinematic_missions import run_mission
    set_name, maze, episode, walkers = key
    row = run_mission(set_name, maze, episode, walkers=walkers, kind=kind, horizon_s=horizon, terminal='walkers',
                      seed=seed, model=model)
    return f'{maze:02d}_{episode}_{walkers}', horizon, row['exit'], row['simulated_s']


def main():
    from go2nav.eval.decision_bank import SOURCES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', required=True, type=Path)
    p.add_argument('--source', default='true_rude4', choices=tuple(SOURCES))
    p.add_argument('--horizons', type=float, nargs='*', default=(12.8, 25.6))
    p.add_argument('--all', action='store_true', help='replan every mission, not only the timeouts')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workers', type=int, default=None)
    a = p.parse_args()
    model, kind, _, seed = SOURCES[a.source]
    files = sorted((a.bank/a.source).glob('[0-9]*.json'))
    rows = {f.stem: json.loads(f.read_text()) for f in files}
    timeouts = [f for f in files if rows[f.stem]['exit'] == 'timeout']
    chosen = files if a.all else timeouts
    jobs = sorted(((rows[f.stem]['key'], model, kind, seed, h) for f in chosen for h in a.horizons), key=lambda j: -j[4])
    with ProcessPoolExecutor(a.workers or os.cpu_count()) as pool:
        states = list(pool.map(timeline, timeouts))
        replans = list(pool.map(replan, jobs))
    report = dict(source=a.source, missions=len(files), timeouts=len(timeouts), states=states,
                  replans=[dict(mission=m, horizon_s=h, exit=e, simulated_s=s) for m, h, e, s in replans],
                  exits={f'{h:g}': dict(Counter(e for _, hh, e, _ in replans if hh == h)) for h in a.horizons})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(dict(timeouts=len(timeouts), trapped_from_start=sum(s.get('trapped_from_start', False)
                                                                           for s in states),
                          trapped_at_end=sum(s['trapped_at_end'] for s in states), exits=report['exits'])))


if __name__ == '__main__':
    main()
