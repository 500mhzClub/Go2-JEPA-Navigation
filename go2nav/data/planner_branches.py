"""Branched windows from the states a planner reaches (docs/dynamic/PLAN.md P22 item 13), for training its model on
the moments and tapes it acts on.

    python -m go2nav.data.planner_branches build CORPUS --mazes-root ROOT --model SPEC [--missions 1000] [--val 100]
        [--tapes 6] [--spacing 20] [--limit 40] [--workers N]
    python -m go2nav.data.planner_branches summary CORPUS

**Why:** N4's contact chances were measured at the plans its own planner makes (`eval.contact_calibration`): on
four walkers that do not yield it over-costs the truly best tape in 21% of plans near walkers (the ring tracker 13%,
the true future with realistic errors 6%), and that false risk is what keeps it waiting until it is boxed in. Its
training windows come from P20's corpus (a short-sighted planner and random tapes) and from random tapes branched
from that corpus (`data.branches`): neither holds the states its own planner reaches, nor the planner's tapes.

**Missions:** a round trip (`planning.mission.RoundTripMission`, as `eval.kinematic_missions` runs it: 6.4 s, the
walkers' terminal distance, at most `FRAMES` frames) under the planner with model SPEC and the rule the environment
sets, on the layout and home of corpus episode i of the split (train layouts only for train, val for val: never an
evaluation layout), with `WALKERS` walkers of a kind drawn per mission (`KINDS`, `KIND_P`; fewer when they do not
fit), placed as missions place them, seeded from (`SEED`, split, i).

**Branch points:** plans at which a walker in the sensor's line of sight is within `data.branches.CLOSE_M` of the
robot (labels, used only to choose), from `HISTORY_FRAMES` on, at most one every `spacing` frames and `limit` a
mission. **Branches:** the plan's chosen tape and `tapes`-1 others drawn from its proposals that keep off the walls,
each run for the tape's 6.4 s on a copy of the world (the walkers' dice shared) with a copy of the ring tracker, its
slots and the walkers' memory (`data.branches.Observer`, on scans rounded to float16 as the corpus's are), exactly as
`data.branches` runs a random tape; a branch ends at a contact the same way.

**Per mission,** CORPUS/planner_branches/<split>/<i>.npz in `data.branches`' layout (so `training.object_jepa.Branches`
reads it with `folder='planner_branches'`; the layout is corpus episode i's), plus `chosen` (B,): the planner's
choice; and <i>.json: the mission's exit, its walkers and the branch count.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.data.branches import (CLOSE_M, FRAMES as BRANCH_FRAMES, HISTORY_FRAMES, LABEL_KEYS, OBJECT_KEYS, SEGMENT,
                                  Observer, _empty_frame, labels)

SEED = 2026111302
FOLDER = 'planner_branches'
WALKERS, KINDS, KIND_P = 4, ('rude', 'polite', 'reactive'), (.5, .25, .25)


def placement(spec, record, split, index):
    """The mission's walkers: `WALKERS` of a drawn kind, fewer when they do not fit; and the kind."""
    from go2nav.sim import obstacles as moving
    rng = np.random.default_rng([SEED, index, split == 'val'])
    kind, count = str(rng.choice(KINDS, p=KIND_P)), WALKERS
    while count:
        try:
            return moving.place_obstacles(spec, record, count, int(rng.integers(2**62)), polite=kind != 'rude',
                                          reactive=kind == 'reactive'), kind
        except ValueError:
            count -= 1
    return [], kind


def mission(corpus, split, index, mazes_root, model, tapes=6, spacing=20, limit=40, seed=0):
    """One mission's branches (a dict of arrays) and its record."""
    from go2nav.data import ring_corpus as rc
    from go2nav.eval.kinematic_missions import ARRIVAL_RADIUS_M, FRAMES
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.mission import RoundTripMission
    from go2nav.planning.sampling import FLOOR_M, Planner, integrate
    from go2nav.sim import kinematic as kin
    meta = json.loads((Path(corpus)/split/f'{index:05d}.json').read_text())
    spec, record = episode_spec(rc.SET, meta['maze'], meta['home_episode'], root=Path(mazes_root))
    walkers, kind = placement(spec, record, split, index)
    world = kin.World(spec, record['home_se2_world'], walkers)
    trip = RoundTripMission(record['beacon_xy_world'], record['home_se2_world'][:2], arrival_radius_m=ARRIVAL_RADIUS_M,
                            navigation_frames=FRAMES)
    planner = Planner(world.geometry, horizon_s=6.4, terminal='walkers', seed=seed, model=model)
    if planner.frames != BRANCH_FRAMES:
        raise ValueError(f'the planner looks {planner.frames} frames ahead; a branch is {BRANCH_FRAMES}')
    last = {}
    costs = planner.costs

    def recorded(snapshot, target, requested):
        total, terms = costs(snapshot, target, requested)
        last.update(requested=requested.copy(), total=np.array(total, float))
        return total, terms
    planner.costs = recorded
    observe = Observer(world.geometry, planner.fields)
    rng = np.random.default_rng([SEED, index, split == 'val', 1])
    out = {k: [] for k in ('t0', 'contact', 'hold', 'chosen', 'requested', 'pose', *OBJECT_KEYS, *LABEL_KEYS)}
    recent, commands = deque(maxlen=HISTORY_FRAMES+1), deque(maxlen=HISTORY_FRAMES)
    requested, queue, frame, points, previous = np.zeros(3, np.float32), [], 0, 0, -10**9
    expected = {}                       # frame: the pose the chosen branch had there (the mission runs its start)
    observes = hasattr(planner.model, 'observe')
    while True:
        state = trip.advance(world.xy, frame=frame, requested=requested)
        if state['terminal'] is not None:
            break
        scan = world.scan()
        if observes:
            planner.model.observe(frame*kin.FRAME_S, scan, world.xy, world.yaw, world.geometry)
        here = labels(world)
        recent.append(dict(pose=np.array([*world.xy, world.yaw], np.float32),
                           **observe(frame, scan[0].astype(np.float16).astype(np.float32), world.xy, world.yaw), **here))
        if state['hold_required']:
            requested, queue = np.zeros(3, np.float32), []
            expected.clear()
        else:
            if not queue:
                tape, _ = planner.plan(world.snapshot(), trip.target())
                queue = list(tape)
                distance = np.where(here['in_sight'], np.linalg.norm(np.nan_to_num(here['walkers_xy'], nan=1e3)-world.xy,
                                                                     axis=-1), np.inf)
                if (frame >= HISTORY_FRAMES and points < limit and frame-previous >= spacing and len(world.walkers)
                        and distance.min() <= CLOSE_M):
                    proposals, chosen = last['requested'], int(last['total'].argmin())
                    xy, yaw = integrate(dict(xy=world.xy, yaw=world.yaw, executed=world.executed), proposals)
                    clear = planner.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1).min(1) >= FLOOR_M
                    others = [k for k in np.flatnonzero(clear) if k != chosen]
                    picks = [chosen]+[int(k) for k in rng.choice(others, min(tapes-1, len(others)), replace=False)]
                    for k in picks:
                        branch(world, observe, recent, commands, proposals[k].astype(np.float32), frame, k == chosen, out)
                    expected = {frame+1+f: out['pose'][-len(picks)][HISTORY_FRAMES+1+f] for f in range(len(queue))}
                    points, previous = points+1, frame
            requested = queue.pop(0)
        if hasattr(planner.model, 'record'):
            planner.model.record(requested)
        commands.append(np.asarray(requested, np.float32))
        world.step(requested)
        if frame+1 in expected and not np.allclose(expected.pop(frame+1), [*world.xy, world.yaw], atol=1e-5):
            raise RuntimeError(f'{split} {index}: the chosen branch left the mission at frame {frame+1}')
        if world.contact is not None:
            break
        frame += 1
    round_trip = state['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    exit_ = 'round_trip' if round_trip else (f'contact_{world.contact}' if world.contact else 'timeout')
    shapes = dict(t0=(), contact=(), hold=(), chosen=(), requested=(SEGMENT-1, 3), pose=(SEGMENT, 3),
                  xy=(SEGMENT, 6, 2), seen=(SEGMENT, 6), id=(SEGMENT, 6), velocity=(SEGMENT, 6, 2),
                  memory=(SEGMENT, 6, 4), walkers_xy=(SEGMENT, 4, 2), waiting=(SEGMENT, 4), in_sight=(SEGMENT, 4))
    dtypes = dict(t0=np.int32, contact=np.int16, hold=bool, chosen=bool, requested=np.float32, pose=np.float32,
                  xy=np.float32, seen=bool, id=np.int32, velocity=np.float32, memory=np.float32, walkers_xy=np.float32,
                  waiting=bool, in_sight=bool)
    arrays = {k: (np.stack(v).astype(dtypes[k]) if v else np.zeros((0, *shapes[k]), dtypes[k])) for k, v in out.items()}
    row = dict(split=split, index=index, maze=meta['maze'], home_episode=meta['home_episode'], kind=kind,
               walkers=len(walkers), model=model, exit=exit_, simulated_s=round(world.frame*kin.FRAME_S, 1),
               points=points, branches=int(len(arrays['t0'])), contacts=int((arrays['contact'] >= 0).sum()),
               rule={k: v for k, v in os.environ.items() if k.startswith('GO2NAV_')})
    return arrays, row


def branch(world, observe, recent, commands, tape, t, chosen, out):
    """One branch from the world at frame t under `tape` (`data.branches.branch_episode`'s inner loop)."""
    branch_world, watcher = copy.deepcopy(world), observe.fork()
    frames, contact = list(recent), -1
    for f in range(len(tape)):
        branch_world.step(tape[f])
        here = np.array([*branch_world.xy, branch_world.yaw], np.float32)
        if contact < 0 and branch_world.contact is not None:
            contact = f+1
        if contact >= 0:
            frames.append(_empty_frame(frames[-1]['pose'] if contact < f+1 else here))
            continue
        scan = branch_world.scan()[0].astype(np.float16).astype(np.float32)
        frames.append(dict(pose=here, **watcher(t+f+1, scan, branch_world.xy, branch_world.yaw), **labels(branch_world)))
    requested = np.concatenate((np.array(commands, np.float32).reshape(-1, 3), tape)).astype(np.float32)
    if contact >= 0:
        requested[HISTORY_FRAMES+contact:] = 0.
    out['t0'].append(t), out['contact'].append(contact), out['hold'].append(not tape.any()), out['chosen'].append(chosen)
    out['requested'].append(requested)
    for k in ('pose', *OBJECT_KEYS, *LABEL_KEYS):
        out[k].append(np.stack([fr[k] for fr in frames]))


def _job(args):
    corpus, mazes_root, split, index, model, tapes, spacing, limit = args
    target = Path(corpus)/FOLDER/split/f'{index:05d}.npz'
    if target.exists():
        return json.loads(target.with_suffix('.json').read_text())
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    arrays, row = mission(corpus, split, index, mazes_root, model, tapes, spacing, limit)
    target.with_suffix('.json').write_text(json.dumps(row)+'\n')
    partial = target.with_name(target.stem+'.tmp.npz')
    np.savez_compressed(partial, **arrays)
    partial.rename(target)
    return row


def build(corpus, mazes_root, model, *, missions=1000, val=100, tapes=6, spacing=20, limit=40, workers=None):
    corpus = Path(corpus)
    for split in ('train', 'val'):
        (corpus/FOLDER/split).mkdir(parents=True, exist_ok=True)
    jobs = [(str(corpus), str(mazes_root), split, i, model, tapes, spacing, limit)
            for split, count in (('val', val), ('train', missions)) for i in range(count)]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        list(pool.map(_job, jobs))
    report = summary(corpus)
    (corpus/FOLDER/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def summary(corpus):
    report = {}
    for split in ('train', 'val'):
        rows = [json.loads(f.read_text()) for f in sorted((Path(corpus)/FOLDER/split).glob('[0-9]*.json'))]
        report[split] = dict(missions=len(rows), branches=sum(r['branches'] for r in rows),
                             contacts=sum(r['contacts'] for r in rows), points=sum(r['points'] for r in rows),
                             exits={e: sum(r['exit'] == e for r in rows) for e in sorted({r['exit'] for r in rows})},
                             kinds={k: sum(r['kind'] == k for r in rows) for k in KINDS})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('corpus', type=Path)
    b.add_argument('--mazes-root', required=True, type=Path)
    b.add_argument('--model', required=True)
    b.add_argument('--missions', type=int, default=1000)
    b.add_argument('--val', type=int, default=100)
    b.add_argument('--tapes', type=int, default=6)
    b.add_argument('--spacing', type=int, default=20)
    b.add_argument('--limit', type=int, default=40)
    b.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('corpus', type=Path)
    a = p.parse_args()
    report = (build(a.corpus, a.mazes_root, a.model, missions=a.missions, val=a.val, tapes=a.tapes, spacing=a.spacing,
                    limit=a.limit, workers=a.workers) if a.command == 'build' else summary(a.corpus))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
