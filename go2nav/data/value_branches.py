"""P22b item 8's branches (docs/dynamic/PLAN.md): at moments of round 0's missions, the world copied exactly and six of
the planner's candidate tapes run from it, each for the cut (3.2 s) and then under the frozen rule's own planner to
the end of the leg.

    python -m go2nav.data.value_branches build --data DATA0 --out OUT [--train 2000] [--val 300] [--workers N]
    python -m go2nav.data.value_branches pack OUT [--workers N]

**Moments** (`moments_of`): from a round-0 mission's kept plans (`data.value_missions`), two: one at random among those
with a walker in a slot within `NEAR_M` of the robot (if any), one at random among the others.

**The mission** is run again exactly as round 0 ran it (the same job, planner and dice: its exit and frames are
checked). At a chosen plan, before the planner plans, everything is copied: the world (the walkers' dice included), the
mission, the planner with the tracker's tracks, and the slots' memory. The layout's distance fields are shared.

**The tapes:** a planner at the cut's horizon on a copy of the model proposes and costs its tapes under the frozen
rule; six are run (`TAPES`): its choice, the hold, the plain way to the target, and random ones it predicts clear of
the walls and the walkers. **A branch:** the tape for the cut (`CUT_S`), then round 0's planner (copied, not exploring)
replanning as missions do, until the leg's target is reached or the mission ends.

**Kept**, per moment (OUT/<split>/<index>_<moment>.npz and .json) and per tape:
- as the planner predicts the tape at the moment (for the ranking measure): the frozen rule's cost of the tape at the
  cut's horizon (`rule`), its chance of no breach (`clean`), its distance term at the cut (`terminal`), and the value's
  inputs at its predicted end (`planning.value.cut_inputs`; the slots' memory and presence are the moment's);
- the cut as it truly came (for training): the robot's pose, the slots (`planning.value.WalkerState`), the distance
  term there (`planning.value.distance_term`), the walking distance, and whether the cut came before the leg ended;
- the cost to go realised from the cut and from the moment: the seconds until the leg's target was reached, or until
  the mission failed plus the frozen rule's walker price.

`pack` gathers each split's tapes into OUT/packed/<split>.npz with each moment's index and grid, and the grids into
OUT/packed/grids.npz, as `data.value_missions.pack` does.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

CUT_S, TAPES, NEAR_M = 3.2, 6, 3.
MOMENT_KEYS = ('rule', 'clean', 'terminal', 'pred_xy', 'pred_yaw', 'pred_own', 'pred_slots_xy', 'pred_slots_v',
               'pred_memory', 'pred_present', 'cut_reached', 'cut_xy', 'cut_yaw', 'cut_slots_xy', 'cut_slots_v',
               'cut_memory', 'cut_present', 'cut_base', 'cut_own', 'cost_cut', 'cost_moment')


def moments_of(arrays, index):
    """Two plan frames of a round-0 mission: one near a walker (if any), one among the others."""
    rng = np.random.default_rng([index, 77])
    gap = np.linalg.norm(arrays['slots_xy']-arrays['xy'][:, None], axis=-1)
    near = ((gap < NEAR_M) & arrays['present']).any(1)
    chosen = []
    for pool in (np.flatnonzero(near), np.flatnonzero(~near)):
        if len(pool):
            chosen.append(int(arrays['frame'][rng.choice(pool)]))
    return sorted(set(chosen))


class Episode:
    """Round 0's mission (`data.value_missions.run`), a frame at a time, copyable."""

    def __init__(self, job, root):
        from go2nav.data import value_missions as vmi
        from go2nav.eval.kinematic_missions import ARRIVAL_RADIUS_M, FRAMES
        from go2nav.eval.mission import episode_spec
        from go2nav.planning.mission import RoundTripMission
        from go2nav.planning.value import WalkerState
        from go2nav.sim import obstacles as moving
        from go2nav.sim.kinematic import World
        spec, record = episode_spec(vmi.SET, job['maze'], job['episode'], root=Path(root))
        placement = moving.place_obstacles(spec, record, job['walkers'], job['placement_seed'],
                                           polite=job['kind'] != 'rude', reactive=job['kind'] == 'reactive')
        self.world = World(spec, record['home_se2_world'], placement)
        self.mission = RoundTripMission(record['beacon_xy_world'], record['home_se2_world'][:2],
                                        arrival_radius_m=ARRIVAL_RADIUS_M, navigation_frames=FRAMES)
        self.planner = vmi.exploring(self.world.geometry, horizon_s=6.4, terminal='walkers', seed=job['index'],
                                     model='ring_velocity', explore_seed=job['index'])
        self.walkers, self.frame, self.requested, self.queue = WalkerState(), 0, np.zeros(3, np.float32), []
        self.state, self.ended = None, None

    def copy(self):
        shared = {id(self.planner.fields): self.planner.fields, id(self.world.geometry): self.world.geometry}
        return copy.deepcopy(self, memo=shared)

    def begin(self):
        """Advance the mission and observe this frame; False once the mission has ended."""
        from go2nav.sim.kinematic import FRAME_S
        self.state = self.mission.advance(self.world.xy, frame=self.frame, requested=self.requested)
        if self.state['terminal'] is not None:
            self.ended = 'round_trip' if self.state['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE' else 'timeout'
            return False
        scan = self.world.scan()
        self.planner.model.observe(self.frame*FRAME_S, scan, self.world.xy, self.world.yaw, self.world.geometry)
        self.walkers.observe(self.frame*FRAME_S, scan, self.world.xy, self.world.yaw, self.world.geometry,
                             self.planner.fields.wall_clearance, tracker=self.planner.model.tracker)
        return True

    def planning(self):
        """Whether this frame takes a new plan (no hold, nothing queued)."""
        return not self.state['hold_required'] and not self.queue

    def finish(self, request=None):
        """Choose this frame's request (the given one, else the planner's), step; False on a contact."""
        if self.state['hold_required']:
            self.requested, self.queue = np.zeros(3, np.float32), []
        elif request is not None:
            self.requested, self.queue = np.asarray(request, np.float32), []
        else:
            if not self.queue:
                tape, _ = self.planner.plan(self.world.snapshot(), self.mission.target())
                self.queue = list(tape)
            self.requested = self.queue.pop(0)
        self.world.step(self.requested)
        if self.world.contact is not None:
            self.ended = f'contact_{self.world.contact}'
            return False
        self.frame += 1
        return True

    def leg(self):
        return int(self.mission.phase != 'OUTBOUND')

    def arrived(self, leg):
        phase = 'OUTBOUND' if leg == 0 else 'RETURN'
        return next((a['frame'] for a in self.mission.arrivals if a['phase'] == phase), None)


def candidates(episode):
    """At a planning frame: the frozen rule's tapes at the cut's horizon and their costs, on copies (the episode is not
    changed), and the `TAPES` to run; returns (tapes (K, T, 3), names, rule, clean, terminal, predicted inputs)."""
    from go2nav.planning.sampling import Planner, integrate
    from go2nav.planning.value import cut_inputs
    model = copy.deepcopy(episode.planner.model, memo={id(episode.planner.fields): episode.planner.fields})
    planner = Planner(episode.world.geometry, horizon_s=CUT_S, terminal='walkers', seed=episode.frame, model=model)
    planner.fields = episode.planner.fields
    model.bind(planner.fields)
    if episode.planner.previous is not None:              # the last plan, as long as this planner's tapes
        planner.previous, planner.target = episode.planner.previous[:planner.frames].copy(), \
            episode.planner.target.copy()
    seen, walkers_of = {}, model.walkers
    model.walkers = lambda *a, **k: seen.setdefault('w', walkers_of(*a, **k))
    snapshot, target = episode.world.snapshot(), np.asarray(episode.mission.target(), float)
    requested, names = planner.proposals(snapshot, target)
    total, terms = planner.costs(snapshot, target, requested)
    rng = np.random.default_rng([episode.frame, 5])
    chosen = [int(total.argmin()), names.index('hold'), next(i for i, n in enumerate(names) if n.startswith('target@'))]
    clear = [i for i in rng.permutation(len(names)) if not terms['breached'][i] and i not in chosen]
    chosen = list(dict.fromkeys(chosen+clear))[:TAPES]
    xy, yaw = integrate(snapshot, requested)
    now = episode.walkers.now
    slots_xy, slots_v, own = cut_inputs(now, seen['w'][chosen], xy[chosen, -1], episode.planner.fields, target)
    predicted = dict(pred_xy=xy[chosen, -1], pred_yaw=yaw[chosen, -1], pred_own=own, pred_slots_xy=slots_xy,
                     pred_slots_v=slots_v, pred_memory=np.tile(now['memory'], (len(chosen), 1, 1)),
                     pred_present=np.tile(now['present'], (len(chosen), 1)))
    clean = terms.get('clean', (~np.asarray(terms['breached'])).astype(float))   # no walker tracked: the usual rule
    terminal = terms.get('terminal', terms['progress'])
    return (requested[chosen], [names[i] for i in chosen], total[chosen], np.asarray(clean)[chosen],
            np.asarray(terminal)[chosen], predicted)


def branch(episode, tape, fail_s):
    """One tape from a copy of the episode: (cut state or None, cost from the cut, cost from the moment)."""
    from go2nav.data.value_missions import own_distance
    from go2nav.planning.value import distance_term
    from go2nav.sim.kinematic import FRAME_S
    e = episode.copy()
    e.planner.greedy = True
    start, leg, cut = e.frame, e.leg(), len(tape)
    status = 'cut'
    for i in range(cut+1):                        # the moment's frame is begun already; then each frame to the cut's
        if i > 0:
            if not e.begin():
                status = 'ended'
                break
            if e.arrived(leg) is not None:
                status = 'arrived'
                break
        if i == cut:
            break
        if not e.finish(tape[i]):
            status = 'ended'
            break
    state = None
    if status == 'cut':
        now, target = e.walkers.now, np.asarray(e.mission.target(), float)
        at = np.where(now['present'][:, None], now['xy'], np.nan)
        state = dict(cut_xy=e.world.xy.copy(), cut_yaw=float(e.world.yaw), cut_slots_xy=now['xy'],
                     cut_slots_v=now['velocity'], cut_memory=now['memory'], cut_present=now['present'],
                     cut_base=float(distance_term(e.planner.fields, target, e.world.xy[None], at[None])[0]),
                     cut_own=float(own_distance(e.planner.fields, target, e.world.xy[None])[0]))
        while e.finish() and e.begin() and e.arrived(leg) is None:
            pass
    reached = e.arrived(leg)
    end = reached if reached is not None else e.world.frame       # as `data.value_missions.cost_to_go`
    price = 0. if reached is not None else fail_s
    return state, (end-start-cut)*FRAME_S+price if state is not None else np.nan, (end-start)*FRAME_S+price


def run_moments(job, frames, root, fail_s):
    """Round 0's mission again, branched at the given plan frames: (exit, frames, [(frame, leg, names, arrays)])."""
    episode, out = Episode(job, root), []
    wanted = set(frames)
    while episode.begin():
        if episode.planning() and episode.frame in wanted:
            tapes, names, rule, clean, terminal, predicted = candidates(episode)
            rows = []
            for tape in tapes:
                rows.append(branch(episode, tape, fail_s))
            arrays = dict(predicted, rule=rule, clean=clean, terminal=terminal,
                          cut_reached=np.array([r[0] is not None for r in rows]),
                          cost_cut=np.array([r[1] for r in rows]), cost_moment=np.array([r[2] for r in rows]))
            blank = dict(cut_xy=np.full(2, np.nan), cut_yaw=np.nan, cut_slots_xy=np.zeros((6, 2)),
                         cut_slots_v=np.zeros((6, 2)), cut_memory=np.zeros((6, 4)), cut_present=np.zeros(6, bool),
                         cut_base=np.nan, cut_own=np.nan)
            for key in blank:
                arrays[key] = np.stack([np.asarray((r[0] or blank)[key]) for r in rows])
            out.append((episode.frame, episode.leg(), names, arrays))
        if not episode.finish():
            break
    return episode.ended, episode.world.frame, out


def _build(args):
    data, out, index, split = args
    folder = Path(out)/split
    done = folder/f'{index:05d}.done'
    if not done.exists():
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        from go2nav.data.value_missions import mazes_root
        meta = json.loads((Path(data)/split/f'{index:05d}.json').read_text())
        arrays = dict(np.load(Path(data)/split/f'{index:05d}.npz'))
        job = {k: meta[k] for k in ('index', 'split', 'maze', 'episode', 'walkers', 'kind', 'placement_seed')}
        frames = moments_of(arrays, index)
        ended, last, moments = run_moments(job, frames, mazes_root(), float(meta['rule']['GO2NAV_WALKER_PRICE']))
        if ended != meta['exit'] or last != meta['frames']:
            raise RuntimeError(f'mission {index} did not repeat: {ended} at {last}, round 0 {meta["exit"]} at '
                               f'{meta["frames"]}')
        folder.mkdir(parents=True, exist_ok=True)
        for m, (frame, leg, names, arr) in enumerate(moments):
            np.savez_compressed(folder/f'{index:05d}_{m}.npz', **arr)
            (folder/f'{index:05d}_{m}.json').write_text(json.dumps(dict(job, frame=frame, leg=leg, names=names))+'\n')
        done.write_text(json.dumps(dict(moments=len(moments), exit=ended))+'\n')
    return json.loads(done.read_text())['moments']


def build(data, out, train=2000, val=300, workers=None):
    from go2nav.data.value_missions import rule
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(data=str(data), train=train, val=val, cut_s=CUT_S, tapes=TAPES, near_m=NEAR_M, rule=rule())
    if json.loads((data/'config.json').read_text())['rule'] != config['rule']:
        raise RuntimeError('the environment does not set the rule round 0 was built under')
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    pick = lambda split, count: sorted(int(f.stem) for f in (data/split).glob('[0-9]*.json'))[:count]
    jobs = [(str(data), str(out), i, 'train') for i in pick('train', train)]+[(str(data), str(out), i, 'val')
                                                                             for i in pick('val', val)]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        moments = list(pool.map(_build, jobs, chunksize=2))
    summary = {s: dict(missions=sum(j[3] == s for j in jobs), moments=sum(m for j, m in zip(jobs, moments) if j[3] == s))
               for s in ('train', 'val')}
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def pack(out, workers=None):
    from go2nav.data.value_missions import _grids
    out = Path(out)
    (out/'packed').mkdir(exist_ok=True)
    keys, parts = {}, {}
    for split in ('train', 'val'):
        chunks = []
        for moment, f in enumerate(sorted((out/split).glob('[0-9]*_[0-9].json'))):
            meta, arrays = json.loads(f.read_text()), dict(np.load(f.with_suffix('.npz')))
            g = keys.setdefault((meta['maze'], meta['episode']), len(keys))
            count = len(arrays['rule'])
            arrays.update(moment=np.full(count, moment, np.int32), grid=np.full(count, g, np.int32),
                          leg=np.full(count, meta['leg'], np.int8), mission=np.full(count, meta['index'], np.int32),
                          hold=np.array([n == 'hold' for n in meta['names']]))
            chunks.append(arrays)
        parts[split] = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        parts[split] = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in parts[split].items()}
        np.savez(out/'packed'/f'{split}.npz', **parts[split])
    order = sorted(keys, key=keys.get)
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        made = list(pool.map(_grids, order, chunksize=8))
    np.savez(out/'packed'/'grids.npz', walls=np.stack([m[0] for m in made]), distance=np.stack([m[1] for m in made]),
             low=np.stack([m[2] for m in made]), keys=np.array(order, np.int32))
    return {s: dict(tapes=len(p['rule']), moments=int(p['moment'].max())+1) for s, p in parts.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--data', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--train', type=int, default=2000)
    b.add_argument('--val', type=int, default=300)
    b.add_argument('--workers', type=int)
    k = sub.add_parser('pack')
    k.add_argument('out', type=Path)
    k.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(build(a.data, a.out, a.train, a.val, a.workers) if a.command == 'build' else pack(a.out, a.workers)))


if __name__ == '__main__':
    main()
