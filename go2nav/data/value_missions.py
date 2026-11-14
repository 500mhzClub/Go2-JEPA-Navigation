"""P22b's data (docs/dynamic/PLAN.md): missions on the ring corpus's layouts under the frozen rule, each plan's state
with the cost to go that the mission then realised.

    python -m go2nav.data.value_missions build --out OUT [--missions 5000] [--val 500] [--workers N]
        [--value PATH --horizon 3.2]
    python -m go2nav.data.value_missions pack OUT [--workers N]

**A mission** (index i, `job`): a layout of `lewm_sets_v1` (training 0-1,199; the last `val` indices on validation
1,200-1,299, so the two share no layout and neither shares one with an evaluation set) and one of its two registered
episodes (home pose and beacon); 1-4 walkers all of one kind (not yielding, polite or reactive), placed as missions
place them; the round trip (`planning.mission.RoundTripMission`) within 480 s. The planner is P17's with the ring
tracker (`ring_velocity`) under the frozen rule, which the environment must set; 6.4 s and the walkers terminal, or
with `--value` P22b's value at `--horizon`. At one plan in `EXPLORE` it takes a random tape of those clear of the walls
(`Exploring.choose`), so that the data also hold states the planner would not choose.

**Kept**, OUT/<split>/<index>.npz and .json, at every plan: the frame, the leg (0 to the beacon, 1 home), the robot's
pose, the walkers' slots (`planning.value.WalkerState`: position, velocity, memory, present), the frozen rule's distance
term there (`planning.value.distance_term`, the walkers where the slots have them), the robot's walking distance to the
target, whether the plan explored, and the cost to go (`cost_to_go`): the seconds until the leg's target was reached,
or, if the mission failed first (a contact stop or the time limit), until then plus the frozen rule's walker price.

`pack` gathers each split into OUT/packed/<split>.npz with each moment's layout grid (`grid`), and the grids into
OUT/packed/grids.npz: for each (layout, episode) its walls' distance and the walking distances to its beacon and home
(`models.value.wall_grid`, `distance_grid`), on `planning.fields.Fields`'s nodes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

SET, TRAIN, VAL = 'lewm_sets_v1', (0, 1200), (1200, 1300)
KINDS, COUNTS, EXPLORE = ('rude', 'polite', 'reactive'), (1, 2, 3, 4), .1
RULE = ('GO2NAV_EXACT', 'GO2NAV_RECOVER', 'GO2NAV_WALKER_PRICE', 'GO2NAV_W_STILL')
ARRAYS = ('frame', 'leg', 'xy', 'yaw', 'slots_xy', 'slots_v', 'memory', 'present', 'base', 'own', 'explored', 'cost')


def mazes_root():
    from go2nav.eval.object_measures import MAZES
    return MAZES


def job(index, missions, val):
    rng = np.random.default_rng([index, 2022])
    split = 'val' if index >= missions-val else 'train'
    low, high = VAL if split == 'val' else TRAIN
    return dict(index=int(index), split=split, maze=int(rng.integers(low, high)), episode=int(rng.integers(2)),
                walkers=int(rng.choice(COUNTS)), kind=str(rng.choice(KINDS)), placement_seed=int(rng.integers(2**62)))


def rule():
    return {k: os.environ.get(k) for k in RULE}


def own_distance(fields, target, xy):
    """The robot's walking distance to the target at points (K, 2), closed ground at `models.value.FAR_M`."""
    from go2nav.models.value import FAR_M
    d = fields.value(fields.static(target), xy)
    return np.where(np.isfinite(d), d, FAR_M)


def exploring(*args, explore_seed, **kwargs):
    """P17's planner that, at one plan in `EXPLORE`, takes a random tape clear of the walls."""
    from go2nav.planning.sampling import DISCARD, Planner

    class Exploring(Planner):
        def choose(self, total, terms):
            self.explored = bool(self.explore_rng.random() < EXPLORE)
            clear = np.flatnonzero(total < DISCARD)
            if self.explored and len(clear):
                return int(self.explore_rng.choice(clear))
            self.explored = False
            return int(total.argmin())
    planner = Exploring(*args, **kwargs)
    planner.explore_rng, planner.explored = np.random.default_rng([explore_seed, 97]), False
    return planner


def cost_to_go(frames, legs, arrivals, end_frame, fail_s, frame_s=.1):
    """Seconds from each plan's frame until its leg's target was reached, or until the mission ended plus `fail_s` if
    the leg's target was never reached."""
    reached = {0: None, 1: None}
    for a in arrivals:
        reached[0 if a['phase'] == 'OUTBOUND' else 1] = a['frame']
    return np.array([(reached[g]-f)*frame_s if reached[g] is not None else (end_frame-f)*frame_s+fail_s
                     for f, g in zip(frames, legs)], float).reshape(-1)


def run(spec_job, horizon_s=6.4, value=None, frames=None, root=None):
    """One mission (`job`): (meta, arrays)."""
    from go2nav.eval.kinematic_missions import ARRIVAL_RADIUS_M, FRAMES
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.mission import RoundTripMission
    from go2nav.planning.value import WalkerState, distance_term
    from go2nav.sim import obstacles as moving
    from go2nav.sim.kinematic import FRAME_S, World
    spec, record = episode_spec(SET, spec_job['maze'], spec_job['episode'], root=Path(root or mazes_root()))
    placement = moving.place_obstacles(spec, record, spec_job['walkers'], spec_job['placement_seed'],
                                       polite=spec_job['kind'] != 'rude', reactive=spec_job['kind'] == 'reactive')
    world = World(spec, record['home_se2_world'], placement)
    mission = RoundTripMission(record['beacon_xy_world'], record['home_se2_world'][:2],
                               arrival_radius_m=ARRIVAL_RADIUS_M, navigation_frames=frames or FRAMES)
    planner = exploring(world.geometry, horizon_s=horizon_s, terminal='walkers', seed=spec_job['index'],
                        model='ring_velocity', value=value, explore_seed=spec_job['index'])
    if planner.walker_price is None or not planner.exact:
        raise RuntimeError(f'the frozen rule is not set: {rule()}')
    walkers, rows = WalkerState(), {k: [] for k in ARRAYS if k != 'cost'}
    requested, queue, frame = np.zeros(3, np.float32), [], 0
    while True:
        state = mission.advance(world.xy, frame=frame, requested=requested)
        if state['terminal'] is not None:
            break
        scan = world.scan()
        planner.model.observe(frame*FRAME_S, scan, world.xy, world.yaw, world.geometry)
        walkers.observe(frame*FRAME_S, scan, world.xy, world.yaw, world.geometry, planner.fields.wall_clearance,
                        tracker=planner.model.tracker)
        if planner.value is not None:
            planner.value.observe(frame*FRAME_S, scan, world.xy, world.yaw, world.geometry,
                                  tracker=planner.model.tracker)
        if state['hold_required']:
            requested, queue = np.zeros(3, np.float32), []
        else:
            if not queue:
                snapshot, target = world.snapshot(), mission.target()
                tape, _ = planner.plan(snapshot, target)
                now = walkers.now
                at = np.where(now['present'][:, None], now['xy'], np.nan)
                for key, item in (('frame', frame), ('leg', int(mission.phase != 'OUTBOUND')), ('xy', world.xy.copy()),
                                  ('yaw', float(world.yaw)), ('slots_xy', now['xy']), ('slots_v', now['velocity']),
                                  ('memory', now['memory']), ('present', now['present']),
                                  ('base', distance_term(planner.fields, target, world.xy[None], at[None])[0]),
                                  ('own', own_distance(planner.fields, target, world.xy[None])[0]),
                                  ('explored', planner.explored)):
                    rows[key].append(item)
                queue = list(tape)
            requested = queue.pop(0)
        world.step(requested)
        if world.contact is not None:
            break
        frame += 1
    round_trip = state['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    exit_ = 'round_trip' if round_trip else (f'contact_{world.contact}' if world.contact else 'timeout')
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays['cost'] = cost_to_go(arrays['frame'], arrays['leg'], mission.arrivals, world.frame, planner.walker_price)
    meta = dict(spec_job, set=SET, horizon_s=horizon_s, value=None if value is None else str(value), rule=rule(),
                exit=exit_, arrivals=[dict(phase=a['phase'], frame=int(a['frame'])) for a in mission.arrivals],
                frames=int(world.frame), plans=len(arrays['frame']), explored=int(arrays['explored'].sum()))
    return meta, arrays


def _build(args):
    out, spec_job, horizon_s, value = args
    folder = Path(out)/spec_job['split']
    target = folder/f'{spec_job["index"]:05d}.json'
    if not target.exists():
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        meta, arrays = run(spec_job, horizon_s, value)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f'{spec_job["index"]:05d}.npz', **arrays)
        partial = target.with_suffix('.tmp')
        partial.write_text(json.dumps(meta)+'\n')
        partial.rename(target)
    return json.loads(target.read_text())['exit']


def build(out, missions=5000, val=500, workers=None, horizon_s=6.4, value=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(set=SET, missions=missions, val=val, horizon_s=horizon_s, value=None if value is None else str(value),
                  rule=rule(), explore=EXPLORE, kinds=KINDS, counts=COUNTS)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise RuntimeError(f'{out} was built with {json.loads((out/"config.json").read_text())}, not {config}')
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    jobs = [(str(out), job(i, missions, val), horizon_s, None if value is None else str(value)) for i in range(missions)]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        exits = list(pool.map(_build, jobs, chunksize=4))
    summary = {s: {e: sum(1 for (_, j, _, _), x in zip(jobs, exits) if j['split'] == s and x == e) for e in set(exits)}
               for s in ('train', 'val')}
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def _grids(key):
    from go2nav.eval.mission import episode_spec
    from go2nav.models import value as vm
    from go2nav.planning.fields import Fields
    from go2nav.sim.kinematic import Geometry
    maze, episode = key
    spec, record = episode_spec(SET, maze, episode, root=Path(mazes_root()))
    fields = Fields(Geometry(spec))
    return (vm.wall_grid(fields), np.stack((vm.distance_grid(fields, record['beacon_xy_world']),
                                            vm.distance_grid(fields, record['home_se2_world'][:2]))),
            fields.low.astype(np.float32))


def pack(out, workers=None):
    out = Path(out)
    (out/'packed').mkdir(exist_ok=True)
    keys, parts = {}, {}
    for split in ('train', 'val'):
        metas = sorted((out/split).glob('[0-9]*.json'))
        chunks = []
        for f in metas:
            meta, arrays = json.loads(f.read_text()), dict(np.load(f.with_suffix('.npz')))
            g = keys.setdefault((meta['maze'], meta['episode']), len(keys))
            count = len(arrays['frame'])
            arrays.update(grid=np.full(count, g, np.int32), mission=np.full(count, meta['index'], np.int32),
                          walkers=np.full(count, meta['walkers'], np.int8),
                          kind=np.full(count, KINDS.index(meta['kind']), np.int8))
            chunks.append(arrays)
        parts[split] = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        parts[split] = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in parts[split].items()}
        np.savez(out/'packed'/f'{split}.npz', **parts[split])
    order = sorted(keys, key=keys.get)
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        made = list(pool.map(_grids, order, chunksize=8))
    shape = np.max([m[0].shape for m in made], 0)
    if any(m[0].shape != tuple(shape) for m in made):
        raise RuntimeError('the layouts\' grids differ in shape')
    np.savez(out/'packed'/'grids.npz', walls=np.stack([m[0] for m in made]), distance=np.stack([m[1] for m in made]),
             low=np.stack([m[2] for m in made]), keys=np.array(order, np.int32))
    return {s: len(p['frame']) for s, p in parts.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--missions', type=int, default=5000)
    b.add_argument('--val', type=int, default=500)
    b.add_argument('--workers', type=int)
    b.add_argument('--horizon', type=float, default=6.4)
    b.add_argument('--value', type=Path)
    k = sub.add_parser('pack')
    k.add_argument('out', type=Path)
    k.add_argument('--workers', type=int)
    a = p.parse_args()
    if a.command == 'build':
        print(json.dumps(build(a.out, a.missions, a.val, a.workers, a.horizon, a.value)))
    else:
        print(json.dumps(pack(a.out, a.workers)))


if __name__ == '__main__':
    main()
