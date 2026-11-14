"""Branched windows from P20's corpus (docs/dynamic/PLAN.md P22 item 11): the same moment of an episode under several
robot tapes, so that what the robot causes can be told from what a walker does anyway.

    python -m go2nav.data.branches build CORPUS --mazes-root ROOT [--tapes 6] [--spacing 10] [--workers N]
    python -m go2nav.data.branches summary CORPUS

**Why:** three plans in four of the corpus come from a planner that sees the walkers, so in it the robot's motion and
a walker's are correlated whichever causes which. The several-futures model learnt that a walker moves when the robot
does: with a polite walker held up by the robot, it gives the walker's giving up a probability near the truth under
the recorded tape and next to none under a hold (STATUS P22, 13 November).

**Branch points:** each frame from `HISTORY_FRAMES` on at which a walker in the sensor's line of sight is within
`CLOSE_M` of the robot (labels, used only to choose), at most one every `spacing` frames. **Branches:** the episode's
world is replayed exactly under its recorded requests (as `data.ring_corpus.replay_world`) and the ring tracker run on
its recorded scans (as `data.object_tracks`); at a branch point the world, the tracker, its slots and the walkers'
memory are copied and stepped `FRAMES` frames under a random tape (`data.ring_corpus.random_tape`: knots every 0.4 s
within the continuous block's bounds, kept off the walls; one in five a hold), `tapes` times. The copies share the
walkers' dice, so a point's branches differ only by what the robot did. A branch's scans are rounded to float16 as
the corpus's are. A branch ends at a contact: later frames hold no objects and the robot stands. The reaction
measure's tapes (hold, and back away at 0.2 m/s for 2 s) are not drawn as such.

**Per episode,** CORPUS/branches/<split>/<index>.npz, for B branches of `SEGMENT` frames (the `HISTORY_FRAMES`
recorded frames before the point, the point, then `FRAMES` branch frames; one training window of 3 + 16 steps of 0.4
s): `t0` (B,), `contact` (B,) (the branch frame of a contact, -1 for none), `hold` (B,) (the tape is a hold),
`requested` (B, SEGMENT-1, 3), `pose` (B, SEGMENT, 3), the tracker's `xy`, `seen`, `id`, `velocity` and `memory` as
in `data.object_tracks`, and labels for measures only: `walkers_xy` (B, SEGMENT, 4, 2; NaN where there is no walker),
`waiting` and `in_sight` (B, SEGMENT, 4).
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

SEED = 2026111301
CLOSE_M, HISTORY_FRAMES, FRAMES = 1.5, 8, 64
SEGMENT = HISTORY_FRAMES+1+FRAMES
SLOTS, MAX_WALKERS = 6, 4
OBJECT_KEYS = ('xy', 'seen', 'id', 'velocity', 'memory')
LABEL_KEYS = ('walkers_xy', 'waiting', 'in_sight')


def branch_points(pose, truth, sight, spacing):
    """Frames at which a walker in sight is within CLOSE_M, at most one every `spacing` frames."""
    distance = np.where(sight, np.linalg.norm(np.nan_to_num(truth, nan=1e3)-pose[:, None, :2], axis=-1), np.inf)
    close = distance.min(1) <= CLOSE_M if distance.shape[1] else np.zeros(len(pose), bool)
    close[:HISTORY_FRAMES] = False
    points, last = [], -10**9
    for t in np.flatnonzero(close):
        if t-last >= spacing:
            points.append(int(t))
            last = t
    return points


class Observer:
    """The ring tracker, its slots and the walkers' memory, as `data.object_tracks` runs them."""

    def __init__(self, geometry, fields):
        from go2nav.perception import ring_walkers as rw
        self.rw, self.geometry, self.fields = rw, geometry, fields
        self.tracker, self.slots, self.memory = rw.Tracker(), rw.Slots(SLOTS), rw.Memory(SLOTS)

    def __call__(self, t, ranges, xy, yaw):
        rw = self.rw
        found = rw.detect(ranges, ranges > 0, xy, yaw, self.fields.wall_clearance)
        self.tracker.update(t*.1, found, rw.mount_of(xy, yaw), self.geometry)
        position, seen, ids, _, velocity = self.slots.update(self.tracker, t)
        remembered = self.memory.update(position, seen, ids, velocity, xy)
        return dict(xy=position.astype(np.float32), seen=seen.copy(), id=ids.astype(np.int32),
                    velocity=velocity.astype(np.float32), memory=remembered.astype(np.float32))

    def fork(self):
        """A copy whose tracker, slots and memory go their own way (the layout is shared)."""
        twin = copy.copy(self)
        twin.tracker, twin.slots, twin.memory = copy.deepcopy((self.tracker, self.slots, self.memory))
        return twin


def labels(world):
    from go2nav.perception.ring_walkers import mount_of
    from go2nav.sim import kinematic as kin
    out = dict(walkers_xy=np.full((MAX_WALKERS, 2), np.nan, np.float32), waiting=np.zeros(MAX_WALKERS, bool),
               in_sight=np.zeros(MAX_WALKERS, bool))
    n = len(world.walkers)
    if n:
        origin = mount_of(world.xy, world.yaw)
        out['walkers_xy'][:n] = world.walkers_xy()
        out['waiting'][:n] = [kin.waiting_on(w, world.xy) for w in world.walkers]
        out['in_sight'][:n] = [world.geometry.visible(origin, xy) for xy in world.walkers_xy()]
    return out


def _empty_frame(pose):
    return dict(pose=pose, xy=np.full((SLOTS, 2), np.nan, np.float32), seen=np.zeros(SLOTS, bool),
                id=np.full(SLOTS, -1, np.int32), velocity=np.zeros((SLOTS, 2), np.float32),
                memory=np.zeros((SLOTS, 4), np.float32), walkers_xy=np.full((MAX_WALKERS, 2), np.nan, np.float32),
                waiting=np.zeros(MAX_WALKERS, bool), in_sight=np.zeros(MAX_WALKERS, bool))


def branch_episode(corpus, split, index, mazes_root, tapes=6, spacing=10):
    """One episode's branches (a dict of arrays, B possibly 0)."""
    from go2nav.data import ring_corpus as rc
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.fields import Fields
    from go2nav.sim import kinematic as kin
    corpus = Path(corpus)
    meta = json.loads((corpus/split/f'{index:05d}.json').read_text())
    n = meta['walkers']
    with np.load(corpus/split/f'{index:05d}.npz') as d:
        ranges, pose, requested = d['ranges'], d['pose'], d['requested']
        truth, sight = d['walkers_xy'][:, :n], d['in_sight'][:, :n]
    points = branch_points(pose, truth, sight, spacing) if n else []
    out = {k: [] for k in ('t0', 'contact', 'hold', 'requested', 'pose', *OBJECT_KEYS, *LABEL_KEYS)}
    if points:
        spec, record = episode_spec(rc.SET, meta['maze'], meta['home_episode'], root=Path(mazes_root))
        world = kin.World(spec, record['home_se2_world'], meta['placement'])
        fields = Fields(world.geometry)
        observe = Observer(world.geometry, fields)
        rng = np.random.default_rng([SEED, index, split == 'val'])
        wanted, recent = set(points), deque(maxlen=HISTORY_FRAMES+1)
        for t in range(points[-1]+1):
            if not np.allclose(pose[t], [*world.xy, world.yaw], atol=1e-4):
                raise RuntimeError(f'{split} {index}: the replay left the record at frame {t}')
            recent.append(dict(pose=np.array([*world.xy, world.yaw], np.float32),
                               **observe(t, ranges[t].astype(np.float32), world.xy, world.yaw), **labels(world)))
            if t in wanted:
                start = dict(xy=world.xy, yaw=world.yaw, executed=world.executed)
                for _ in range(tapes):
                    tape = rc.random_tape(rng, FRAMES, start, fields)
                    branch, watcher = copy.deepcopy(world), observe.fork()
                    frames, contact = list(recent), -1
                    for f in range(FRAMES):
                        branch.step(tape[f])
                        here = np.array([*branch.xy, branch.yaw], np.float32)
                        if contact < 0 and branch.contact is not None:
                            contact = f+1
                        if contact >= 0:
                            frames.append(_empty_frame(frames[-1]['pose'] if contact < f+1 else here))
                            continue
                        scan = branch.scan()[0].astype(np.float16).astype(np.float32)
                        frames.append(dict(pose=here, **watcher(t+f+1, scan, branch.xy, branch.yaw), **labels(branch)))
                    commands = np.concatenate((requested[t-HISTORY_FRAMES:t], tape)).astype(np.float32)
                    if contact >= 0:
                        commands[HISTORY_FRAMES+contact:] = 0.
                    out['t0'].append(t), out['contact'].append(contact), out['hold'].append(not tape.any())
                    out['requested'].append(commands)
                    for k in ('pose', *OBJECT_KEYS, *LABEL_KEYS):
                        out[k].append(np.stack([fr[k] for fr in frames]))
            world.step(requested[t])
    shapes = dict(t0=(), contact=(), hold=(), requested=(SEGMENT-1, 3), pose=(SEGMENT, 3), xy=(SEGMENT, SLOTS, 2),
                  seen=(SEGMENT, SLOTS), id=(SEGMENT, SLOTS), velocity=(SEGMENT, SLOTS, 2), memory=(SEGMENT, SLOTS, 4),
                  walkers_xy=(SEGMENT, MAX_WALKERS, 2), waiting=(SEGMENT, MAX_WALKERS), in_sight=(SEGMENT, MAX_WALKERS))
    dtypes = dict(t0=np.int32, contact=np.int16, hold=bool, requested=np.float32, pose=np.float32, xy=np.float32,
                  seen=bool, id=np.int32, velocity=np.float32, memory=np.float32, walkers_xy=np.float32, waiting=bool,
                  in_sight=bool)
    return {k: (np.stack(v).astype(dtypes[k]) if v else np.zeros((0, *shapes[k]), dtypes[k])) for k, v in out.items()}


def _job(args):
    corpus, mazes_root, split, index, tapes, spacing = args
    target = Path(corpus)/'branches'/split/f'{index:05d}.npz'
    if target.exists():
        with np.load(target) as done:
            return int(len(done['t0']))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    arrays = branch_episode(corpus, split, index, mazes_root, tapes, spacing)
    partial = target.with_name(target.stem+'.tmp.npz')
    np.savez_compressed(partial, **arrays)
    partial.rename(target)
    return int(len(arrays['t0']))


def build(corpus, mazes_root, tapes=6, spacing=10, workers=None):
    corpus = Path(corpus)
    for split in ('train', 'val'):
        (corpus/'branches'/split).mkdir(parents=True, exist_ok=True)
    jobs = [(str(corpus), str(mazes_root), split, int(p.stem), tapes, spacing) for split in ('val', 'train')
            for p in sorted((corpus/split).glob('*.json'))]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        counts = list(pool.map(_job, jobs, chunksize=2))
    report = summary(corpus)
    (corpus/'branches'/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def summary(corpus):
    report = {}
    for split in ('train', 'val'):
        branches = holds = contacts = 0
        for f in sorted((Path(corpus)/'branches'/split).glob('[0-9][0-9][0-9][0-9][0-9].npz')):
            with np.load(f) as d:
                branches, holds, contacts = branches+len(d['t0']), holds+int(d['hold'].sum()), contacts+int((d['contact'] >= 0).sum())
        report[split] = dict(branches=branches, holds=holds, contacts=contacts)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('corpus', type=Path)
    b.add_argument('--mazes-root', required=True, type=Path)
    b.add_argument('--tapes', type=int, default=6)
    b.add_argument('--spacing', type=int, default=10)
    b.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('corpus', type=Path)
    a = p.parse_args()
    report = build(a.corpus, a.mazes_root, a.tapes, a.spacing, a.workers) if a.command == 'build' else summary(a.corpus)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
