"""The ring tracker's objects over P20's corpus (docs/dynamic/PLAN.md P22 item 1), and the layouts' wall rasters.

    python -m go2nav.data.object_tracks build CORPUS --mazes-root ROOT [--workers N]

Each episode's scans are replayed through `perception.ring_walkers` frame by frame, as a mission runs it (detection
by the walls' distance, the 0.3-s velocity window, forgetting what is seen gone), from the scan, the pose and the
layout only. Tracks are held in `SLOTS` slots (`perception.ring_walkers.Slots`: a track keeps its slot; a new one
takes a free slot or the slot of the remembered track seen longest ago).

Per episode, CORPUS/objects/<split>/<index>.npz: `xy` (T, SLOTS, 2) float32 world positions (NaN where the slot is
empty), `seen` (T, SLOTS) bool (detected this frame, not only remembered), `id` (T, SLOTS) int32 (-1 empty), and
`velocity` (T, SLOTS, 2) float32, the tracker's own (m/s, world frame; zero for a remembered track), and `memory`
(T, SLOTS, 4) float32 (`perception.ring_walkers.Memory`: what each slot's walker has done over its track's life). Per
layout, CORPUS/objects/rasters/<maze>.npz: the walls' distance on a `RASTER_M` grid (`planning.wall_raster`).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

SLOTS = 6
SET = 'lewm_sets_v1'


def track_episode(npz, meta, mazes_root):
    from go2nav.eval.mission import episode_spec
    from go2nav.perception import ring_walkers as rw
    from go2nav.planning.fields import Fields
    from go2nav.sim import kinematic as kin
    spec, _ = episode_spec(SET, meta['maze'], meta['home_episode'], root=Path(mazes_root))
    geometry = kin.Geometry(spec)
    fields = Fields(geometry)
    with np.load(npz) as data:
        ranges, pose = data['ranges'].astype(np.float32), data['pose'].astype(np.float64)
    frames = len(pose)
    xy = np.full((frames, SLOTS, 2), np.nan, np.float32)
    velocity = np.zeros((frames, SLOTS, 2), np.float32)
    memory = np.zeros((frames, SLOTS, rw.Memory.FEATURES), np.float32)
    remembered = rw.Memory(SLOTS)
    seen = np.zeros((frames, SLOTS), bool)
    ids = np.full((frames, SLOTS), -1, np.int32)
    tracker, slots, dropped = rw.Tracker(), rw.Slots(SLOTS), 0
    for t in range(frames):
        position, yaw = pose[t, :2], float(pose[t, 2])
        found = rw.detect(ranges[t], ranges[t] > 0, position, yaw, fields.wall_clearance)
        tracker.update(t*kin.FRAME_S, found, rw.mount_of(position, yaw), geometry)
        xy[t], seen[t], ids[t], left_out, velocity[t] = slots.update(tracker, t)
        memory[t] = remembered.update(xy[t], seen[t], ids[t], velocity[t], position)
        dropped += left_out
    return dict(xy=xy, seen=seen, id=ids, velocity=velocity, memory=memory), dict(dropped=dropped, maze=meta['maze'],
                                                                                    spec=spec)


def _job(args):
    corpus, mazes_root, split, index = args
    from go2nav.planning.wall_raster import WallRaster
    out = Path(corpus)/'objects'
    target = out/split/f'{index:05d}.npz'
    meta = json.loads((Path(corpus)/split/f'{index:05d}.json').read_text())
    raster = out/'rasters'/f"{meta['maze']:04d}.npz"
    if target.exists() and raster.exists():
        with np.load(target) as done:
            if 'memory' in done.files:
                return 0
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    arrays, info = track_episode(Path(corpus)/split/f'{index:05d}.npz', meta, mazes_root)
    if not raster.exists():
        from go2nav.sim import kinematic as kin
        WallRaster.of(kin.Geometry(info['spec'])).save(raster)
    partial = target.with_name(target.stem+'.tmp.npz')
    np.savez_compressed(partial, **arrays)
    partial.rename(target)
    return info['dropped']


def build(corpus, mazes_root, workers=None):
    corpus = Path(corpus)
    for sub in ('train', 'val', 'rasters'):
        (corpus/'objects'/sub).mkdir(parents=True, exist_ok=True)
    jobs = [(str(corpus), str(mazes_root), split, int(p.stem)) for split in ('val', 'train')
            for p in sorted((corpus/split).glob('*.json'))]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        dropped = sum(pool.map(_job, jobs, chunksize=4))
    return dict(episodes=len(jobs), dropped_track_frames=int(dropped))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('corpus', type=Path)
    b.add_argument('--mazes-root', required=True, type=Path)
    b.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(build(a.corpus, a.mazes_root, a.workers)))


if __name__ == '__main__':
    main()
