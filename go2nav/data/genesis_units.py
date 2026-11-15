"""P24 stage 1 (docs/dynamic/PLAN.md): corpus v2's Genesis units in the ring corpus's format, so that P22's measures
(`eval.object_measures`) run on them unchanged.

    python -m go2nav.data.genesis_units build --corpus CORPUS --ranges RANGES --out OUT [--split lewm_val] [--workers N]

CORPUS holds corpus v2's units (`data.lewm_corpus_v2`: steps.npz, dynamic.npz and meta.json each) and RANGES their ring
scans cast in Genesis (`data.range_backfill`). Every unit with a scan is written, in sorted order, as OUT/val/<index>.npz
and .json:
- `ranges` (T, 720) float16, 0 where a beam gave no valid return;
- `pose` (T, 3): x, y and yaw (about the vertical, from the base's quaternion in x, y, z, w order);
- `requested` (T, 3): the frame's request (constant over its five 50-Hz steps);
- labels, for choosing what is scored and scoring it, never a model's input: `walkers_xy` (T, 4, 2), NaN where there
  is no walker; `waiting` (T, 4), the unit's own; `in_sight` (T, 4), the walker in the range sensor's line of sight from
  its mount, through the layout (`sim.kinematic.Geometry.visible`, as `data.ring_corpus` labels it);
- meta: the layout and its episode (`home_episode`: the unit's, or 0 for a unit started from a free pose; the walls are
  the same), the walkers' kind as the ring corpus names it (`polite`, or `rude` for walkers that do not yield), their
  count, and the unit's own kind (`unit_kind`: explore, free, mission, transfer) and name.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

MAX_WALKERS = 4


def yaw_of(quaternions):
    """Yaw (T,) of base quaternions (T, 4) in x, y, z, w order."""
    x, y, z, w = quaternions.T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def convert(args):
    unit, corpus, ranges, out, index, *options = args
    split = options[0] if options else 'val'  # preserve P24 callers
    from go2nav.eval.mission import episode_spec
    from go2nav.eval.object_measures import MAZES
    from go2nav.perception.ring_walkers import mount_of
    from go2nav.sim import kinematic as kin
    if split not in ('train', 'val'):
        raise ValueError('Target split must be train or val')
    target = Path(out)/split/f'{index:05d}.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if existing['unit'] != unit or existing['split'] != split:
            raise ValueError('Existing unit index belongs to a different source')
        return existing
    folder, meta = Path(corpus)/unit, json.loads((Path(corpus)/unit/'meta.json').read_text())
    # Later units recorded scans live; earlier units use the separately backfilled cache.
    scan_path = folder/'range.npz' if (folder/'range.npz').exists() else Path(ranges)/unit/'range.npz'
    steps, dynamic, scan = (np.load(folder/'steps.npz'), np.load(folder/'dynamic.npz'), np.load(scan_path))
    frames = int(meta['frames'])
    pose = np.concatenate((steps['pose'][:frames, :2], yaw_of(steps['pose'][:frames, 3:7])[:, None]), 1)
    obstacles = meta.get('obstacles') or {}
    count = int(obstacles.get('count', 0))
    episode = meta['episode'] if meta['episode'] is not None else 0
    spec, _ = episode_spec(meta['set'], meta['maze'], episode, root=MAZES)
    geometry = kin.Geometry(spec)
    walkers = np.full((frames, MAX_WALKERS, 2), np.nan, np.float32)
    waiting, sight = np.zeros((frames, MAX_WALKERS), bool), np.zeros((frames, MAX_WALKERS), bool)
    if count:
        walkers[:, :count] = dynamic['obstacles_xy'][:frames, :count]
        waiting[:, :count] = dynamic['waiting'][:frames, :count]
        for t in range(frames):
            origin = mount_of(pose[t, :2], pose[t, 2])
            sight[t, :count] = [geometry.visible(origin, w) for w in walkers[t, :count]]
    arrays = dict(ranges=np.where(scan['valid'][:frames], scan['ranges_m'][:frames], 0.).astype(np.float16),
                  pose=pose, requested=steps['requested'][:frames, 0].astype(np.float32), walkers_xy=walkers,
                  waiting=waiting, in_sight=sight)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), **arrays)
    row = dict(schema='go2nav_genesis_units.v1', index=index, split=split, set=meta['set'], maze=int(meta['maze']),
               home_episode=int(episode), walkers=count, kind=None if not count else
               ('polite' if obstacles.get('polite', True) else 'rude'), unit_kind=meta['kind'], unit=unit,
               frames=frames)
    partial = target.with_suffix('.tmp')
    partial.write_text(json.dumps(row)+'\n')
    partial.rename(target)
    return row


def build(corpus, ranges, out, split='lewm_val', workers=None, units=None):
    target_split = {'lewm_train': 'train', 'lewm_val': 'val'}[split]
    units = sorted(units) if units is not None else sorted(f'{split}/{p.name}' for p in (Path(ranges)/split).iterdir()
                                                           if (p/'range.npz').exists())
    if any(Path(u).parts[-2] != split for u in units):
        raise ValueError('Unit does not belong to the requested split')
    jobs = [(u, str(corpus), str(ranges), str(out), i, target_split) for i, u in enumerate(units)]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(convert, jobs, chunksize=4))
    summary = dict(units=len(rows), with_walkers=sum(r['walkers'] > 0 for r in rows),
                   kinds={k: sum(r['kind'] == k for r in rows) for k in ('polite', 'rude')},
                   unit_kinds={k: sum(r['unit_kind'] == k for r in rows) for k in sorted({r['unit_kind'] for r in rows})})
    (Path(out)/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    (Path(out)/f'{target_split}_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--corpus', required=True, type=Path)
    b.add_argument('--ranges', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--split', default='lewm_val')
    b.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(build(a.corpus, a.ranges, a.out, a.split, a.workers)))


if __name__ == '__main__':
    main()
