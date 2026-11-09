"""Range scans for corpus v2, cast offline (docs/dynamic/PLAN.md, P14 preparation, approved 8 November).

    python -m go2nav.data.range_backfill run CORPUS OUT [--workers N] [--limit N]
    python -m go2nav.data.range_backfill check CORPUS OUT [--units N]

Each unit's frames get the scan the range sensor (`go2nav.sim.range_sensor`) would have given at that frame. It uses:
- the recorded base pose (`steps.npz` `pose`: x, y, z and the quaternion in the corpus's x, y, z, w order);
- the walkers' recorded positions (`dynamic.npz` `obstacles_xy`) and heights (`meta.json`'s placement);
- the maze's walls (the unit's set, `CORPUS/mazes`).

The noise is seeded by layout and frame as in a mission. Writes `OUT/<unit path>/range.npz` (`ranges_m` float16
(frames, 720), `valid` bool (frames, 720)), atomically and resumably (a unit already written is skipped), and never
writes into `CORPUS`.

`check` compares the scans with the units' own truth labels (`free_rays.npy`: the free distance from the body origin
along 16 directions, capped at 1.5 m). The mount lies on the body's forward axis, so straight ahead the free distance
is the range plus the mount's offset, and straight behind the range minus it. Frames where the ring meets the floor first
are left out.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.mazes.sets import MazeSet
from go2nav.perception.range import AZIMUTHS, MOUNT_BODY_M
from go2nav.sim.range_sensor import RangeSensor

FILE = 'range.npz'
FREE_RAYS, FREE_CAP_M = 16, 1.5


def units(corpus):
    return json.loads((Path(corpus)/'index_12.json').read_text())['units']


def unit_scans(corpus, unit, specs=None):
    """(ranges_m float16 (n, 720), valid bool (n, 720)) of one unit's frames."""
    corpus, folder = Path(corpus), Path(corpus)/unit['path']
    meta = json.loads((folder/'meta.json').read_text())
    key = (meta['set'], meta['maze'])
    if specs is None or key not in specs:
        spec = MazeSet(meta['set'], root=corpus/'mazes').maze(meta['maze'])
        if specs is not None:
            specs[key] = spec
    else:
        spec = specs[key]
    sensor = RangeSensor(spec)
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose'].astype(float)
    with np.load(folder/'dynamic.npz') as dynamic:
        walkers = dynamic['obstacles_xy'].astype(float)
    placement = (meta.get('obstacles') or {}).get('placement') or []
    heights = [float(o['height_m']) for o in placement]
    n = len(pose)
    ranges, valid = np.zeros((n, len(AZIMUTHS)), np.float16), np.zeros((n, len(AZIMUTHS)), bool)
    for k in range(n):
        x, y, z, qx, qy, qz, qw = pose[k]
        xy = walkers[k] if walkers.shape[1] else np.zeros((0, 2))
        r, v = sensor.scan((x, y, z), (qw, qx, qy, qz), xy, heights[:len(xy)], frame=k)
        ranges[k], valid[k] = r.astype(np.float16), v
    return ranges, valid


def backfill_unit(job):
    corpus, out, unit = job
    target = Path(out)/unit['path']/FILE
    if target.exists():
        return unit['path'], 'kept'
    ranges, valid = unit_scans(corpus, unit)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(FILE+'.partial.npz')
    np.savez_compressed(partial, ranges_m=ranges, valid=valid)
    os.replace(partial, target)
    return unit['path'], 'written'


def backfill(corpus, out, workers, limit=None):
    jobs = [(str(corpus), str(out), u) for u in units(corpus)][:limit]
    counts = dict(written=0, kept=0)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('forkserver')) as pool:
        for i, (path, status) in enumerate(pool.map(backfill_unit, jobs, chunksize=4)):
            counts[status] += 1
            if i % 200 == 0:
                print(f'{i+1}/{len(jobs)} {path} {status}', flush=True)
    (Path(out)/'backfill.json').write_text(json.dumps(dict(corpus=str(corpus), units=len(jobs), **counts), indent=1)+'\n')
    print(json.dumps(counts))


def check(corpus, out, count):
    """Median and 95th-percentile disagreement (m) with the truth labels ahead and behind, over the units checked."""
    forward, behind = int(np.argmin(np.abs(AZIMUTHS))), 0
    errors = dict(ahead=[], behind=[])
    chosen = [u for u in units(corpus) if (Path(out)/u['path']/FILE).exists()][:count]
    for u in chosen:
        with np.load(Path(out)/u['path']/FILE) as scans:
            ranges, valid = scans['ranges_m'].astype(float), scans['valid']
        free = np.load(Path(corpus)/u['path']/'free_rays.npy').astype(float)
        with np.load(Path(corpus)/u['path']/'steps.npz') as steps:
            pose = steps['pose'].astype(float)
        qx, qy, qz, qw = pose[:, 3:7].T
        pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
        level = np.abs(pitch) < .03                               # the ring stays off the floor within reach
        for name, beam, label, offset in (('ahead', forward, 0, MOUNT_BODY_M[0]), ('behind', behind, FREE_RAYS//2, -MOUNT_BODY_M[0])):
            expected = free[:, label]
            measured = ranges[:, beam]+offset
            keep = level & valid[:, beam] & (expected < FREE_CAP_M-.05) & (measured < FREE_CAP_M-.05)
            errors[name] += list(np.abs(measured-expected)[keep])
    report = {k: dict(frames=len(v), median_m=round(float(np.median(v)), 4) if v else None,
                      p95_m=round(float(np.percentile(v, 95)), 4) if v else None) for k, v in errors.items()}
    print(json.dumps(dict(units=len(chosen), **report), indent=1))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='cast and write the scans')
    r.add_argument('corpus', type=Path)
    r.add_argument('out', type=Path)
    r.add_argument('--workers', type=int, default=16)
    r.add_argument('--limit', type=int, default=None)
    c = sub.add_parser('check', help='compare the scans with the free-ray labels')
    c.add_argument('corpus', type=Path)
    c.add_argument('out', type=Path)
    c.add_argument('--units', type=int, default=50)
    a = p.parse_args()
    if a.command == 'check':
        check(a.corpus, a.out, a.units)
    else:
        backfill(a.corpus, a.out, a.workers, a.limit)


if __name__ == '__main__':
    main()
