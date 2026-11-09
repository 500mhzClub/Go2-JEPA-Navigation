"""Phase C's non-camera inputs per frame (docs/dynamic/PLAN.md P15), as one memory-mapped array per unit.

    python -m go2nav.data.observations CORPUS OUT [--range-root ROOT] [--workers N]

Writes OUT/units/<role>/<unit>/obs.npy, float16 (frames, `OBS_DIM`) (the corpus is never touched):
- `RANGE` (720): the range ring (`perception.range`: beams counter-clockwise from -pi in the body frame), each return
  as log(r / 0.05 m) / log(20 m / 0.05 m) in [0, 1]; an invalid beam (no return) reads 1;
- `MASK` (720): 1 where the beam returned;
- `PROPRIO` (51): what an IMU and the policy give the real robot: the body-frame angular velocity (3, rad/s), the
  projected gravity (3, unit vector in the body frame), and the applied command history (15 x 3, normalised by corpus
  v2's control statistics, as the action blocks are). The base's linear velocity is left out: hardware has no sensor
  for it.
The scans come from the unit's own `range.npz` (P14's units record them), else from the offline backfill
(`data.range_backfill`, `--range-root`).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.perception.range import BEAMS, MAX_RANGE_M, MIN_RANGE_M

RANGE, MASK = slice(0, BEAMS), slice(BEAMS, 2*BEAMS)
PROPRIO_DIM = 3+3+15*3
PROPRIO = slice(2*BEAMS, 2*BEAMS+PROPRIO_DIM)
OBS_DIM = 2*BEAMS+PROPRIO_DIM
FILE = 'obs.npy'


def rotate_into_body(quaternion, vectors):
    """World vectors (n, 3) into the body frame of quaternions (n, 4: x, y, z, w)."""
    x, y, z, w = (quaternion[:, i:i+1] for i in range(4))
    q = np.concatenate([-x, -y, -z], 1)             # the inverse rotation's vector part (w unchanged)
    t = 2*np.cross(q, vectors)
    return vectors+w*t+np.cross(q, t)


def unit_observations(folder, range_file):
    from go2nav.models.lewm import control_stats
    with np.load(range_file) as scan:
        ranges, valid = scan['ranges_m'].astype(np.float32), scan['valid']
    with np.load(folder/'steps.npz') as steps:
        pose, twist, history = steps['pose'], steps['twist'], steps['applied_history'].astype(np.float32)
    if len(ranges) != len(pose):
        raise ValueError(f'{folder}: {len(ranges)} scans for {len(pose)} frames')
    log = np.log(np.clip(ranges, MIN_RANGE_M, MAX_RANGE_M)/MIN_RANGE_M)/np.log(MAX_RANGE_M/MIN_RANGE_M)
    out = np.empty((len(pose), OBS_DIM), np.float32)
    out[:, RANGE] = np.where(valid, log, 1.)
    out[:, MASK] = valid
    q = pose[:, 3:7].astype(np.float64)
    gyro = rotate_into_body(q, twist[:, 3:6].astype(np.float64))
    gravity = rotate_into_body(q, np.tile([0., 0., -1.], (len(q), 1)))
    mean, std = control_stats(3)
    out[:, PROPRIO] = np.concatenate([gyro, gravity, ((history-mean)/std).reshape(len(pose), -1)], 1)
    return out.astype(np.float16)


def _job(args):
    root, out, range_root, path = (Path(a) if a else None for a in args)
    target = out/path/FILE
    if target.exists():
        return 0
    own = root/path/'range.npz'
    obs = unit_observations(root/path, own if own.exists() else range_root/path/'range.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name('obs.tmp.npy')
    np.save(partial, obs)
    partial.rename(target)
    return len(obs)


def build(root, out, *, range_root=None, workers=None):
    from go2nav.data import lewm_corpus as v1
    root, out = Path(root), Path(out)
    range_root = Path(range_root) if range_root else Path(str(root)+'_range')
    paths = [str(f.relative_to(root)) for f in v1.unit_folders(root, ['lewm_train', 'lewm_val'])]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        written = list(pool.map(_job, [(str(root), str(out), str(range_root), p) for p in paths], chunksize=4))
    return dict(units=len(paths), written=sum(w > 0 for w in written), frames=sum(written))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('corpus', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--range-root', type=Path)
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(build(a.corpus, a.out, range_root=a.range_root, workers=a.workers), indent=1))


if __name__ == '__main__':
    main()
