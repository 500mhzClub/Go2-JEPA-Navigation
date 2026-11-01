"""Per-frame mover labels for training-only supervision (docs/dynamic/PLAN.md, P4d), from corpus v2's records.

    python -m go2nav.data.mover_labels ROOT [--roles lewm_train lewm_val] [--workers N]

Per unit, `mover_columns.npy` (frames, COLUMNS, 3) float16 beside its frames: for each image column of the grid
world model (the primary camera's field of view in `COLUMNS` equal columns), whether a walker is in view there
(`mover_probe.in_view`: within 5 m, inside the field of view, centre not hidden by the walls) and the nearest such
walker's position (x forward, y left, m) in the body frame. A walker's column is its centre's bearing from the camera
projected through a pinhole of the camera's field of view (the nearest edge column for a walker whose centre is out
of the image but whose body is in view). Frames with fewer than `MIN_OBSTACLE_PIXELS` obstacle depth
pixels have no walker in view. The labels are never a model input; a unit already labelled is skipped (resumable).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.data import lewm_corpus as v1
from go2nav.data import lewm_corpus_v2 as v2
from go2nav.eval import mover_probe as mp

COLUMNS = 4
FILE = 'mover_columns.npy'


def column(x, y):
    """The image column (0 = left) of a walker centred at (x forward, y left) from the camera. A centre outside the
    image (a walker close by, only its body in view, as `mover_probe.in_view` allows) takes the nearest edge column."""
    if x <= 0:
        return 0 if y > 0 else COLUMNS-1
    u = .5-(y/x)/(2*math.tan(mp.HALF_FOV_RAD))     # 0 at the left edge, 1 at the right
    return min(max(int(math.floor(u*COLUMNS)), 0), COLUMNS-1)


def unit_columns(args):
    """Label one unit's frames (see the module docstring); returns (folder, frames labelled, frames with a walker)."""
    root, folder = Path(args[0]), Path(args[1])
    out = folder/FILE
    if out.exists():
        return str(folder), None, None
    meta = json.loads((folder/'meta.json').read_text())
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose']
    labels = np.zeros((len(pose), COLUMNS, 3), np.float16)
    if meta['obstacles']['placement']:
        with np.load(folder/'dynamic.npz') as dynamic:
            movers, pixels = dynamic['obstacles_xy'].astype(float), dynamic['obstacle_pixels']
        sight = mp.sight_line(v2.FreeDriverV2(mp.unit_spec(root, meta), np.random.default_rng(0)).wall_clearance)
        for t in np.flatnonzero(pixels >= v2.MIN_OBSTACLE_PIXELS):
            for _, k in mp.in_view(pose[t], movers[t], sight):              # nearest first
                x, y = mp.body_frame(pose[t], movers[t, k])
                c = column(x-mp.CAMERA_X_M, y)
                if not labels[t, c, 0]:
                    labels[t, c] = (1., x, y)
    tmp = folder/(FILE+'.tmp.npy')
    np.save(tmp, labels)
    os.replace(tmp, out)
    return str(folder), len(labels), int(labels[:, :, 0].any(1).sum())


def label(root, *, roles=('lewm_train', 'lewm_val'), workers=None):
    root = Path(root)
    jobs = [(str(root), str(f)) for f in v1.unit_folders(root, list(roles))]
    frames = seen = done = 0
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2)-2),
                             mp_context=multiprocessing.get_context('forkserver')) as pool:
        for i, (_, n, k) in enumerate(pool.map(unit_columns, jobs, chunksize=1)):
            if n is not None:
                frames, seen, done = frames+n, seen+k, done+1
            if (i+1) % 200 == 0:
                print(json.dumps(dict(units=i+1, of=len(jobs), labelled=done, frames=frames, with_walker=seen)), flush=True)
    return dict(units=len(jobs), labelled=done, frames=frames, with_walker=seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root')
    parser.add_argument('--roles', nargs='+', default=['lewm_train', 'lewm_val'])
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    print(json.dumps(label(args.root, roles=args.roles, workers=args.workers)))


if __name__ == '__main__':
    main()
