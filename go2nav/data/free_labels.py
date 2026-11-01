"""Per-frame free-space labels for training-only supervision (docs/dynamic/PLAN.md, P6), from corpus v2's records.

    python -m go2nav.data.free_labels ROOT [--roles lewm_train lewm_val] [--workers N]

Per unit, `free_rays.npy` (frames, RAYS) float16 beside its frames: along each of `RAYS` directions round the robot, evenly
spaced from straight ahead, anticlockwise, the free distance (m) from the body origin to the nearest wall or walker, capped
at `MAX_M`. Walls are the maze's boxes and a walker is its disk, from the simulator's records. The labels are never a model
input; a unit already labelled is skipped (resumable).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.data import lewm_corpus as v1
from go2nav.eval import mover_probe as mp
from go2nav.sim.obstacles import RADIUS_M

RAYS = 16
MAX_M = 1.5
FILE = 'free_rays.npy'


def wall_edges(walls):
    """Each wall box's four edges as (start (n, 2), direction (n, 2))."""
    starts, directions = [], []
    for w in walls:
        c, s = np.cos(w['yaw_rad']), np.sin(w['yaw_rad'])
        half = np.array(w['size_xyz'][:2], float)/2
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*half
        corners = np.asarray(w['centre_xyz'][:2], float)+local@np.array([[c, s], [-s, c]])
        starts.append(corners)
        directions.append(np.roll(corners, -1, axis=0)-corners)
    return np.concatenate(starts), np.concatenate(directions)


def ray_distances(origins, angles, edges, discs=None):
    """Free distance from each origin (F, 2) along its world-frame rays (F, R), capped at `MAX_M`, against wall edges
    and walker centres `discs` (F, M, 2) (NaN for an absent walker)."""
    dx, dy = np.cos(angles), np.sin(angles)
    start, along = edges
    ax, ay = (start[None, None]-origins[:, None, None]).transpose(3, 0, 1, 2)
    bx, by = along[:, 0][None, None], along[:, 1][None, None]
    dx3, dy3 = dx[:, :, None], dy[:, :, None]
    determinant = dx3*by-dy3*bx
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ax*by-ay*bx)/determinant
        u = (ax*dy3-ay*dx3)/determinant
    t = np.where((np.abs(determinant) > 1e-12)&(t >= 0)&(u >= 0)&(u <= 1), t, MAX_M)
    best = np.minimum(t.min(2), MAX_M)
    if discs is not None and discs.size:
        c = discs[:, None]-origins[:, None, None]
        ahead = c[..., 0]*dx3+c[..., 1]*dy3
        off = c[..., 0]*dy3-c[..., 1]*dx3
        with np.errstate(invalid='ignore'):
            root = RADIUS_M**2-off*off
            hit = ahead-np.sqrt(np.where(root > 0, root, np.nan))
        hit = np.where((hit >= 0)&(ahead > 0), hit, MAX_M)
        best = np.minimum(best, np.nanmin(np.where(np.isnan(hit), MAX_M, hit), axis=2))
    return best


def unit_rays(args):
    """Label one unit's frames (see the module docstring); returns (folder, frames labelled)."""
    root, folder = Path(args[0]), Path(args[1])
    out = folder/FILE
    if out.exists():
        return str(folder), None
    meta = json.loads((folder/'meta.json').read_text())
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose'].astype(float)
    edges = wall_edges(mp.unit_spec(root, meta)['geometry']['wall_boxes'])
    discs = None
    if meta['obstacles']['placement']:
        with np.load(folder/'dynamic.npz') as dynamic:
            discs = dynamic['obstacles_xy'].astype(float)
    yaw = np.array([float(v1.planar_yaw(p[3:7])) for p in pose])
    bearings = 2*np.pi*np.arange(RAYS)/RAYS
    rays = np.zeros((len(pose), RAYS), np.float16)
    for s in range(0, len(pose), 256):
        e = min(s+256, len(pose))
        rays[s:e] = ray_distances(pose[s:e, :2], yaw[s:e, None]+bearings, edges, None if discs is None else discs[s:e])
    tmp = folder/(FILE+'.tmp.npy')
    np.save(tmp, rays)
    os.replace(tmp, out)
    return str(folder), len(rays)


def label(root, *, roles=('lewm_train', 'lewm_val'), workers=None):
    root = Path(root)
    jobs = [(str(root), str(f)) for f in v1.unit_folders(root, list(roles))]
    frames = done = 0
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2)-2),
                             mp_context=multiprocessing.get_context('forkserver')) as pool:
        for i, (_, n) in enumerate(pool.map(unit_rays, jobs, chunksize=1)):
            if n is not None:
                frames, done = frames+n, done+1
            if (i+1) % 200 == 0:
                print(json.dumps(dict(units=i+1, of=len(jobs), labelled=done, frames=frames)), flush=True)
    return dict(units=len(jobs), labelled=done, frames=frames)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root')
    p.add_argument('--roles', nargs='+', default=['lewm_train', 'lewm_val'])
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(label(a.root, roles=a.roles, workers=a.workers)))


if __name__ == '__main__':
    main()
