"""Walker-occupancy labels around the robot, seen or not, for the memory handover (docs/dynamic/PLAN.md, P9).

    python -m go2nav.data.memory_labels ROOT [--roles lewm_val lewm_train] [--workers N]

Per corpus v2 unit with walkers, `memory_cells.npz` beside its frames, every `STRIDE`-th frame (`t`):
- `occupied` (T, CELLS, CELLS): a walker's centre within `OCCUPIED_M` of the cell's centre, on a body-frame grid of
  `CELL_M` cells spanning ±`HALF_M` (row i is x, forward, from -HALF_M; column j is y, left, from -HALF_M);
- `visible` (T, CELLS, CELLS): the cell's centre in the primary camera's view (`mover_probe`'s field of view and range)
  and not hidden by the walls from the body origin, as `mover_probe.in_view` tests a walker's centre;
- `seen` (frames, K): walker k in view at that frame (`mover_probe.in_view`, and the frame has at least
  `MIN_OBSTACLE_PIXELS` obstacle depth pixels), for every frame, so that a baseline can follow what the robot saw;
- `age_s` (T, K): seconds since walker k was last seen (inf if never), and `polite` (K).
The labels come from the simulator's records and are never a model input; a unit already labelled is skipped.
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

FILE = 'memory_cells.npz'
CELL_M, HALF_M = .2, 1.6
CELLS = round(2*HALF_M/CELL_M)
OCCUPIED_M = .275                  # the walker's radius (0.175 m) and half a cell
STRIDE, FRAME_S = 2, .1
SIGHT_MARGIN_M, SIGHT_END_M = .05, .2   # as `mover_probe.sight_line`: clear by 5 cm, the last 0.2 m not tested
CENTRES = (np.arange(CELLS)+.5)*CELL_M-HALF_M
GRID = np.stack(np.meshgrid(CENTRES, CENTRES, indexing='ij'), -1).reshape(-1, 2)   # body frame (x, y)


def world(pose, body):
    """Body-frame xy (..., 2) of a pose in world coordinates."""
    yaw = float(v1.planar_yaw(np.asarray(pose)[3:7]))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c*body[..., 0]-s*body[..., 1], s*body[..., 0]+c*body[..., 1]], -1)+np.asarray(pose)[:2]


def hidden(a, b, centres, halves):
    """Whether the walls (boxes, grown by `SIGHT_MARGIN_M`) cross each segment a-b (b: (N, 2)), leaving out its first
    5 cm and last `SIGHT_END_M`, as `mover_probe.sight_line` samples it."""
    d = b-a
    length = np.linalg.norm(d, axis=1)
    s0, s1 = SIGHT_MARGIN_M/np.maximum(length, 1e-9), (length-SIGHT_END_M)/np.maximum(length, 1e-9)
    lo, hi = centres-halves-SIGHT_MARGIN_M, centres+halves+SIGHT_MARGIN_M          # (B, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1./d[:, None, :]                                                          # (N, 1, 2)
        t1, t2 = (lo[None]-a)*inv, (hi[None]-a)*inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    flat = d[:, None, :] == 0
    inside = (lo[None] <= a) & (a <= hi[None])
    tmin = np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2)).max(-1)
    tmax = np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2)).min(-1)
    hit = (tmax >= np.maximum(tmin, s0[:, None])) & (tmin <= s1[:, None]) & (tmax >= tmin)
    return hit.any(1) & (s1 > s0)


def cell_visible(pose, centres, halves):
    """The grid cells whose centres are in the primary camera's view and not hidden by the walls (flat, CELLS**2)."""
    x, y = GRID[:, 0]-mp.CAMERA_X_M, GRID[:, 1]
    distance = np.hypot(x, y)
    cone = (distance > .2) & (distance < mp.MAX_VIEW_M) & (np.abs(np.arctan2(y, x)) <= mp.HALF_FOV_RAD)
    out = np.zeros(len(GRID), bool)
    if cone.any():
        out[cone] = ~hidden(np.asarray(pose[:2], float), world(pose, GRID[cone]), centres, halves)
    return out


def unit_cells(args):
    """Label one unit (see the module docstring); returns (folder, frames labelled)."""
    root, folder = Path(args[0]), Path(args[1])
    out = folder/FILE
    if out.exists():
        return str(folder), None
    meta = json.loads((folder/'meta.json').read_text())
    placement = meta['obstacles']['placement']
    if not placement:
        return str(folder), 0
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose']
    with np.load(folder/'dynamic.npz') as dynamic:
        movers, pixels = dynamic['obstacles_xy'].astype(float), dynamic['obstacle_pixels']
    geometry = v2.FreeDriverV2(mp.unit_spec(root, meta), np.random.default_rng(0))
    centres, halves = geometry.centres, geometry.halves
    sight = mp.sight_line(geometry.wall_clearance)
    n, K = movers.shape[:2]
    seen = np.zeros((n, K), bool)
    for t in np.flatnonzero(pixels >= v2.MIN_OBSTACLE_PIXELS):
        for _, k in mp.in_view(pose[t], movers[t], sight):
            seen[t, k] = True
    last = np.full(K, -np.inf)
    age = np.zeros((n, K))
    for t in range(n):
        last[seen[t]] = t
        age[t] = (t-last)*FRAME_S
    ts = np.arange(0, n, STRIDE)
    occupied = np.zeros((len(ts), CELLS*CELLS), bool)
    visible = np.zeros((len(ts), CELLS*CELLS), bool)
    for i, t in enumerate(ts):
        body = mp.body_frame(pose[t], movers[t])                                    # (K, 2)
        occupied[i] = (np.linalg.norm(GRID[:, None]-body[None], axis=2) < OCCUPIED_M).any(1)
        visible[i] = cell_visible(pose[t], centres, halves)
    shape = (len(ts), CELLS, CELLS)
    tmp = folder/(FILE+'.tmp.npz')
    np.savez_compressed(tmp, t=ts, occupied=occupied.reshape(shape), visible=visible.reshape(shape), seen=seen,
                        age_s=age[ts].astype(np.float32), polite=np.array([o.get('polite', True) for o in placement]))
    os.replace(tmp, out)
    return str(folder), len(ts)


def label(root, roles=('lewm_val', 'lewm_train'), workers=None):
    root = Path(root)
    folders = [f for role in roles for f in sorted((root/'units'/role).iterdir()) if (f/'meta.json').exists()]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('spawn')) as pool:
        done = 0
        for folder, frames in pool.map(unit_cells, [(str(root), str(f)) for f in folders], chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f'{done} of {len(folders)} units', flush=True)
    print(f'{len(folders)} units labelled', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('root', type=Path)
    p.add_argument('--roles', nargs='+', default=['lewm_val', 'lewm_train'])
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    label(a.root, a.roles, a.workers)


if __name__ == '__main__':
    main()
