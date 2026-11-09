"""Training-only labels for phase C's heads (docs/dynamic/PLAN.md P15), from each unit's recorded truth.

    python -m go2nav.data.head_labels CORPUS OUT [--workers N] [--roles R ...]

Per frame of every unit, into OUT/units/<role>/<unit>/heads.npz (the corpus is never touched):
- `clearance` (n,) float16: the body outline's clearance in metres, the smallest distance from 8 points of a
  rectangle `OUTLINE_HALF_M` (4 corners and 4 edge midpoints, at the recorded pose) to a wall's surface or a walker's
  (a cylinder of `sim.obstacles.RADIUS_M`). Contact means a clearance under `CONTACT_M`.
- `presence` (n,) bool: a walker within `PRESENCE_M` of the camera, in its view and not hidden by the walls
  (`eval.mover_probe.in_view`);
- `presence_ring` (n,) bool: a walker within `PRESENCE_M` of the body centre at any bearing, not hidden by the walls
  (what the range ring can see).
The robot's own motion is not stored: it is the recorded pose's, computed when loading.

These labels are privileged (the simulator's walls, walkers and pose). They train heads with stop-gradient only, so
they never shape the representation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np

OUTLINE_HALF_M = (.35, .16)     # the Go2's body with its legs, along and across
CONTACT_M = .10
PRESENCE_M = 2.


def outline(pose):
    """(n, 8, 2) world points of the body outline at each pose (n, 7)."""
    from go2nav.data.lewm_corpus import planar_yaw
    a, b = OUTLINE_HALF_M
    local = np.array([[a, b], [a, -b], [-a, b], [-a, -b], [a, 0], [-a, 0], [0, b], [0, -b]], float)
    yaw = np.array([planar_yaw(q) for q in pose[:, 3:7]])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x = pose[:, None, 0]+c*local[None, :, 0]-s*local[None, :, 1]
    y = pose[:, None, 1]+s*local[None, :, 0]+c*local[None, :, 1]
    return np.stack([x, y], -1)


def wall_distance(points, centres, halves):
    """(..., ) distance of each point (..., 2) to the nearest axis-aligned wall box's surface."""
    outside = np.maximum(np.abs(points[..., None, :]-centres)-halves, 0.)
    return np.sqrt((outside**2).sum(-1)).min(-1)


def unit_labels(root, folder):
    from go2nav.data import lewm_corpus_v2 as v2
    from go2nav.eval import mover_probe as mp
    from go2nav.sim.obstacles import RADIUS_M
    meta = json.loads((folder/'meta.json').read_text())
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose'].astype(float)
    with np.load(folder/'dynamic.npz') as dynamic:
        walkers = dynamic['obstacles_xy'].astype(float)
    spec = mp.unit_spec(root, meta)
    boxes = spec['geometry']['wall_boxes']
    centres = np.array([b['centre_xyz'][:2] for b in boxes], float)
    halves = np.array([b['size_xyz'][:2] for b in boxes], float)/2
    points = outline(pose)
    clearance = wall_distance(points, centres, halves).min(-1)
    presence, ring = np.zeros(len(pose), bool), np.zeros(len(pose), bool)
    if walkers.shape[1]:
        to_walkers = np.linalg.norm(points[:, :, None]-walkers[:, None], axis=-1).min(1)-RADIUS_M
        clearance = np.minimum(clearance, to_walkers.min(-1))
        sight = mp.sight_line(v2.FreeDriverV2(spec, np.random.default_rng(0)).wall_clearance)
        for t in range(len(pose)):
            presence[t] = any(d < PRESENCE_M for d, _ in mp.in_view(pose[t], walkers[t], sight))
            ring[t] = any(np.linalg.norm(w-pose[t, :2]) < PRESENCE_M and sight(pose[t, :2], w) for w in walkers[t])
    return dict(clearance=clearance.astype(np.float16), presence=presence, presence_ring=ring)


def _job(args):
    root, out, path = (Path(a) for a in args)
    target = out/path/'heads.npz'
    if not target.exists():
        labels = unit_labels(root, root/path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name('heads.tmp.npz')
        np.savez(partial, **labels)
        partial.rename(target)
    with np.load(target) as d:
        return int(len(d['clearance'])), int((d['clearance'] < CONTACT_M).sum()), int(d['presence'].sum())


def build(root, out, *, roles=('lewm_train', 'lewm_val'), workers=None):
    """Label every unit of `roles` (resumable: units already labelled are kept). Returns the counts."""
    from go2nav.data import lewm_corpus as v1
    root, out = Path(root), Path(out)
    paths = [str(f.relative_to(root)) for f in v1.unit_folders(root, list(roles))]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        counts = list(pool.map(_job, [(str(root), str(out), p) for p in paths], chunksize=4))
    frames, contact, presence = (sum(c[i] for c in counts) for i in range(3))
    summary = dict(units=len(paths), frames=frames, contact_frames=contact, contact_share=round(contact/max(frames, 1), 5),
                   contact_units=sum(c[1] > 0 for c in counts), presence_share=round(presence/max(frames, 1), 4))
    (out/'heads_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('corpus', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--roles', nargs='+', default=['lewm_train', 'lewm_val'])
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    print(json.dumps(build(a.corpus, a.out, roles=a.roles, workers=a.workers), indent=1))


if __name__ == '__main__':
    main()
