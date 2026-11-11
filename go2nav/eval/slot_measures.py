"""P23's measures (docs/dynamic/PLAN.md P23 item 5): do the slots bind to the walkers in view, and where do they put them?

    python -m go2nav.eval.slot_measures score --model PATH --features FEATURES --corpus CORPUS --out FILE [--units N]

Each val unit's frames are run through the slot model in order (slots carried from frame to frame, as they would be in a
mission), and each patch is given to the slot whose decoder mask is largest there. A slot's image columns are those
holding at least `MIN_PATCHES` of its patches.

- **In view** (labels for scoring only): `eval.mover_probe.in_view` on the unit's true walker positions and the layout's
  walls; the walker's columns are its cylinder's bearings through the primary camera (`perception.cameras`).
- **Binding:** a walker in view is bound when one slot covers at least half its columns with a column IoU of at least
  `IOU`.
- **Position** (engineered, from observations): the bound slot's columns give bearings; the ring's nearest valid return
  over them (corpus v2's backfilled scans), pushed one walker radius along its beam, is the slot's position; its error
  against the walker's true position in the body frame.
- Reported over all walker-frames in view and split by whether the robot turns faster than `TURNING_RAD_S`.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from go2nav.data import lewm_corpus as v1
from go2nav.eval import mover_probe as mp
from go2nav.models.slots import PATCHES
from go2nav.perception.cameras import FOCAL, WIDTH
from go2nav.perception.range import AZIMUTHS, MOUNT_BODY_M
from go2nav.sim.obstacles import RADIUS_M

ROWS, COLUMNS = 12, 16
MIN_PATCHES, IOU, NEAR_M, TURNING_RAD_S, CHUNK = 2, .5, .15, .3, 32


def columns_of(bearing_low, bearing_high):
    """The image columns (of COLUMNS) between two bearings (body frame, left positive) through the pinhole."""
    u = WIDTH/2-FOCAL*np.tan(np.array([bearing_high, bearing_low]))
    lo, hi = np.clip(np.floor(u/WIDTH*COLUMNS), 0, COLUMNS-1).astype(int)
    return set(range(int(lo), int(hi)+1))


def bearing_of_column(c):
    u = (c+.5)/COLUMNS*WIDTH
    return math.atan2(WIDTH/2-u, FOCAL)


@torch.no_grad()
def slot_columns(model, features, device):
    """Per frame, per slot, the set of image columns it holds (T lists of K sets)."""
    out, slots = [], None
    for i in range(0, len(features), CHUNK):
        x = torch.as_tensor(np.asarray(features[i:i+CHUNK]), device=device)[None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            s, _ = model(x, slots=slots)
            _, _, masks = model.decode(s)
        slots = s[:, -1]
        owner = masks[0].float().argmax(1).cpu().numpy().reshape(-1, ROWS, COLUMNS)           # (T, rows, cols)
        for frame in owner:
            counts = [(frame == k).sum(0) for k in range(model.config.slots)]
            out.append([set(np.flatnonzero(c >= MIN_PATCHES).tolist()) for c in counts])
    return out


def score_unit(model, unit, features_file, device):
    from go2nav.mazes.sets import MazeSet
    from go2nav.sim import kinematic as kin
    meta = json.loads((unit/'meta.json').read_text())
    with np.load(unit/'dynamic.npz') as d:
        walkers = d['obstacles_xy']
    if walkers.shape[1] == 0:
        return []
    with np.load(unit/'steps.npz') as d:
        pose = d['pose']
    range_file = unit/'range.npz'                       # P14's units record their own scans; others are backfilled
    if not range_file.exists():
        range_file = Path(str(unit).replace('go2nav_lewm_v2', 'go2nav_lewm_v2_range'))/'range.npz'
    with np.load(range_file) as d:
        ranges, valid = d['ranges_m'].astype(np.float32), d['valid']
    geometry = kin.Geometry(MazeSet(meta['set'], root=v1.mazes_root(unit.parents[2])).maze(meta['maze']))   # the walls only
    columns = slot_columns(model, np.load(features_file, mmap_mode='r'), device)
    yaw = np.array([float(v1.planar_yaw(p[3:7])) for p in pose])
    rate = np.abs(np.angle(np.exp(1j*np.diff(yaw, prepend=yaw[0]))))/.1
    rows = []
    for t in range(len(columns)):
        for distance, k in mp.in_view(pose[t], walkers[t], geometry.visible):
            x, y = mp.body_frame(pose[t], walkers[t, k])
            bearing = math.atan2(y, x-mp.CAMERA_X_M)
            half = math.asin(min(1., RADIUS_M/max(distance, 1e-6)))
            truth = columns_of(bearing-half, bearing+half)
            best, best_iou = None, 0.
            for s, cols in enumerate(columns[t]):
                if not cols or len(cols & truth) < len(truth)/2:
                    continue
                iou = len(cols & truth)/len(cols | truth)
                if iou > best_iou:
                    best, best_iou = s, iou
            bound = best is not None and best_iou >= IOU
            error = None
            if bound:
                bearings = [bearing_of_column(c) for c in columns[t][best]]
                lo, hi = min(bearings)-.5*math.radians(78.323/COLUMNS), max(bearings)+.5*math.radians(78.323/COLUMNS)
                beams = valid[t] & (AZIMUTHS >= lo) & (AZIMUTHS <= hi)
                if beams.any():
                    j = int(np.flatnonzero(beams)[ranges[t][beams].argmin()])
                    r = float(ranges[t][j])+RADIUS_M
                    px, py = MOUNT_BODY_M[0]+r*math.cos(AZIMUTHS[j]), r*math.sin(AZIMUTHS[j])
                    error = math.hypot(px-x, py-y)
            rows.append(dict(bound=bool(bound), iou=round(best_iou, 3), error=error, turning=bool(rate[t] > TURNING_RAD_S),
                             distance=round(distance, 2)))
    return rows


def summarise(rows):
    def part(sel):
        if not sel:
            return None
        bound = [r for r in sel if r['bound']]
        placed = [r for r in bound if r['error'] is not None]
        return dict(walker_frames=len(sel), bound=round(len(bound)/len(sel), 3),
                    within_015=round(sum(r['error'] <= NEAR_M for r in placed)/max(len(bound), 1), 3),
                    median_error_m=round(float(np.median([r['error'] for r in placed])), 3) if placed else None)
    return dict(all=part(rows), turning=part([r for r in rows if r['turning']]),
                straight=part([r for r in rows if not r['turning']]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--model', required=True, type=Path)
    s.add_argument('--features', required=True, type=Path)
    s.add_argument('--corpus', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    s.add_argument('--units', type=int, default=100)
    a = p.parse_args()
    from go2nav.training.slots import load_model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.enable_flash_sdp(False), torch.backends.cuda.enable_mem_efficient_sdp(False)
    model = load_model(a.model, device)
    index = json.loads((a.features/'index.json').read_text())
    rows = []
    for path in index['units']['lewm_val'][:a.units]:
        rows += score_unit(model, a.corpus/path, a.features/'lewm_val'/f'{Path(path).name}.npy', device)
    report = summarise(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
