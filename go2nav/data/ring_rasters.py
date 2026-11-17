"""P31 (docs/dynamic/PLAN.md P31 item 1): the ring world's scans as images, on the GPU, and the corpus's arrays.

**A frame** is a pose (x, y, yaw) the image is drawn in: its origin at the pose, its x axis along the heading. A
return's point is the sensor's mount plus the range along the beam (`perception.range`: `AZIMUTHS` in the body
frame, `MOUNT_BODY_M`); it is splatted bilinearly into a `SIZE` x `SIZE` image of `CELL` m a pixel, centred on the
frame's origin, and the image is clipped at 1. Pixel k's centre is at k + 0.5 (as `sim.balls`), so a pixel position
p is the frame coordinate p * CELL - HALF_M. Column is the frame's x, row its y.

`Corpus` holds a split's packed scans, poses and requests (`data.ring_corpus pack`) and, for measures only, the
walkers' true positions and sight, in the same frame order.
"""
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.perception.range import AZIMUTHS, MOUNT_BODY_M

SIZE, CELL = 64, .1
HALF_M = SIZE*CELL/2


def rotate(v, yaw):
    """Vectors (..., 2) turned by yaw (...)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c*v[..., 0]-s*v[..., 1], s*v[..., 0]+c*v[..., 1]), -1)


def to_frame(world, frame):
    """World points (N, ..., 2) into frames (N, 3)."""
    lead = (len(world),)+(1,)*(world.dim()-2)
    return rotate(world-frame[:, :2].reshape(*lead, 2), -frame[:, 2].reshape(lead))


def to_world(local, frame):
    lead = (len(local),)+(1,)*(local.dim()-2)
    return rotate(local, frame[:, 2].reshape(lead))+frame[:, :2].reshape(*lead, 2)


def returns(ranges, poses):
    """Scans (N, 720) (0: no return) and the robot's poses (N, 3) to the returns' world points (N, 720, 2) and
    validity (N, 720)."""
    r = ranges.float()
    yaw = poses[:, 2].float()
    mount = poses[:, :2].float()+MOUNT_BODY_M[0]*torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
    angle = yaw[:, None]+torch.as_tensor(AZIMUTHS, dtype=torch.float32, device=r.device)
    points = mount[:, None]+r[..., None]*torch.stack((torch.cos(angle), torch.sin(angle)), -1)
    return points, r > 0


def to_pixels(local):
    return (local+HALF_M)/CELL


def to_local(pixels):
    return pixels*CELL-HALF_M


def rasterise(ranges, poses, frames):
    """Scans (N, 720), the robot's poses when they were taken (N, 3) and the frames to draw them in (N, 3) to images
    (N, 1, SIZE, SIZE) in [0, 1]."""
    points, valid = returns(ranges, poses)
    p = to_pixels(to_frame(points, frames.float()))-.5                                   # pixel-centre coordinates
    base = torch.floor(p)
    f = p-base
    n = len(ranges)
    image = torch.zeros(n, SIZE*SIZE, device=ranges.device)
    rows = torch.arange(n, device=ranges.device)[:, None].expand_as(valid)
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = base[..., 0]+dx, base[..., 1]+dy
            w = (f[..., 0] if dx else 1-f[..., 0])*(f[..., 1] if dy else 1-f[..., 1])
            inside = valid & (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
            index = (cy.clamp(0, SIZE-1)*SIZE+cx.clamp(0, SIZE-1)).long()
            image.index_put_((rows[inside], index[inside]), w[inside], accumulate=True)
    return image.clamp(max=1.).view(n, 1, SIZE, SIZE)


def nearest_return(points, scan_points, scan_valid, clip=1.):
    """The distance from points (N, P, 2) to the nearest valid return of the matching scan (N, 720, 2), clipped:
    the scan's stand-in for the layout's wall distance (N, P)."""
    size = max(1, int(2e8)//max(1, points.shape[1]*scan_points.shape[1]))         # bounded memory: chunks of frames
    out = []
    for i in range(0, len(points), size):
        d = torch.cdist(points[i:i+size].float(), scan_points[i:i+size].float())
        d = torch.where(scan_valid[i:i+size, None], d, torch.full_like(d, 1e3))
        out.append(d.min(-1).values)
    return torch.cat(out).clamp(max=clip)


class Corpus:
    """A split's packed arrays on `device`: `ranges` (F, 720) float16, `pose`, `requested` (F, 3), `starts` (E + 1,);
    with `labels`, the walkers' `walkers_xy` (F, 4, 2) and `in_sight` (F, 4) as numpy, for measures only."""

    def __init__(self, root, split, device, labels=False):
        packed = Path(root)/'packed'
        self.ranges = torch.from_numpy(np.load(packed/f'{split}_ranges.npy')).to(device)
        self.pose = torch.from_numpy(np.load(packed/f'{split}_pose.npy')).to(device)
        self.requested = torch.from_numpy(np.load(packed/f'{split}_requested.npy')).to(device)
        self.starts = np.load(packed/f'{split}_starts.npy')
        self.device = device
        if labels:
            xy, sight = [], []
            for k in range(len(self.starts)-1):
                with np.load(Path(root)/split/f'{k:05d}.npz') as z:
                    xy.append(z['walkers_xy']), sight.append(z['in_sight'])
            self.walkers_xy, self.in_sight = np.concatenate(xy), np.concatenate(sight)
            assert len(self.walkers_xy) == self.starts[-1]
        self.layouts = None

    def mazes(self, root, split):
        return [json.loads((Path(root)/split/f'{k:05d}.json').read_text())['maze'] for k in range(len(self.starts)-1)]
