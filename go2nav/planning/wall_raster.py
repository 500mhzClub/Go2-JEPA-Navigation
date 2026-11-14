"""The walls' distance of a layout on a regular grid (docs/dynamic/PLAN.md P22): the walls round an object or the
robot as the object-level predictor sees them, looked up the same way in training (in torch, many layouts in one
batch) and in planning (in numpy, one layout).

A raster holds the distance to the nearest wall's surface (`sim.kinematic.Geometry.wall_distance`) at nodes
`RASTER_M` apart from `low`, with `MARGIN_M` of floor beyond the walls' bounds; a lookup is bilinear and clamps to the
raster, so a point outside the layout reads as the nearest edge. `context` samples a square grid of `size` x `size`
points `spacing` apart, centred on each point and turned by a heading: the walls round a thing, seen from the robot.
"""
from pathlib import Path

import numpy as np

RASTER_M, MARGIN_M = .04, .5


def grid_offsets(size, spacing):
    """(size*size, 2) offsets of a square grid centred on 0, x-major."""
    half = (size-1)/2
    a = (np.arange(size)-half)*spacing
    return np.stack(np.meshgrid(a, a, indexing='ij'), -1).reshape(-1, 2)


class WallRaster:
    def __init__(self, distance, low, step=RASTER_M):
        self.distance, self.low, self.step = np.asarray(distance, np.float32), np.asarray(low, np.float64), float(step)

    @classmethod
    def of(cls, geometry, step=RASTER_M):
        low = geometry.low-MARGIN_M
        shape = np.ceil((geometry.high+MARGIN_M-low)/step).astype(int)+1
        nodes = low+np.stack(np.meshgrid(*(np.arange(n)*step for n in shape), indexing='ij'), -1)
        return cls(geometry.wall_distance(nodes), low, step)

    def save(self, path):
        path = Path(path)
        partial = path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(partial, distance=self.distance.astype(np.float16), low=self.low, step=self.step)
        partial.rename(path)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['distance'].astype(np.float32), data['low'], float(data['step']))

    def lookup(self, points):
        """The walls' distance at points (..., 2), bilinear."""
        p = (np.asarray(points, float)-self.low)/self.step
        limit = np.array(self.distance.shape)-1
        i = np.clip(np.floor(p).astype(int), 0, limit-1)
        f = np.clip(p-i, 0., 1.)
        d = self.distance
        a, b = d[i[..., 0], i[..., 1]], d[i[..., 0]+1, i[..., 1]]
        c, e = d[i[..., 0], i[..., 1]+1], d[i[..., 0]+1, i[..., 1]+1]
        return (a*(1-f[..., 0])+b*f[..., 0])*(1-f[..., 1])+(c*(1-f[..., 0])+e*f[..., 0])*f[..., 1]

    def context(self, centres, heading, size, spacing):
        """The walls' distance on a grid round each centre (..., 2), turned by `heading` (...,): (..., size*size)."""
        offsets = grid_offsets(size, spacing)
        c, s = np.cos(heading)[..., None], np.sin(heading)[..., None]
        x = np.asarray(centres)[..., None, 0]+c*offsets[:, 0]-s*offsets[:, 1]
        y = np.asarray(centres)[..., None, 1]+s*offsets[:, 0]+c*offsets[:, 1]
        return self.lookup(np.stack((x, y), -1))
