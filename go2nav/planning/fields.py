"""Distances to walk through a layout (docs/dynamic/PLAN.md P17): what the sampling planner's proposals follow and what
its cost reads as progress.

`Fields` lays a `RESOLUTION_M` grid over a `sim.kinematic.Geometry`. A node is closed within `HARD_M` of a wall (the
body's half-width and a little more) and costs more to cross up to `SOFT_M` from one, so paths keep to the middle of a
passage. `static(target)` is every node's shortest walking distance to a target through the walls alone;
`with_walkers(target, walkers)` closes the nodes within `WALKER_M` of each walker too (its radius, the body's half-width
and the planner's clearance floor), so a passage a walker stands in is sealed: an infinite distance. Both are cached.
`value` reads a field at positions (the least over the 3 x 3 nodes round each, so a position on a closed node's edge
still reads). `wall_clearance` reads the walls' distance at points from a finer grid (`FINE_M`), for the planner's
rollouts; the world itself measures exactly.
"""
from collections import OrderedDict

import numpy as np
from skimage.graph import MCP_Geometric

from go2nav.sim import obstacles as moving

RESOLUTION_M, FINE_M = .05, .02
HARD_M, SOFT_M, SOFT_COST = .20, .30, 4.
WALKER_M = moving.RADIUS_M+.16+.08
QUANTUM_M = .1                      # walkers' positions are rounded to this for the cache
CACHE = 512


class Fields:
    def __init__(self, geometry):
        self.geometry = geometry
        self.low = geometry.low.copy()
        shape = np.ceil((geometry.high-geometry.low)/RESOLUTION_M).astype(int)+1
        self.shape = tuple(int(n) for n in shape)
        self.nodes = self.low+np.stack(np.meshgrid(*(np.arange(n)*RESOLUTION_M for n in self.shape), indexing='ij'), -1)
        wall = geometry.wall_distance(self.nodes)
        self.cost = np.where(wall < HARD_M, -1., 1.+SOFT_COST*np.clip((SOFT_M-wall)/(SOFT_M-HARD_M), 0., 1.))
        fine = np.ceil((geometry.high-geometry.low)/FINE_M).astype(int)+1
        self.fine = geometry.wall_distance(self.low+np.stack(np.meshgrid(*(np.arange(n)*FINE_M for n in fine),
                                                                         indexing='ij'), -1))
        self._static, self._moving = {}, OrderedDict()

    def index(self, xy):
        """The nearest node (..., 2) int of positions (..., 2), inside the grid."""
        i = np.rint((np.asarray(xy, float)-self.low)/RESOLUTION_M).astype(int)
        return np.clip(i, 0, np.array(self.shape)-1)

    def _field(self, cost, target):
        start = tuple(int(v) for v in self.index(target))
        if cost[start] < 0:
            return np.full(self.shape, np.inf)
        return MCP_Geometric(cost, fully_connected=True).find_costs([start])[0]*RESOLUTION_M

    def static(self, target):
        key = tuple(np.round(np.asarray(target, float), 3))
        if key not in self._static:
            self._static[key] = self._field(self.cost, target)
        return self._static[key]

    def with_walkers(self, target, walkers):
        walkers = np.round(np.asarray(walkers, float).reshape(-1, 2)/QUANTUM_M)*QUANTUM_M
        inside = ((walkers > self.low-WALKER_M) & (walkers < self.nodes[-1, -1]+WALKER_M)).all(1)
        walkers = walkers[inside]                  # one outside the layout closes no node (a placeholder slot)
        if not len(walkers):
            return self.static(target)
        key = (tuple(np.round(np.asarray(target, float), 3)), tuple(map(tuple, walkers)))
        if key in self._moving:
            self._moving.move_to_end(key)
            return self._moving[key]
        cost = self.cost.copy()
        for w in walkers:
            near = np.linalg.norm(self.nodes-w, axis=-1) < WALKER_M
            cost[near] = -1.
        field = self._field(cost, target)
        self._moving[key] = field
        if len(self._moving) > CACHE:
            self._moving.popitem(last=False)
        return field

    def value(self, field, xy):
        """A field at positions (..., 2): the least over the 3 x 3 nodes round each."""
        i = self.index(xy)
        out = np.full(i.shape[:-1], np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = np.clip(i+(dx, dy), 0, np.array(self.shape)-1)
                out = np.minimum(out, field[j[..., 0], j[..., 1]])
        return out

    def wall_clearance(self, points):
        """The walls' distance at points (..., 2), bilinear on the fine grid."""
        p = (np.asarray(points, float)-self.low)/FINE_M
        limit = np.array(self.fine.shape)-1
        i = np.clip(np.floor(p).astype(int), 0, limit-1)
        f = np.clip(p-i, 0., 1.)
        a, b = self.fine[i[..., 0], i[..., 1]], self.fine[i[..., 0]+1, i[..., 1]]
        c, d = self.fine[i[..., 0], i[..., 1]+1], self.fine[i[..., 0]+1, i[..., 1]+1]
        return (a*(1-f[..., 0])+b*f[..., 0])*(1-f[..., 1])+(c*(1-f[..., 0])+d*f[..., 0])*f[..., 1]
