"""Map-grid geometry for routing and clearance: cells, segments, inflation and obstacle distances.

The map (`go2nav.perception.mapping`) has 5-cm cells (index (i, j) covers [i, i+1) x [j, j+1) x 5 cm) within ±8 m,
and keeps obstacles at 1 cm as well. The robot's nominal footprint is a 0.45-m disk.
"""
from functools import lru_cache
import heapq
import math

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

CELL_M, HALF_CELLS = .05, 160
FINE_CELL_M, FINE_HALF_CELLS = .01, 800
MAP_HALF_WIDTH_M = 8.
FOOTPRINT_RADIUS_M = .45
NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))
PREFERRED_CLEARANCE_M, CLEARANCE_WEIGHT = .6, 2.


def centre(cell):
    """Centre of a 5-cm cell (map frame, metres)."""
    return (np.asarray(cell, float)+.5)*CELL_M


def cell_of(point):
    return tuple(map(int, np.floor(np.asarray(point)/CELL_M)))


def segment_cells(start, end):
    """Every 5-cm cell a closed segment touches (corner contacts included)."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    low = np.floor(np.minimum(a, b)/CELL_M).astype(int)-1
    high = np.floor(np.maximum(a, b)/CELL_M).astype(int)
    x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1), indexing='ij')
    cells = np.column_stack((x.ravel(), y.ravel()))
    lo = cells*CELL_M
    hi = lo+CELL_M
    entry, leave = np.zeros(len(cells)), np.ones(len(cells))
    for axis in range(2):
        delta = b[axis]-a[axis]
        if abs(delta) < 1e-15:
            outside = (a[axis] < lo[:, axis]-1e-12)|(a[axis] > hi[:, axis]+1e-12)
            entry[outside], leave[outside] = 1., 0.
        else:
            u, v = (lo[:, axis]-a[axis])/delta, (hi[:, axis]-a[axis])/delta
            entry = np.maximum(entry, np.minimum(u, v))
            leave = np.minimum(leave, np.maximum(u, v))
    return {tuple(map(int, c)) for c in cells[entry <= leave+1e-12]}


def segment_cell_distances(start, end, cells, cell_m=CELL_M):
    """Euclidean distance from a closed segment to each cell square (0 where it crosses the square)."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    keys = np.asarray(cells)
    if keys.size == 0:
        return np.empty(0, float)
    low = keys*cell_m
    high = low+cell_m
    delta = b-a
    length2 = float(delta@delta)
    distances = np.minimum(np.linalg.norm(np.maximum(np.maximum(low-a, a-high), 0.), axis=1),
                           np.linalg.norm(np.maximum(np.maximum(low-b, b-high), 0.), axis=1))
    corners = low[:, None]+np.array([[0, 0], [0, 1], [1, 0], [1, 1]])*cell_m
    t = np.clip(((corners-a)@delta)/length2, 0., 1.) if length2 else np.zeros(corners.shape[:2])
    distances = np.minimum(distances, np.linalg.norm(corners-(a+t[..., None]*delta), axis=2).min(axis=1))
    entry, leave, cross = np.zeros(len(keys)), np.ones(len(keys)), np.ones(len(keys), bool)
    for axis in range(2):
        if delta[axis] == 0.:
            cross &= (a[axis] >= low[:, axis])&(a[axis] <= high[:, axis])
        else:
            u, v = (low[:, axis]-a[axis])/delta[axis], (high[:, axis]-a[axis])/delta[axis]
            entry, leave = np.maximum(entry, np.minimum(u, v)), np.minimum(leave, np.maximum(u, v))
    distances[cross&(entry <= leave)] = 0.
    return distances


def disk_connector(start, end, cells, *, radius_m=FOOTPRINT_RADIUS_M, cell_m=CELL_M):
    """Whether a disk swept along a segment stays clear of the cells (strictly beyond its radius)."""
    distances = segment_cell_distances(start, end, cells, cell_m)
    return not len(distances) or float(distances[int(np.argmin(distances))]) > radius_m+1e-12


def nearest_cell_distance(point, cells, cell_m=CELL_M):
    """Distance from a point to the nearest cell square (infinite with no cells)."""
    distances = segment_cell_distances(point, point, cells, cell_m)
    return float(distances.min()) if len(distances) else math.inf


def inflated_cells(occupied, radius_m=FOOTPRINT_RADIUS_M):
    """Cells within `radius_m` of an occupied cell (square to square)."""
    return set(_inflated(frozenset(occupied), radius_m))


@lru_cache(maxsize=2)
def _inflated(occupied, radius_m):
    if not occupied:
        return frozenset()
    n = math.ceil(radius_m/CELL_M)+1
    offsets = np.asarray([(x, y) for x in range(-n, n+1) for y in range(-n, n+1)
                          if np.hypot(max(abs(x)-1, 0), max(abs(y)-1, 0))*CELL_M <= radius_m+1e-12], dtype=int)
    points = np.asarray(tuple(occupied), dtype=int)
    lo, hi = points.min(0)-n, points.max(0)+n
    mask = np.zeros(tuple(hi-lo+1), bool)
    local = points-lo
    for start in range(0, len(local), 512):
        indices = local[start:start+512, None, :]+offsets[None, :, :]
        mask[indices[:, :, 0], indices[:, :, 1]] = True
    return frozenset((int(x+lo[0]), int(y+lo[1])) for x, y in np.argwhere(mask))


class FineClearance:
    """Distance from a segment to the nearest 1-cm obstacle cell (None without obstacles)."""

    def __init__(self, cells):
        self.cells = np.asarray(sorted(cells), dtype=int).reshape(-1, 2)
        self.low = self.cells*FINE_CELL_M
        self.high = self.low+FINE_CELL_M
        self.tree = cKDTree((self.low+self.high)*.5) if len(self.cells) else None
        self._memo = {}

    def minimum(self, start, end):
        key = (tuple(start), tuple(end))
        if key not in self._memo:
            self._memo[key] = self._minimum(np.asarray(start, float), np.asarray(end, float))
        return self._memo[key]

    def _minimum(self, a, b):
        if not len(self.cells):
            return None
        midpoint = (a+b)*.5
        _, nearest = self.tree.query(midpoint)
        gap = np.maximum(np.maximum(self.low[nearest]-midpoint, midpoint-self.high[nearest]), 0.)
        upper = float(np.linalg.norm(gap))
        # Any closer square's centre lies within half the segment, that bound and a half diagonal of the midpoint.
        radius = float(np.linalg.norm(b-a))*.5+upper+np.sqrt(2.)*(FINE_CELL_M/2)+1e-12
        indices = self.tree.query_ball_point(midpoint, radius)
        return float(segment_cell_distances(a, b, self.cells[indices], FINE_CELL_M).min())

    def clear(self, start, end, radius_m=FOOTPRINT_RADIUS_M):
        d = self.minimum(start, end)
        return d is None or d > radius_m+1e-12


@lru_cache(maxsize=2)
def fine_clearance(cells):
    """FineClearance for a frozenset of 1-cm cells (the last two maps are kept)."""
    return FineClearance(cells)


@njit(cache=False)
def _axis_minimum(a, b, low, high):
    xmin, xmax = min(a[0], b[0]), max(a[0], b[0])
    ymin, ymax = min(a[1], b[1]), max(a[1], b[1])
    best = math.inf
    for i in range(len(low)):
        dx = max(low[i, 0]-xmax, xmin-high[i, 0], 0.)
        dy = max(low[i, 1]-ymax, ymin-high[i, 1], 0.)
        squared = dx*dx+dy*dy
        if squared < best:
            best = squared
    return math.sqrt(best)


class AxisClearance:
    """`FineClearance` for axis-aligned segments (grid edges), by brute force over every obstacle cell."""

    def __init__(self, cells):
        self.fine = fine_clearance(cells)
        self._memo = {}

    def minimum(self, start, end):
        key = (tuple(start), tuple(end))
        if key not in self._memo:
            a, b = np.asarray(start, float), np.asarray(end, float)
            if not len(self.fine.cells):
                self._memo[key] = None
            elif a[0] != b[0] and a[1] != b[1]:
                self._memo[key] = self.fine.minimum(a, b)
            else:
                self._memo[key] = float(_axis_minimum(a, b, self.fine.low, self.fine.high))
        return self._memo[key]


@lru_cache(maxsize=2)
def axis_clearance(cells):
    return AxisClearance(cells)


@lru_cache(maxsize=2)
def clearance_costs(occupied):
    """Per-cell routing cost: 1, plus 2 x ((0.6 m - clearance)/0.15 m)² where an obstacle is closer than 0.6 m."""
    mask = np.ones((2*HALF_CELLS, 2*HALF_CELLS), dtype=bool)
    for x, y in occupied:
        mask[x+HALF_CELLS, y+HALF_CELLS] = False
    if not occupied:
        return np.ones_like(mask, dtype=float)
    distance = np.maximum(0., distance_transform_edt(mask)*CELL_M-math.sqrt(2.)*CELL_M/2)
    deficit = np.maximum(0., PREFERRED_CLEARANCE_M-distance)/.15
    return 1.+CLEARANCE_WEIGHT*deficit**2


def cost_step(cost, a, b):
    return .5*(cost[a[0]+HALF_CELLS, a[1]+HALF_CELLS]+cost[b[0]+HALF_CELLS, b[1]+HALF_CELLS])


def astar(start, target, passable, cost, edge_clear=None):
    """Clearance-weighted A* over 4-connected cells (Manhattan heuristic): (path, cost, cells reached) or None."""
    def heuristic(c):
        return abs(c[0]-target[0])+abs(c[1]-target[1])
    queue, distance, parent, edges = [(heuristic(start), 0., start)], {start: 0.}, {start: None}, {}
    while queue:
        _, value, cell = heapq.heappop(queue)
        if value != distance[cell]:
            continue
        if cell == target:
            path, here = [], cell
            while here is not None:
                path.append(here)
                here = parent[here]
            return path[::-1], value, len(parent)
        for dx, dy in NEIGHBOURS:
            nxt = (cell[0]+dx, cell[1]+dy)
            if not passable(nxt):
                continue
            candidate = value+cost_step(cost, cell, nxt)
            if candidate >= distance.get(nxt, float('inf')):
                continue
            if edge_clear is not None:
                edge = tuple(sorted((cell, nxt)))
                if edge not in edges:
                    edges[edge] = edge_clear(centre(cell), centre(nxt))
                if not edges[edge]:
                    continue
            distance[nxt], parent[nxt] = candidate, cell
            heapq.heappush(queue, (candidate+heuristic(nxt), candidate, nxt))
    return None
