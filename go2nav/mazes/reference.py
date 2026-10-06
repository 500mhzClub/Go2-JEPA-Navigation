"""Shortest paths on the true maze geometry, for a disk-shaped robot footprint (evaluator only).

The robot is a disk of radius `radius_m` that must keep `clearance_m` from every wall. Distances are geodesics on a
regular grid (`resolution_m`) of the free space, 8-connected, with diagonal corner cutting forbidden; a straight segment
is used directly when it is clear. Used to choose episodes (start-beacon distance and clearance) and to score shortest
path length. Controllers never see it.
"""
import heapq
import math

import numpy as np


def walls_from_boxes(wall_boxes):
    """Wall boxes from a maze specification, as (centre, size, yaw) records in the plane."""
    return [dict(center=np.asarray(w['centre_xyz'][:2]), size=np.asarray(w['size_xyz'][:2]), yaw=w['yaw_rad'])
            for w in wall_boxes]


def segment_crosses_wall(start, end, walls):
    """True when the segment start-end intersects any axis-aligned wall box (slab test)."""
    start, end = np.asarray(start), np.asarray(end)
    for wall in walls:
        if wall['yaw'] != 0:
            raise ValueError('axis-aligned walls required')
        low, high = wall['center']-wall['size']/2, wall['center']+wall['size']/2
        enter, leave = 0., 1.
        for k in (0, 1):
            delta = end[k]-start[k]
            if abs(delta) < 1e-14:
                if not low[k] <= start[k] <= high[k]:
                    enter, leave = 1., 0.
                    break
            else:
                a, b = sorted(((low[k]-start[k])/delta, (high[k]-start[k])/delta))
                enter, leave = max(enter, a), min(leave, b)
        if enter <= leave:
            return True
    return False


class ReferenceGeometry:
    """Distance from any point to one target, over the inflated free space."""

    def __init__(self, walls, bounds, target_xy, *, radius_m, clearance_m, resolution_m):
        self.walls = walls
        self.bounds = np.asarray(bounds, float)
        self.target = np.asarray(target_xy, float)
        self.radius, self.clearance, self.resolution = float(radius_m), float(clearance_m), float(resolution_m)
        self.shape = tuple((np.floor((self.bounds[1]-self.bounds[0])/self.resolution).astype(int)+1).tolist())
        self.distance_field = None
        self.free = None

    def footprint_clearance(self, xy):
        """Signed gap between the footprint disk at each point and the nearest wall or world bound (metres)."""
        points = np.asarray(xy, float)
        flat = points.reshape(-1, 2)
        margin = np.minimum(flat-self.bounds[0], self.bounds[1]-flat).min(axis=1)
        for wall in self.walls:
            c, s = math.cos(wall['yaw']), math.sin(wall['yaw'])
            local = (flat-wall['center']) @ np.array([[c, -s], [s, c]])
            delta = np.abs(local)-np.asarray(wall['size'])/2
            distance = np.linalg.norm(np.maximum(delta, 0), axis=1)+np.minimum(np.max(delta, axis=1), 0)
            margin = np.minimum(margin, distance)
        return (margin-self.radius).reshape(points.shape[:-1])

    def segment_clear(self, start, end):
        """Conservative check that the whole straight segment keeps the clearance (wall distance is 1-Lipschitz)."""
        start, end = np.asarray(start), np.asarray(end)
        distance = np.linalg.norm(end-start)
        count = max(1, int(math.ceil(distance/(self.resolution/4))))
        samples = start+np.linspace(0, 1, count+1)[:, None]*(end-start)
        lower = float(np.min(self.footprint_clearance(samples)))-distance/(2*count)
        return lower >= self.clearance

    def _point(self, index):
        return self.bounds[0]+np.asarray(index)*self.resolution

    def _nearby(self, point):
        middle = np.rint((point-self.bounds[0])/self.resolution).astype(int)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                index = (int(middle[0]+dx), int(middle[1]+dy))
                if all(0 <= index[k] < self.shape[k] for k in (0, 1)):
                    yield index

    def _build_field(self):
        """Dijkstra from the target over free grid points (a half-diagonal margin protects interpolation in a cell)."""
        grid = np.indices(self.shape).transpose(1, 2, 0)*self.resolution+self.bounds[0]
        self.free = self.footprint_clearance(grid) >= self.clearance+self.resolution/math.sqrt(2)
        distances = np.full(self.shape, np.inf)
        queue = []
        for index in self._nearby(self.target):
            if self.free[index] and self.segment_clear(self.target, self._point(index)):
                d = float(np.linalg.norm(self._point(index)-self.target))
                distances[index] = d
                heapq.heappush(queue, (d, *index))
        steps = [(dx, dy, self.resolution*math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            d, x, y = heapq.heappop(queue)
            if d != distances[x, y]:
                continue
            for dx, dy, length in steps:
                xx, yy = x+dx, y+dy
                if not (0 <= xx < self.shape[0] and 0 <= yy < self.shape[1]) or not self.free[xx, yy]:
                    continue
                if dx and dy and not (self.free[x+dx, y] and self.free[x, y+dy]):
                    continue
                if d+length < distances[xx, yy]:
                    distances[xx, yy] = d+length
                    heapq.heappush(queue, (d+length, xx, yy))
        self.distance_field = distances

    def distance_and_heading(self, point):
        """Geodesic distance from `point` to the target and the initial heading of that path.

        Returns dict(valid, distance_m, heading_rad, direct_segment) or dict(valid=False, reason).
        """
        point = np.asarray(point, float)
        if self.footprint_clearance(self.target) < self.clearance:
            return dict(valid=False, reason='TARGET_PHYSICALLY_INVALID')
        if self.footprint_clearance(point) < self.clearance:
            return dict(valid=False, reason='ENDPOINT_OUTSIDE_REFERENCE_FREE_SPACE')
        vector = self.target-point
        direct = float(np.linalg.norm(vector))
        if direct <= 1e-12:
            return dict(valid=True, distance_m=0., heading_rad=None, direct_segment=True)
        if self.segment_clear(point, self.target):
            return dict(valid=True, distance_m=direct, heading_rad=math.atan2(vector[1], vector[0]), direct_segment=True)
        if self.distance_field is None:
            self._build_field()
        options = []
        for index in self._nearby(point):
            d = self.distance_field[index]
            if np.isfinite(d) and self.segment_clear(point, self._point(index)):
                options.append((d+np.linalg.norm(self._point(index)-point), index))
        if not options:
            return dict(valid=False, reason='NO_REFERENCE_GEODESIC')
        distance, index = min(options)
        # Follow the descent 0.20 m ahead so the heading is not a grid-edge artefact of a tiny endpoint offset.
        current = index
        for _ in range(int(math.ceil(.20/self.resolution))):
            neighbours = [j for j in self._nearby(self._point(current))
                          if self.distance_field[j] < self.distance_field[current]
                          and self.segment_clear(self._point(current), self._point(j))]
            if not neighbours:
                break
            candidate = min(neighbours, key=lambda j: (self.distance_field[j], j))
            if not self.segment_clear(point, self._point(candidate)):
                break
            current = candidate
        vector = self._point(current)-point
        if np.linalg.norm(vector) <= 1e-12:
            return dict(valid=False, reason='UNDEFINED_GEODESIC_DESCENT')
        return dict(valid=True, distance_m=float(distance), heading_rad=math.atan2(vector[1], vector[0]), direct_segment=False)
