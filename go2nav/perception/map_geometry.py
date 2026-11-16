"""The built map as a layout (docs/dynamic/PLAN.md P24 2b): what P17's planner, the ring tracker and M5 read from a
maze's walls, from the routing map instead (`perception.mapping.MapSnapshot`), in the map frame.

**Walls** are the map's occupied 5-cm cells never seen as floor. A wall's cell is never seen as floor; a walker stands
on floor seen free before, so it is not a wall (the reasoning of `perception.tracks`). A walker standing where the
floor was never seen counts as a wall until it moves. Cells nobody has seen are free: the planner's distance runs
through them, and the hand harness's rule keeps the robot off floor not yet seen (`planning.selection`).

`MapGeometry` gives what `sim.kinematic.Geometry` gives the planner (`planning.fields.Fields`, `planning.wall_raster`,
`perception.ring_walkers`): `low` and `high` (the known cells' bounds and the robot, with `MARGIN_M`), the distance of
points to the nearest wall's surface (`wall_distance`: a Euclidean distance transform of the wall cells, read
bilinearly, less half a cell), the line of sight (`visible`), and no maze cells (`neighbours`).
"""
import numpy as np
from scipy import ndimage

from go2nav.perception.mapping import CELL_M

MARGIN_M = 1.


class MapGeometry:
    def __init__(self, snapshot, robot_xy=None):
        walls = set(snapshot.occupied)-set(snapshot.floor)
        known = np.array(sorted(set(snapshot.occupied) | set(snapshot.floor)) or [(0, 0)], float)
        corners = np.concatenate((known*CELL_M, (known+1)*CELL_M))
        if robot_xy is not None:
            corners = np.concatenate((corners, np.asarray(robot_xy, float)[None]))
        self.low, self.high = corners.min(0)-MARGIN_M, corners.max(0)+MARGIN_M
        self.origin = np.floor(self.low/CELL_M).astype(int)                # the grid's first cell
        shape = np.ceil(self.high/CELL_M).astype(int)-self.origin+1
        grid = np.zeros(tuple(shape), bool)
        if walls:
            cells = np.array(sorted(walls))-self.origin
            inside = ((cells >= 0) & (cells < shape)).all(1)
            grid[cells[inside, 0], cells[inside, 1]] = True
        self.walls = grid
        self.distance = ndimage.distance_transform_edt(~grid)*CELL_M if grid.any() else np.full(grid.shape, 1e3)
        self.neighbours, self.pitch = {}, 1.3
        self.cells = len(walls)

    def wall_distance(self, points):
        """(...,) the distance of each point (..., 2) to the nearest wall cell's surface (0 inside one)."""
        p = np.asarray(points, float)/CELL_M-.5-self.origin                  # in cell-centre units
        limit = np.array(self.distance.shape)-1
        i = np.clip(np.floor(p).astype(int), 0, np.maximum(limit-1, 0))
        f = np.clip(p-i, 0., 1.)
        d = self.distance
        j = np.minimum(i+1, limit)
        a, b = d[i[..., 0], i[..., 1]], d[j[..., 0], i[..., 1]]
        c, e = d[i[..., 0], j[..., 1]], d[j[..., 0], j[..., 1]]
        value = (a*(1-f[..., 0])+b*f[..., 0])*(1-f[..., 1])+(c*(1-f[..., 0])+e*f[..., 0])*f[..., 1]
        return np.maximum(value-CELL_M/2, 0.)

    def visible(self, a, b, step=.04):
        """Whether the straight line from a to b crosses no wall (sampled every `step`)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(2, int(np.linalg.norm(b-a)/step)+1)
        return bool((self.wall_distance(a+(b-a)*np.linspace(0., 1., count)[:, None]) > 0.).all())
