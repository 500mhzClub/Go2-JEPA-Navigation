"""Module oracles for the failure analysis (docs/dynamic/PLAN.md, oracle substitution): one harness module at a time
is given the simulator's ground truth, and its gain bounds what learning that module can give.

**Frames.** Truth is taken relative to the robot's true pose and placed through its estimated (registered) pose, so an
oracle module agrees with the pose the rest of the harness believes: with only the map oracle on, a true wall stands
where the robot's own frame would put it if the cameras saw it perfectly. With the pose oracle on, the two coincide.

- **Pose** (`registered`): the true pose in the settled initial-body frame replaces visual odometry and the floor
  registration (no pose loss, no view recovery).
- **Map** (`MapOracle`):
  - 'obstacles': the moving obstacles' pixels are removed from the depth images the routing map is built from (by
    true geometry), and the obstacles' true current cells are added to every snapshot instead (never stored, so they
    leave no trail). Walls, floor and the explored extent stay as perceived.
  - 'full': every snapshot is the true map: all floor inside the maze, the walls and the obstacles where they are now.
  - 'static': the true map with the obstacles only in the 1-cm cells (the clearance check, the stopping projection
    and the tracker see them), not in the 5-cm cells the route is planned on. On a perfect maze there is one path to
    the goal, and a moving obstacle on it, inflated by the footprint, otherwise removes the route until it leaves
    (docs/dynamic/STATUS.md, failure analysis).
- **Obstacle prediction** (`OracleTrack`): the clearance check sees every obstacle at its true future positions, from a
  copy of the walkers stepped ahead with the robot held where it is (their yield rule sees it there).
"""
import copy
from dataclasses import dataclass, field, replace

import numpy as np

from go2nav.perception import cameras
from go2nav.perception.mapping import CELL_M, FINE_CELL_M
from go2nav.perception.registration import RegisteredPose
from go2nav.sim import obstacles as moving
from go2nav.sim.sensors import rotation_xyzw

OBSTACLE_MARGIN_M = .03              # pixels this close to an obstacle's surface are the obstacle's
STORED_MARGIN_M = .08                # stored cells this close to an oracle track are taken out of the stored map
FUTURE_S, FUTURE_STEP_S = 4., .02    # how far, and how finely, the walkers are stepped ahead
MAP_MODES = ('obstacles', 'full', 'static')


class GroundTruth:
    """The simulator's state as the oracles need it. Built right after settling (the settled pose is the initial-body
    frame's origin)."""

    def __init__(self, session):
        self.session = session
        pose = session.samples[-1]['base_pose_world']
        self.p0, self.R0 = np.asarray(pose[:3], float), rotation_xyzw(np.asarray(pose[3:], float))
        self.walls = session.spec['geometry']['wall_boxes']

    def world_pose(self):
        pose = self.session.samples[-1]['base_pose_world']
        return np.asarray(pose[:3], float), rotation_xyzw(np.asarray(pose[3:], float))

    def registered(self, frame, measured_ns):
        """The true pose in the initial-body frame, as a registered pose."""
        p, R = self.world_pose()
        position, rotation = self.R0.T@(p-self.p0), self.R0.T@R
        return RegisteredPose(frame, measured_ns, position, rotation, position, rotation, False, None)

    def placed(self, position, rotation):
        """(A, b): world point w -> A w + b in the estimated initial-body frame (truth relative to the true pose, placed
        through the estimated pose `position`, `rotation`)."""
        p, R = self.world_pose()
        A = np.asarray(rotation, float)@R.T
        return A, np.asarray(position, float)-A@p

    def obstacles(self):
        """[(x, y)] world positions of the moving obstacles now."""
        if self.session.obstacles is None:
            return []
        return [w.xy.copy() for w in self.session.obstacles.walkers]

    def futures(self):
        """(times_s, xy (times, obstacles, 2)): the walkers stepped ahead from now with the robot held in place."""
        field_ = self.session.obstacles
        times = np.arange(0., FUTURE_S+1e-9, FUTURE_STEP_S)
        if field_ is None:
            return times, np.zeros((len(times), 0, 2))
        walkers, robot = copy.deepcopy(field_.walkers), self.session.base_xy
        xy = [[w.xy.copy() for w in walkers]]
        for _ in times[1:]:
            xy.append([w.step(FUTURE_STEP_S, robot).copy() for w in walkers])
        return times, np.asarray(xy)

    def obstacle_pixels(self, depth_m, valid, mount):
        """Pixels of one depth image that lie on a moving obstacle (true camera pose and obstacle positions)."""
        centres = self.obstacles()
        if not centres:
            return np.zeros_like(valid)
        p, R = self.world_pose()
        yy, xx = np.indices((cameras.HEIGHT, cameras.WIDTH))
        optical = np.stack((depth_m*(xx+.5-320)/cameras.FOCAL, depth_m*(yy+.5-240)/cameras.FOCAL, depth_m), axis=-1)
        world = (optical@mount[:3, :3].T+mount[:3, 3])@R.T+p
        hit = np.zeros_like(valid)
        for c in centres:
            near = np.hypot(world[..., 0]-c[0], world[..., 1]-c[1]) <= moving.RADIUS_M+OBSTACLE_MARGIN_M
            hit |= near&(world[..., 2] <= moving.HEIGHT_M+OBSTACLE_MARGIN_M)
        return hit&valid


def _cells(points, B, A, b, size):
    """Map cells (at a cell size) of world points (n, 3) placed by (A, b) and the map orientation B."""
    mapped = (points@A.T+b)@np.asarray(B).T
    return set(map(tuple, np.unique(np.floor(mapped[:, :2]/size).astype(int), axis=0).tolist()))


def _disk(centre, size, margin=0.):
    """World points filling an obstacle's footprint (grown by `margin`) at a sampling step of half a cell (z: half its
    height)."""
    step = size/2
    r = moving.RADIUS_M+margin
    g = np.arange(-r, r+step, step)
    x, y = np.meshgrid(g, g)
    keep = x*x+y*y <= r*r
    return np.stack((x[keep]+centre[0], y[keep]+centre[1], np.full(keep.sum(), moving.HEIGHT_M/2)), axis=1)


def _box(wall, size):
    """World points filling a wall box's footprint at a sampling step of half a cell."""
    step = size/2
    sx, sy = wall['size_xyz'][0]/2, wall['size_xyz'][1]/2
    x, y = np.meshgrid(np.arange(-sx, sx+step, step), np.arange(-sy, sy+step, step))
    c, s = np.cos(wall['yaw_rad']), np.sin(wall['yaw_rad'])
    cx, cy, _ = wall['centre_xyz']
    return np.stack((cx+c*x.ravel()-s*y.ravel(), cy+s*x.ravel()+c*y.ravel(), np.full(x.size, .3)), axis=1)


class MapOracle:
    """See the module docstring: `mask` the depth images before the map update, `apply` to the snapshot after it."""

    def __init__(self, truth, mode):
        if mode not in MAP_MODES:
            raise ValueError(f'map oracle mode {mode!r}')
        self.truth, self.mode = truth, mode
        if mode != 'obstacles':
            self.wall_points = {size: np.concatenate([_box(w, size) for w in truth.walls])
                                for size in (CELL_M, FINE_CELL_M)}
            corners = self.wall_points[CELL_M][:, :2]
            low, high = corners.min(0), corners.max(0)
            g = [np.arange(lo, hi, CELL_M/2) for lo, hi in zip(low, high)]
            x, y = np.meshgrid(*g)
            self.floor_points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)

    def mask(self, depth, auxiliary_depth):
        """The depth pairs with the obstacles' pixels made invalid ('obstacles'; unchanged in the other modes)."""
        if self.mode != 'obstacles':
            return depth, auxiliary_depth
        out = []
        for (depth_m, valid), mount in ((depth, cameras.PRIMARY), (auxiliary_depth, cameras.AUXILIARY)):
            out.append((depth_m, valid&~self.truth.obstacle_pixels(depth_m, valid, mount)))
        return tuple(out)

    def apply(self, snapshot, position, rotation):
        """The snapshot with the oracle's cells (see the module docstring)."""
        A, b = self.truth.placed(position, rotation)
        B = snapshot.map_from_initial
        disks = {size: [_disk(c, size) for c in self.truth.obstacles()] for size in (CELL_M, FINE_CELL_M)}
        coarse = set().union(*(_cells(d, B, A, b, CELL_M) for d in disks[CELL_M]))
        fine = set().union(*(_cells(d, B, A, b, FINE_CELL_M) for d in disks[FINE_CELL_M]))
        if self.mode == 'obstacles':
            return replace(snapshot, occupied=snapshot.occupied|coarse, fine_occupied=snapshot.fine_occupied|fine,
                           current_occupied=snapshot.current_occupied|coarse,
                           current_fine_occupied=snapshot.current_fine_occupied|fine)
        walls = _cells(self.wall_points[CELL_M], B, A, b, CELL_M)
        fine_walls = _cells(self.wall_points[FINE_CELL_M], B, A, b, FINE_CELL_M)
        floor = frozenset(_cells(self.floor_points, B, A, b, CELL_M)-walls)
        if self.mode == 'static':
            coarse = set()
        occupied, fine_occupied = frozenset(walls|coarse), frozenset(fine_walls|fine)
        return replace(snapshot, floor=floor, occupied=occupied, fine_occupied=fine_occupied, current_floor=floor,
                       current_occupied=occupied, current_fine_occupied=fine_occupied, cleared=frozenset())


@dataclass
class OracleTrack:
    """One obstacle for the clearance check: its true cells now (1-cm map cells) and its true map-frame displacement
    `t` seconds ahead (`displacement`), in the planner's track interface. `stored_cells` (the disk grown by
    `STORED_MARGIN_M`) leave the stored cells: what the map perceived of it, give or take pose error."""
    id: int
    cells: frozenset
    stored_cells: frozenset
    seen_ns: int
    times: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)   # (times, 2) map-frame displacement from now

    confirmed = True

    @property
    def velocity(self):
        return (self.offsets[1]-self.offsets[0])/(self.times[1]-self.times[0])

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    def displacement(self, t):
        return np.array([np.interp(t, self.times, self.offsets[:, k]) for k in (0, 1)])

    def summary(self, now_ns):
        return dict(id=self.id, oracle=True, cells=len(self.cells),
                    displacement_1s_m=[round(float(v), 3) for v in self.displacement(1.)])


def oracle_tracks(truth, snapshot, position, rotation):
    """Every moving obstacle as an `OracleTrack` at this snapshot's time."""
    A, b = truth.placed(position, rotation)
    B = np.asarray(snapshot.map_from_initial)
    times, xy = truth.futures()
    tracks = []
    for k in range(xy.shape[1]):
        cells = frozenset(_cells(_disk(xy[0, k], FINE_CELL_M), B, A, b, FINE_CELL_M))
        stored = frozenset(_cells(_disk(xy[0, k], FINE_CELL_M, STORED_MARGIN_M), B, A, b, FINE_CELL_M))
        moved = np.concatenate((xy[:, k]-xy[0, k], np.zeros((len(times), 1))), axis=1)
        offsets = (moved@A.T@B.T)[:, :2]
        tracks.append(OracleTrack(k, cells, stored, int(snapshot.measured_ns), times, offsets))
    return tracks
