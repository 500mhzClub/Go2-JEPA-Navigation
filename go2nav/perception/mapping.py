"""The routing map: observed floor and obstacles on a fixed grid, from the registered pose and both depth images.

**Map frame.** Gravity-aligned at the start: z is up (the initial accelerometer direction), x the initial forward
direction projected on the floor, origin at the initial body origin. `map_from_initial` (B) takes initial-body
coordinates into it. The floor height is measured once, from the first frame's joint floor plane.

**Cells.** The map spans ±8 m with 5-cm cells (320 x 320) and keeps:
- `floor`: cells seen as floor. A cell counts as seen when its square, projected into a camera image at the floor
  height, is fully visible and every image quad it overlaps has valid depth within 1 cm of the floor height
  (`floor_coverage`; without a current floor plane, the quads must also be locally planar and level);
- `occupied`: cells containing a depth sample (every 4 px) between 3 and 65 cm above the floor;
- `fine_occupied`: the same samples at 1 cm.
Cells accumulate over the mission; the snapshot also gives this observation's own cells. The map is updated every
fourth frame (2.5 Hz).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from go2nav.perception import cameras, floor
from go2nav.perception.obstacles import sampled_points
from go2nav.perception.rigid import UnportedBranch

CELL_M, HALF_CELLS = .05, 160
FINE_CELL_M, FINE_HALF_CELLS = .01, 800
MIN_HEIGHT_M, MAX_HEIGHT_M = .03, .65
FLOOR_BAND_M = .01
GRID = np.array([(i, j) for i in range(-HALF_CELLS, HALF_CELLS) for j in range(-HALF_CELLS, HALF_CELLS)], np.int64)
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


@dataclass(frozen=True)
class MapSnapshot:
    """The map after one update (cells are (i, j) map-grid indices)."""
    frame: int
    measured_ns: int
    floor: frozenset
    occupied: frozenset
    position_map: tuple            # the body origin in the map frame
    map_from_initial: tuple        # B (3 x 3, rows)
    floor_height: float
    primary_current_floor_cells: int
    auxiliary_current_floor_cells: int
    fine_occupied: frozenset
    current_floor: frozenset
    current_occupied: frozenset
    current_fine_occupied: frozenset


def body_projection(depth_m):
    """Body-frame point of every pixel (primary-camera geometry)."""
    yy, xx = np.indices((cameras.HEIGHT, cameras.WIDTH))
    optical = np.stack((depth_m*(xx+.5-320)/cameras.FOCAL, depth_m*(yy+.5-240)/cameras.FOCAL, depth_m), axis=-1)
    return optical@cameras.PRIMARY[:3, :3].T+cameras.PRIMARY[:3, 3]


def valid_quads(valid):
    """Image quads (between four pixel centres) whose corners are all valid."""
    return valid[:-1, :-1]&valid[1:, :-1]&valid[:-1, 1:]&valid[1:, 1:]


def level_floor_quads(depth_m, valid, up):
    """Valid quads more than 15 cm below the body whose two triangles are planar (3 mm) and level (normal within
    acos(0.97) of up); used for coverage when no current floor plane is available."""
    d = np.asarray(depth_m)
    u = (np.arange(640)+.5-320)/cameras.FOCAL
    v = (np.arange(480)+.5-240)/cameras.FOCAL
    optical = np.stack((d*u[None], d*v[:, None], d), axis=2)
    points = optical@cameras.PRIMARY[:3, :3].T+cameras.PRIMARY[:3, 3]
    a, b, c, e = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    good = valid[:-1, :-1]&valid[:-1, 1:]&valid[1:, 1:]&valid[1:, :-1]
    for p in (a, b, c, e):
        good &= p@up < -.15
    for pair, (left, right) in enumerate(((b, e), (b, c), (c, e))):
        rows, columns = np.nonzero(good)
        if not len(rows):
            break
        anchor = a[rows, columns]
        normal = np.cross(left[rows, columns]-anchor, right[rows, columns]-anchor)
        length = np.linalg.norm(normal, axis=1)
        alignment = np.abs(normal@up)
        tolerance = 64*np.finfo(float).eps*length
        if (np.any(np.abs(length-1e-10) <= 64*np.finfo(float).eps*1e-10)
                or np.any(np.abs(alignment-.97*length) <= tolerance)):
            raise UnportedBranch('ambiguous last-bit floor quad (the original uses its dense index here)')
        accepted = (length > 1e-10)&(alignment >= .97*length)
        if pair == 0:
            error = np.abs(np.sum((c[rows, columns]-anchor)*normal, axis=1))
            if np.any(np.abs(error-.003*length) <= tolerance):
                raise UnportedBranch('ambiguous last-bit floor quad (the original uses its dense index here)')
            accepted &= error <= .003*length
        good[rows, columns] = accepted
    return good


def _prefers_dense_index(depth_m, valid, up):
    rows, columns = np.arange(0, 480, 16), np.arange(0, 640, 16)
    z, mask = depth_m[np.ix_(rows, columns)], valid[np.ix_(rows, columns)]
    if not np.isfinite(z).all() or np.any(z[mask] < .2) or np.any(z[mask] > 5):
        return True
    optical = np.stack((z*((columns+.5-320)/cameras.FOCAL)[None], z*((rows+.5-240)/cameras.FOCAL)[:, None], z), axis=2)
    points = optical@cameras.PRIMARY[:3, :3].T+cameras.PRIMARY[:3, 3]
    return bool(np.count_nonzero(mask&(points@up < -.15)) > .5*mask.size)


@njit
def _polygon_coverage(good, uv, visible, a, b, original):
    """Projected-square coverage: a square is covered unless an image quad it overlaps (separating-axis test,
    boundary contact included) is not good floor."""
    covered = original.copy()
    axes = np.empty((6, 2))
    lower = np.empty(6)
    upper = np.empty(6)
    for i in range(len(covered)):
        if covered[i] or not visible[i]:
            continue
        axes[0, 0] = 1.
        axes[0, 1] = 0.
        axes[1, 0] = 0.
        axes[1, 1] = 1.
        for edge in range(4):
            following = (edge+1)%4
            axes[edge+2, 0] = -(uv[i, following, 1]-uv[i, edge, 1])
            axes[edge+2, 1] = uv[i, following, 0]-uv[i, edge, 0]
        for axis in range(6):
            lo = 1e300
            hi = -1e300
            for corner in range(4):
                value = uv[i, corner, 0]*axes[axis, 0]+uv[i, corner, 1]*axes[axis, 1]
                lo = min(lo, value)
                hi = max(hi, value)
            lower[axis] = lo
            upper[axis] = hi
        rejected = False
        for y in range(a[i, 1], b[i, 1]+1):
            for x in range(a[i, 0], b[i, 0]+1):
                if good[y, x]:
                    continue
                intersects = True
                for axis in range(6):
                    nx, ny = axes[axis, 0], axes[axis, 1]
                    lo = (x if nx >= 0 else x+1)*nx+(y if ny >= 0 else y+1)*ny
                    hi = lo+abs(nx)+abs(ny)
                    if hi < lower[axis]-1e-9 or lo > upper[axis]+1e-9:
                        intersects = False
                        break
                if intersects:
                    rejected = True
                    break
            if rejected:
                break
        covered[i] = not rejected
    return covered


def floor_coverage(depth_m, valid, map_from_body, translation_map, floor_height, plane_available):
    """Which GRID cells one camera image shows as floor (see the module docstring). The camera pose is given as the
    map pose of a body whose primary camera would coincide with it (`cameras` reference-adapter convention)."""
    R, p = np.asarray(map_from_body, float), np.asarray(translation_map, float)
    up_body = R[2]
    if plane_available:
        quads = valid_quads(valid)
    else:
        depth_m, valid = cameras.local_inverse_depth(depth_m, valid)
        if _prefers_dense_index(depth_m, valid, up_body):
            raise UnportedBranch('dense floor-cell index')
        quads = level_floor_quads(depth_m, valid, up_body)
    near = np.abs(body_projection(depth_m)@up_body+p[2]-floor_height) <= FLOOR_BAND_M
    good = quads&near[:-1, :-1]&near[:-1, 1:]&near[1:, :-1]&near[1:, 1:]
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~good).cumsum(0).cumsum(1)
    xy = (GRID[:, None, :]+SQUARE)*CELL_M
    world = np.concatenate((xy, np.full((*xy.shape[:-1], 1), floor_height)), axis=-1)
    T = cameras.PRIMARY
    camera = ((world-p)@R-T[:3, 3])@T[:3, :3]
    z = camera[..., 2]
    uv = camera[..., :2]/np.maximum(z[..., None], 1e-12)*cameras.FOCAL+[319.5, 239.5]
    lo, hi = uv.min(1)-1e-9, uv.max(1)+1e-9
    visible = ((z >= .2)&(z <= 5.)).all(1)&(lo >= 0).all(1)&(hi < [639, 479]).all(1)
    a, b = np.floor(np.clip(lo, -1, 640)).astype(int), np.floor(np.clip(hi, -1, 640)).astype(int)
    covered = np.zeros(len(GRID), bool)
    ids = np.flatnonzero(visible)
    if len(ids):
        x0, y0 = a[ids].T
        x1, y1 = (b[ids]+1).T
        covered[ids] = prefix[y1, x1]-prefix[y0, x1]-prefix[y1, x0]+prefix[y0, x0] == 0
    if plane_available:
        covered = _polygon_coverage(good, uv, visible, a, b, covered)
    return covered


def measured_floor_height(plane, map_from_body, position_map):
    """Height of the measured floor plane below the map origin."""
    normal = np.asarray(map_from_body)@np.asarray(plane['normal_body'])
    offset = float(plane['offset_body_m']-normal@np.asarray(position_map))
    return -offset/float(normal[2])


class RoutingMap:
    """See the module docstring. `update` takes the registered pose and both depth images of one frame."""

    def __init__(self):
        self.B = None
        self.floor_height = None
        self.floor, self.occupied, self.fine_occupied = set(), set(), set()
        self.latest = None

    def update(self, frame, measured_ns, position, rotation, depth, auxiliary_depth, specific_force):
        """`position`/`rotation`: the registered pose (initial-body frame); `depth`/`auxiliary_depth`: (depth_m, valid)
        pairs; `specific_force`: the (20, 3) accelerometer history (used once, for the map orientation)."""
        p, R = np.asarray(position, float), np.asarray(rotation, float)
        if self.floor_height is None:
            if self.B is None:
                if frame != 0:
                    raise UnportedBranch('first map update after frame 0')
                up = np.asarray(specific_force).mean(0)
                up /= np.linalg.norm(up)
                forward = np.array([1., 0., 0.])-up*up[0]
                forward /= np.linalg.norm(forward)
                self.B = np.stack((forward, np.cross(up, forward), up))
            up_body = R.T@self.B[2]
            up_body /= np.linalg.norm(up_body)
            plane = floor.joint_plane(*floor.paired_floor_points(depth, auxiliary_depth, up_body)[0], up_body)
            if not plane['available']:
                raise UnportedBranch('no floor plane at the first map update (start-up recovery)')
            self.floor_height = measured_floor_height(plane, self.B@R, self.B@p)
        current_up = R.T@self.B[2]
        plane_available = floor.joint_plane(*floor.paired_floor_points(depth, auxiliary_depth, current_up)[0],
                                            current_up)['available']
        Q, q = self.B@R, self.B@p
        counts, new_floor, new_occupied, new_fine = [], set(), set(), set()
        for auxiliary, (depth_m, valid) in ((False, depth), (True, auxiliary_depth)):
            if auxiliary:
                camera_R = Q@cameras.AUXILIARY[:3, :3]@cameras.PRIMARY[:3, :3].T
                camera_p = q+Q@cameras.AUXILIARY[:3, 3]-camera_R@cameras.PRIMARY[:3, 3]
            else:
                camera_R, camera_p = Q, q
            covered = floor_coverage(depth_m, valid, camera_R, camera_p, self.floor_height, plane_available)
            new_floor.update(tuple(map(int, cell)) for cell in GRID[covered])
            counts.append(int(covered.sum()))
            points = sampled_points(depth_m, valid, cameras.AUXILIARY if auxiliary else cameras.PRIMARY)
            mapped = ((points@R.T+p)@self.B.T) if auxiliary else (points@Q.T+q)
            above = mapped[(mapped[:, 2] > self.floor_height+MIN_HEIGHT_M)&(mapped[:, 2] < self.floor_height+MAX_HEIGHT_M)]
            keys = np.unique(np.floor(above[:, :2]/CELL_M).astype(int), axis=0)
            new_occupied.update(tuple(map(int, cell)) for cell in keys
                                if np.all(cell >= -HALF_CELLS) and np.all(cell < HALF_CELLS))
            fine_keys = np.unique(np.floor(above[:, :2]/FINE_CELL_M).astype(int), axis=0)
            new_fine.update(tuple(map(int, cell)) for cell in fine_keys
                            if np.all(cell >= -FINE_HALF_CELLS) and np.all(cell < FINE_HALF_CELLS))
        self.floor.update(new_floor)
        self.occupied.update(new_occupied)
        self.fine_occupied.update(new_fine)
        self.latest = MapSnapshot(frame, measured_ns, frozenset(self.floor), frozenset(self.occupied),
                                  tuple(map(float, q)), tuple(tuple(map(float, row)) for row in self.B), self.floor_height,
                                  *counts, frozenset(self.fine_occupied), frozenset(new_floor), frozenset(new_occupied),
                                  frozenset(new_fine))
        return self.latest
