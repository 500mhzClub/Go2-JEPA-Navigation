"""Depth obstacles: what both cameras currently see above the floor, near the robot.

Each frame (independently of the visual odometry), the floor plane is fitted to both cameras' floor points along the
gyro's up direction (`floor`). Every depth sample (every 4 px) between 3 and 65 cm above that plane marks its 1-cm
cell of the current body frame, within 1 m of the body origin. The 50 Hz dispatch check tests commands against
these cells.
"""
from dataclasses import dataclass

import numpy as np

from go2nav.perception import cameras, floor
from go2nav.perception.rigid import UnportedBranch
from go2nav.perception.rotations import GyroAttitude

CELL_M = .01
HALF_EXTENT_CELLS = 100
MIN_HEIGHT_M, MAX_HEIGHT_M = .03, .65
STRIDE = 4


@dataclass(frozen=True)
class Obstacles:
    """Occupied 1-cm cells (i, j) of the body frame at one frame: x in [i, i+1) cm, y in [j, j+1) cm."""
    frame: int
    measured_ns: int
    cells: frozenset
    returns: tuple  # valid depth samples per camera (primary, auxiliary)


def sampled_points(depth_m, valid, mount):
    """Body-frame points of the valid depth samples on a 4-px grid."""
    rows, columns = np.arange(STRIDE//2, cameras.HEIGHT, STRIDE), np.arange(STRIDE//2, cameras.WIDTH, STRIDE)
    u, v = np.meshgrid(columns+.5, rows+.5)
    z = depth_m[np.ix_(rows, columns)]
    keep = valid[np.ix_(rows, columns)]
    optical = np.stack((z*(u-320)/cameras.FOCAL, z*(v-240)/cameras.FOCAL, z), axis=-1)
    return (optical@mount[:3, :3].T+mount[:3, 3])[keep]


class DepthObstacleObserver:
    """See the module docstring. `observe` returns the frame's `Obstacles`, or None without a usable floor plane."""

    def __init__(self):
        self.gyro = GyroAttitude(orthonormalize=False)
        self.initial_up = None
        self.frames = 0

    def observe(self, depth, auxiliary_depth, gyro_rates, specific_force, now):
        """`depth`/`auxiliary_depth`: (depth_m, valid) pairs; `gyro_rates`: the frame's (51, 3) block;
        `specific_force`: the (20, 3) accelerometer history (used once, for the initial up direction)."""
        G = self.gyro.update(gyro_rates)
        if self.initial_up is None:
            up = np.asarray(specific_force).mean(0)
            self.initial_up = up/np.linalg.norm(up)
        up = G.T@self.initial_up
        clouds = [sampled_points(*depth, cameras.PRIMARY), sampled_points(*auxiliary_depth, cameras.AUXILIARY)]
        floor_points, _ = floor.paired_floor_points(depth, auxiliary_depth, up)
        plane = floor.joint_plane(*floor_points, up)
        frame = self.frames
        self.frames += 1
        if not plane['available'] and plane['reason'] == 'insufficient_combined_two_axis_extent':
            raise UnportedBranch('obstacle floor plane lacks extent (the original tries a gyro-height plane)')
        if not plane['available']:
            return None
        if any(not len(points) for points in clouds):
            raise UnportedBranch('obstacles with an empty camera (the original tries auxiliary-only obstacles)')
        normal, cells = plane['normal_body'], set()
        for points in clouds:
            height = points@normal+plane['offset_body_m']
            above = points[(height > MIN_HEIGHT_M)&(height < MAX_HEIGHT_M)]
            cells.update(tuple(map(int, k)) for k in np.unique(np.floor(above[:, :2]/CELL_M).astype(int), axis=0))
        local = frozenset(k for k in cells if all(-HALF_EXTENT_CELLS <= v < HALF_EXTENT_CELLS for v in k))
        return Obstacles(frame, now, local, tuple(map(len, clouds)))
