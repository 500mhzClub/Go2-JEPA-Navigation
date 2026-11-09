"""The simulated range sensor (docs/dynamic/PLAN.md, P13 item 1): the beams of `go2nav.perception.range`, cast analytically.

Each 10-Hz frame, every beam leaves the mount at the robot's current pose (the ring tilts with the body) and stops at
the nearest of: a wall box (`geometry.wall_boxes`, each a box turned by its yaw), a walker's cylinder (radius
`obstacles.RADIUS_M`, from the floor to its height, at its current position) or the floor (z = 0). The robot's own body
is never hit, an idealisation of the real sensor's self-occlusion; nor are the floor's returns otherwise special: they
lie below the obstacle band, so perception drops them as it drops floor depth. Ranges carry Gaussian noise of
`NOISE_SIGMA_M`, seeded by layout and frame so a mission replays exactly; a beam is valid when its noisy range lies
within `MIN_RANGE_M`-`MAX_RANGE_M` (a surface nearer than the near blind spot gives no return, as on the L1).
"""
import numpy as np

from go2nav.perception.range import DIRECTIONS_BODY, MAX_RANGE_M, MIN_RANGE_M, MOUNT_BODY_M
from go2nav.sim.obstacles import RADIUS_M

NOISE_SIGMA_M = .01
NOISE_SEED = 2026110701
EPSILON = 1e-12


def rotation_wxyz(q):
    """The rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def box_distances(origin, directions, boxes):
    """(n,) distance along each unit direction from `origin` to the nearest box (inf where none is hit). `boxes` is a
    (m, 7) array: centre xyz, half sizes xyz, yaw. A ray starting inside a box hits it at 0."""
    if not len(boxes):
        return np.full(len(directions), np.inf)
    c, s = np.cos(-boxes[:, 6]), np.sin(-boxes[:, 6])                                 # (m,) into each box's axes
    o = origin[None, :]-boxes[:, :3]                                                  # (m, 3) origin from each centre
    o_local = np.stack((c*o[:, 0]-s*o[:, 1], s*o[:, 0]+c*o[:, 1], o[:, 2]), axis=1)
    dx, dy, dz = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]           # (n, 1)
    d_local = np.stack((c*dx-s*dy, s*dx+c*dy, np.broadcast_to(dz, (len(directions), len(boxes)))), axis=-1)  # (n, m, 3)
    h = boxes[None, :, 3:6]
    parallel = np.abs(d_local) < EPSILON
    safe = np.where(parallel, 1., d_local)
    t1, t2 = (-h-o_local[None])/safe, (h-o_local[None])/safe
    outside = parallel & (np.abs(o_local[None]) > h)                                 # parallel to a slab and outside it
    t_near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(-1)
    t_far = np.where(parallel, np.inf, np.maximum(t1, t2)).min(-1)
    hit = (t_near <= t_far) & (t_far >= 0) & ~outside.any(-1)
    return np.where(hit, np.maximum(t_near, 0.), np.inf).min(1)


def cylinder_distances(origin, directions, centres_xy, heights, radius):
    """(n,) distance along each unit direction to the nearest vertical cylinder standing on the floor (inf where none)."""
    if not len(centres_xy):
        return np.full(len(directions), np.inf)
    o = origin[None, :2]-np.asarray(centres_xy, float)                               # (m, 2)
    d = directions[:, None, :2]                                                       # (n, 1, 2)
    a = (d*d).sum(-1)                                                                 # (n, 1)
    b = 2*(d*o[None]).sum(-1)                                                         # (n, m)
    c = (o*o).sum(-1)[None]-radius**2                                                 # (1, m)
    disc = b*b-4*a*c
    root = np.sqrt(np.maximum(disc, 0.))
    a_safe = np.where(a < EPSILON, 1., a)
    near, far = (-b-root)/(2*a_safe), (-b+root)/(2*a_safe)
    t = np.where(c <= 0, 0., near)                                                    # starting inside: hit at once
    t = np.where((disc >= 0) & (a >= EPSILON) & (far >= 0), np.maximum(t, 0.), np.inf)
    with np.errstate(invalid='ignore'):
        z = origin[2]+np.where(np.isfinite(t), t, 0.)*directions[:, None, 2]
    t = np.where(np.isfinite(t) & (z >= 0) & (z <= np.asarray(heights, float)[None]), t, np.inf)
    return t.min(1)


def floor_distances(origin, directions):
    """(n,) distance along each unit direction to the floor plane z = 0 (inf for beams level or rising)."""
    dz = directions[:, 2]
    return np.where(dz < -EPSILON, -origin[2]/np.where(dz < -EPSILON, dz, -1.), np.inf)


class RangeSensor:
    """The range sensor of one mission's scene: `scan` gives one frame's (ranges_m, valid)."""

    def __init__(self, spec, noise_sigma_m=NOISE_SIGMA_M):
        boxes = spec['geometry']['wall_boxes']
        self.boxes = np.array([[*b['centre_xyz'], *(np.asarray(b['size_xyz'], float)/2), float(b.get('yaw_rad', 0.))]
                               for b in boxes], float).reshape(-1, 7)
        self.noise_sigma_m = noise_sigma_m
        self.layout = int(spec.get('layout_index', 0))

    def true_distances(self, base_position, base_wxyz, walkers_xy=(), walker_heights=()):
        """(n,) noiseless distances from the mount along each beam to the nearest surface (inf where none)."""
        R = rotation_wxyz(base_wxyz)
        origin = np.asarray(base_position, float)+R@MOUNT_BODY_M
        directions = DIRECTIONS_BODY@R.T
        return np.minimum.reduce([box_distances(origin, directions, self.boxes),
                                  cylinder_distances(origin, directions, np.asarray(walkers_xy, float).reshape(-1, 2),
                                                     walker_heights, RADIUS_M),
                                  floor_distances(origin, directions)])

    def scan(self, base_position, base_wxyz, walkers_xy=(), walker_heights=(), *, frame):
        """One frame's (ranges_m float32 (BEAMS,), valid bool (BEAMS,)); invalid beams carry range 0."""
        t = self.true_distances(base_position, base_wxyz, walkers_xy, walker_heights)
        if self.noise_sigma_m:
            rng = np.random.default_rng([NOISE_SEED, self.layout, int(frame)])
            t = t+rng.normal(0., self.noise_sigma_m, len(t))
        valid = np.isfinite(t) & (t >= MIN_RANGE_M) & (t <= MAX_RANGE_M)
        return np.where(valid, t, 0.).astype(np.float32), valid
