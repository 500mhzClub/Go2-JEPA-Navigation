"""A kinematic world for planning studies (docs/dynamic/PLAN.md P17): the maze's walls, the simulator's own walkers and
a robot that moves exactly as commanded. No rendering and no physics engine.

- **The robot.** Each 100-ms frame takes one requested command (forward, lateral, yaw). It passes the continuous block's
  bounds and its rate limit once per frame (`locomotion.safety.BLOCKS['continuous']`, as `sim.session` applies them) and
  is integrated in the body frame every `DT_S` (2 ms). Tracking is perfect, which the real robot's is not.
- **The walkers** are `sim.obstacles`' (polite, not yielding, reactive), built from a placement as `ObstacleField`
  builds them and stepped every 2 ms against the robot's position (reactive ones also its yaw and the other walkers),
  before the robot moves, as in a mission.
- **Contact:** the body outline (`data.head_labels.OUTLINE_HALF_M`, 0.70 by 0.32 m) touching a wall box or a walker's
  cylinder; checked every 10 ms, it ends the episode.

`Geometry` holds the walls and the distance functions; `World` the state and `step`; `World.snapshot` what a planner's
model starts from (the walkers are copies, random generators included).
"""
import copy
import math

import numpy as np

from go2nav.data.head_labels import OUTLINE_HALF_M
from go2nav.locomotion.safety import BLOCKS
from go2nav.sim import obstacles as moving

DT_S, FRAME_S = .002, .1
SUBSTEPS = round(FRAME_S/DT_S)
CONTACT_EVERY = 5                      # substeps between contact checks (10 ms)


def outline_points(count_long=7, count_short=3):
    """Points round the body outline in the body frame (n, 2), at most 0.12 m apart."""
    a, b = OUTLINE_HALF_M
    xs, ys = np.linspace(-a, a, count_long), np.linspace(-b, b, count_short)
    points = [(x, s*b) for x in xs for s in (1, -1)]+[(s*a, y) for y in ys[1:-1] for s in (1, -1)]
    return np.array(points, float)


OUTLINE = outline_points()


def rectangle_gap(points, xy, yaw, half=OUTLINE_HALF_M):
    """The distance from points (..., 2) to the rectangle of half-sizes `half` at xy (..., 2) and yaw (...) (negative
    inside); the leading shapes broadcast."""
    d = np.asarray(points, float)-np.asarray(xy, float)
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.abs(np.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]), -1))
    half = np.asarray(half, float)
    outside = np.linalg.norm(np.maximum(local-half, 0.), axis=-1)
    inside = (local <= half).all(-1)
    return np.where(inside, -np.min(half-local, axis=-1), outside)


def body_points(xy, yaw):
    """The outline's world points (..., n, 2) at xy (..., 2) and yaw (...)."""
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    x = np.asarray(xy)[..., None, 0]+c*OUTLINE[:, 0]-s*OUTLINE[:, 1]
    y = np.asarray(xy)[..., None, 1]+s*OUTLINE[:, 0]+c*OUTLINE[:, 1]
    return np.stack((x, y), -1)


class Geometry:
    """The walls of a maze spec (axis-aligned boxes) and its passages."""

    def __init__(self, spec):
        boxes = spec['geometry']['wall_boxes']
        self.centres = np.array([b['centre_xyz'][:2] for b in boxes], float)
        self.halves = np.array([b['size_xyz'][:2] for b in boxes], float)/2
        self.low, self.high = (self.centres-self.halves).min(0), (self.centres+self.halves).max(0)
        layout = spec.get('evaluation_layout')
        self.pitch = moving.pitch_of(spec)
        self.neighbours = moving.passages(spec) if layout else {}

    def wall_distance(self, points):
        """(...,) the distance of each point (..., 2) to the nearest wall's surface (0 inside one)."""
        outside = np.maximum(np.abs(np.asarray(points, float)[..., None, :]-self.centres)-self.halves, 0.)
        return np.sqrt((outside**2).sum(-1)).min(-1)

    def visible(self, a, b, step=.04):
        """Whether the straight line from a to b crosses no wall (sampled every `step`)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(2, int(np.linalg.norm(b-a)/step)+1)
        points = a+(b-a)*np.linspace(0., 1., count)[:, None]
        return bool((self.wall_distance(points) > 0.).all())

    def clearance(self, xy, yaw, walkers=()):
        """The body outline's clearance at one pose: to the walls (through the outline's points) and to the walkers'
        cylinders (exact)."""
        value = float(self.wall_distance(body_points(np.asarray(xy, float), np.asarray(yaw, float))).min())
        if len(walkers):
            value = min(value, float(rectangle_gap(np.asarray(walkers, float), xy, yaw).min())-moving.RADIUS_M)
        return value


def make_walkers(spec, placement):
    """The walk states of a placement (`sim.obstacles.place_obstacles`), as `ObstacleField` builds them."""
    neighbours, pitch = moving.passages(spec), moving.pitch_of(spec)
    out = []
    for o in placement:
        args = (neighbours, tuple(o['start_cell']), o['speed_m_s'], np.random.default_rng(o['walk_seed']))
        if o.get('behaviour') == 'reactive':
            out.append(moving.ReactiveWalker(*args, patience=float(o['patience_s']), yields=bool(o['yields']), pitch=pitch))
        else:
            out.append(moving.Walker(*args, polite=o.get('polite', True), pitch=pitch))
    return out


def step_walkers(walkers, dt, xy, yaw):
    """Advance every walker by dt against the robot's pose, as `ObstacleField.update` does."""
    for k, walker in enumerate(walkers):
        if isinstance(walker, moving.ReactiveWalker):
            walker.step(dt, xy, yaw, [w.xy for j, w in enumerate(walkers) if j != k])
        else:
            walker.step(dt, xy)


def limit(requested, previous):
    """One frame's executed command: the request through the continuous block's bounds and rate limit."""
    lower, upper, delta = BLOCKS['continuous']
    return np.clip(np.clip(np.asarray(requested, np.float32), lower, upper), previous-delta, previous+delta)


def waiting_on(walker, xy):
    """Whether a walker is held up by the robot now: stopped for it (polite, not yielding) or blocked (reactive)."""
    if walker.pause_left > 0:
        return False
    if isinstance(walker, moving.ReactiveWalker):
        return walker.blocked_for > 0
    return bool(walker.yields_to(xy))


class World:
    """See the module docstring. `xy`, `yaw`: the robot's pose; `executed`: the command of the last frame; `contact`:
    None, or 'wall' or 'walker' once it has touched (then `step` does nothing)."""

    def __init__(self, spec, start_se2, placement=()):
        self.geometry = Geometry(spec)
        self.walkers = make_walkers(spec, placement)
        self.xy, self.yaw = np.array(start_se2[:2], float), float(start_se2[2])
        self.executed = np.zeros(3, np.float32)
        self.frame, self.contact, self.path_m, self.minimum_clearance = 0, None, 0., math.inf
        self.waiting_frames = 0
        self.walkers_velocity = np.zeros((len(self.walkers), 2))

    def walkers_xy(self):
        return np.array([w.xy for w in self.walkers], float).reshape(-1, 2)

    def clearance(self):
        return self.geometry.clearance(self.xy, self.yaw, self.walkers_xy())

    def step(self, requested):
        """One 100-ms frame under a requested command; returns the executed one."""
        if self.contact is not None:
            return self.executed
        self.executed = limit(requested, self.executed)
        vx, vy, wz = (float(v) for v in self.executed)
        before = self.walkers_xy()
        for k in range(SUBSTEPS):
            step_walkers(self.walkers, DT_S, self.xy, self.yaw)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            move = np.array([c*vx-s*vy, s*vx+c*vy])*DT_S
            self.xy, self.yaw = self.xy+move, self.yaw+wz*DT_S
            self.path_m += float(np.linalg.norm(move))
            if (k+1) % CONTACT_EVERY == 0:
                walls = float(self.geometry.wall_distance(body_points(self.xy, np.asarray(self.yaw))).min())
                walkers = float(rectangle_gap(self.walkers_xy(), self.xy, self.yaw).min())-moving.RADIUS_M \
                    if self.walkers else math.inf
                self.minimum_clearance = min(self.minimum_clearance, walls, walkers)
                if min(walls, walkers) <= 0.:
                    self.contact = 'wall' if walls <= walkers else 'walker'
                    break
        self.frame += 1
        self.walkers_velocity = (self.walkers_xy()-before)/FRAME_S
        self.waiting_frames += int(any(waiting_on(w, self.xy) for w in self.walkers))
        return self.executed

    def snapshot(self):
        """What a planner's model starts from: the robot's pose and last command, copies of the walkers, and their
        positions and velocities over the last frame (what the simpler models are given)."""
        return dict(xy=self.xy.copy(), yaw=self.yaw, executed=self.executed.copy(), walkers=copy.deepcopy(self.walkers),
                    walkers_xy=self.walkers_xy(), walkers_velocity=self.walkers_velocity.copy())
