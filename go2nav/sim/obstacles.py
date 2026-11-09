"""Moving obstacles (docs/dynamic/PLAN.md, P2): scripted kinematic cylinders that walk the maze's passages.

**Placement** (`place_obstacles`), seeded per set, maze, episode and count: each obstacle starts at the centre of a cell
at least `MIN_START_HOPS` passages from the home cell and not the beacon cell (distinct cells), with a walking speed
drawn from `SPEED_M_S` and its own walk seed. Obstacles are cylinders `RADIUS_M` across and `HEIGHT_M` tall, so the
depth layer (3-65 cm above the floor) sees them.

**Walk** (`ObstacleField.update`, every physics step after settling): from cell centre to a neighbouring cell centre
along a passage, never straight back unless the cell is a dead end; at each cell a pause of `PAUSE_S` with probability
`PAUSE_PROBABILITY`. **Polite** (the first condition): an obstacle stops while the robot's base is within `YIELD_M`
ahead of it (in front of its heading and within `YIELD_HALF_WIDTH_M` of its line of travel), and after `GIVE_UP_S`
stopped it turns back to the cell it came from. **Non-polite** (`polite=False`, a harder condition and a share of
the corpus v2 units): it walks on until it is about to touch the robot (heading towards the robot's centre, from any
side, with its edge within `ASSERTIVE_GAP_M` of the robot's reach `ROBOT_REACH_M`), then waits there for the way to
clear and never turns back. **Reactive** (`behaviour: 'reactive'`, docs/dynamic/PLAN.md P13 item 3): it walks the
same passages but steers round the robot (an oriented rectangle, `ROBOT_HALF_LENGTH_M` by `ROBOT_HALF_WIDTH_M`, the
legs included) and round the other walkers. Every `DECIDE_S` it keeps to the passage's centre if that is clear for
`LOOKAHEAD_M`, else takes the nearest clear lane across the passage, else waits: a yielding one (`yields`) waits
`YIELD_GAP_M` short of the robot, the others `PASS_GAP_M`. After waiting its `patience_s` it turns back to the cell it
left. It never steps closer than its gap to the robot or another walker. Every draw comes from the obstacle's own
generator, so a mission replays exactly.

The obstacles are fixed rigid entities moved by `set_pos` (kinematic: they push the robot and are not pushed). Any
contact with one is a disallowed contact (`go2nav.sim.contacts`); the evaluator attributes it from the trace.
"""
from collections import deque
import copy
from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

from go2nav.mazes.grid import PITCH_M, WALL_THICKNESS_M

RADIUS_M, HEIGHT_M = .175, .9
SPEED_M_S = (.15, .35)
MIN_START_HOPS = 3
PAUSE_PROBABILITY, PAUSE_S = .3, (1., 4.)
YIELD_M, YIELD_HALF_WIDTH_M, GIVE_UP_S = .8, .6, 3.
ROBOT_REACH_M, ASSERTIVE_GAP_M = .40, .05   # the body's half-diagonal is about 0.38 m
REACTIVE_PATIENCE_S, REACTIVE_YIELD_SHARE = (2., 20.), .5
ROBOT_HALF_LENGTH_M, ROBOT_HALF_WIDTH_M = .36, .20   # the body with its legs, as a reactive walker passes it
PASS_GAP_M, YIELD_GAP_M = .05, .40
LOOKAHEAD_M, DECIDE_S, LANE_SAMPLES, LANE_MARGIN_M, LANE_BLEND_M = 1.5, .1, 17, .02, .4
LATERAL_FRACTION = .6                                # lateral speed as a share of the walking speed
COLOURS = ((.80, .35, .25), (.25, .45, .80), (.85, .70, .20), (.35, .65, .35))


def placement_seed(set_name, maze, episode, count, series='v1'):
    """sha256 of "obstacles-<series>:<set>:<maze>:<episode>:<count>", first 8 bytes (big-endian). Series v1 is the
    development series; a held-out evaluation uses a series not used before."""
    key = f'obstacles-{series}:{set_name}:{int(maze)}:{int(episode)}:{int(count)}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def passages(spec):
    """The layout's cells and their neighbours (sorted)."""
    layout = spec['evaluation_layout']
    neighbours = {tuple(c): [] for c in layout['cells']}
    for a, b in layout['edges']:
        neighbours[tuple(a)].append(tuple(b))
        neighbours[tuple(b)].append(tuple(a))
    return {c: sorted(n) for c, n in neighbours.items()}


def hops_from(neighbours, start):
    seen, queue = {start: 0}, deque([start])
    while queue:
        c = queue.popleft()
        for n in neighbours[c]:
            if n not in seen:
                seen[n] = seen[c]+1
                queue.append(n)
    return seen


def pitch_of(spec):
    """The layout's cell pitch (`evaluation_layout.pitch_m`; families other than the tree vary it, PLAN P13)."""
    return float(spec.get('evaluation_layout', {}).get('pitch_m', PITCH_M))


def cell_of(xy, pitch=PITCH_M):
    return tuple(int(round(v/pitch)) for v in xy[:2])


def place_obstacles(spec, episode, count, seed, speed=SPEED_M_S, polite=True, reactive=False):
    """The obstacles of an episode: [dict(start_cell, speed_m_s, walk_seed, radius_m, height_m, colour)], speeds drawn
    from `speed` (the same draws for any range, so a range changes only the speeds); non-polite obstacles also carry
    `polite: False`. Reactive ones (`reactive`) carry `behaviour: 'reactive'`, `patience_s` and `yields`, drawn from a
    generator of their own walk seed, so the start cells, speeds and walk seeds are those of the other behaviours."""
    if reactive and not polite:
        raise ValueError('reactive walkers draw their own yielding (yields); polite must stay on')
    rng = np.random.default_rng(seed)
    neighbours = passages(spec)
    pitch = pitch_of(spec)
    hops = hops_from(neighbours, cell_of(episode['home_se2_world'], pitch))
    beacon = cell_of(episode['beacon_xy_world'], pitch)
    candidates = sorted(c for c, h in hops.items() if h >= MIN_START_HOPS and c != beacon)
    if len(candidates) < count:
        raise ValueError(f'{len(candidates)} cells can hold an obstacle, {count} asked')
    starts = [candidates[i] for i in rng.choice(len(candidates), size=count, replace=False)]
    placed = [dict(start_cell=list(c), speed_m_s=float(rng.uniform(*speed)), walk_seed=int(rng.integers(2**62)),
                   radius_m=RADIUS_M, height_m=HEIGHT_M, colour=list(COLOURS[k % len(COLOURS)]),
                   **({} if polite else dict(polite=False)))
              for k, c in enumerate(starts)]
    if reactive:
        for o in placed:
            g = np.random.default_rng([o['walk_seed'], 1])
            o.update(behaviour='reactive', patience_s=float(g.uniform(*REACTIVE_PATIENCE_S)),
                     yields=bool(g.random() < REACTIVE_YIELD_SHARE))
    return placed


def robot_gap(points, robot_xy, robot_yaw):
    """(n,) distance from each point (n, 2) to the robot's outline: the oriented rectangle of `ROBOT_HALF_LENGTH_M` by
    `ROBOT_HALF_WIDTH_M` (negative inside), or without a yaw the disc of `ROBOT_REACH_M`."""
    d = np.asarray(points, float).reshape(-1, 2)-np.asarray(robot_xy[:2], float)
    if robot_yaw is None:
        return np.linalg.norm(d, axis=1)-ROBOT_REACH_M
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    local = np.abs(np.stack((c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1]), axis=1))
    half = np.array([ROBOT_HALF_LENGTH_M, ROBOT_HALF_WIDTH_M])
    outside = np.linalg.norm(np.maximum(local-half, 0.), axis=1)
    return np.where((local <= half).all(1), -np.min(half-local, axis=1), outside)


@dataclass
class Walker:
    """One obstacle's walk state (see the module docstring)."""
    neighbours: dict
    cell: tuple
    speed: float
    rng: np.random.Generator
    came_from: tuple = None
    target: tuple = None
    polite: bool = True
    pause_left: float = 0.
    blocked_for: float = 0.
    xy: np.ndarray = field(default=None)
    pitch: float = PITCH_M

    def __post_init__(self):
        self.xy = np.asarray(self.cell, float)*self.pitch
        self._choose()

    def _choose(self):
        options = [n for n in self.neighbours[self.cell] if n != self.came_from] or list(self.neighbours[self.cell])
        self.target = options[int(self.rng.integers(len(options)))]
        if self.rng.random() < PAUSE_PROBABILITY:
            self.pause_left = float(self.rng.uniform(*PAUSE_S))

    def heading(self):
        d = np.asarray(self.target, float)*self.pitch-self.xy
        n = np.linalg.norm(d)
        return d/n if n > 1e-9 else np.zeros(2)

    def yields_to(self, robot_xy):
        if robot_xy is None:
            return False
        h, r = self.heading(), np.asarray(robot_xy[:2], float)-self.xy
        ahead, across = float(h@r), abs(float(h[0]*r[1]-h[1]*r[0]))
        if not self.polite:
            return ahead > 0. and float(np.linalg.norm(r)) < RADIUS_M+ROBOT_REACH_M+ASSERTIVE_GAP_M
        return 0. < ahead < YIELD_M and across < YIELD_HALF_WIDTH_M

    def step(self, dt, robot_xy):
        """Advance by dt seconds; returns the new position (x, y)."""
        if self.pause_left > 0:
            self.pause_left = max(0., self.pause_left-dt)
            return self.xy
        if self.yields_to(robot_xy):
            self.blocked_for += dt
            if self.polite and self.blocked_for >= GIVE_UP_S:   # turn back to the cell it left
                self.cell, self.target, self.blocked_for = self.target, self.cell, 0.
            return self.xy
        self.blocked_for = 0.
        goal = np.asarray(self.target, float)*self.pitch
        d = goal-self.xy
        distance, travel = float(np.linalg.norm(d)), self.speed*dt
        if travel >= distance:
            self.xy = goal
            self.came_from, self.cell = self.cell, self.target
            self._choose()
        else:
            self.xy = self.xy+d/distance*travel
        return self.xy


@dataclass
class ReactiveWalker(Walker):
    """A reactive walker's walk state (see the module docstring)."""
    patience: float = 10.
    yields: bool = False
    lane: float = 0.          # the lane it steers to: metres left of the passage's centre line, along its heading
    decide_left: float = 0.
    clear: bool = True        # whether its last decision found a clear lane
    turned: int = 0           # turnarounds after waiting its patience

    def _frame(self):
        a, b = np.asarray(self.cell, float)*self.pitch, np.asarray(self.target, float)*self.pitch
        length = float(np.linalg.norm(b-a))
        u = (b-a)/length if length > 1e-9 else np.array([1., 0.])
        return a, u, np.array([-u[1], u[0]]), length

    @staticmethod
    def _gap(points, robot_xy, robot_yaw, others):
        """(n,) the smallest gap from the walker's disc at each point to the robot and to the other walkers."""
        points = np.asarray(points, float).reshape(-1, 2)
        gap = np.full(len(points), np.inf)
        if robot_xy is not None:
            gap = np.minimum(gap, robot_gap(points, robot_xy, robot_yaw)-RADIUS_M)
        for o in others:
            gap = np.minimum(gap, np.linalg.norm(points-np.asarray(o, float)[None], axis=1)-2*RADIUS_M)
        return gap

    def _decide(self, robot_xy, robot_yaw, others):
        """(lane, clear): the centre if it is clear for `LOOKAHEAD_M`, else the nearest clear lane, else (where it is, False)."""
        a, u, n, length = self._frame()
        s, off = float((self.xy-a)@u), float((self.xy-a)@n)
        ahead = np.arange(.05, LOOKAHEAD_M+1e-9, .05)
        ahead = ahead[s+ahead <= length+.3]                       # past the next cell's centre it turns
        if not len(ahead):
            return 0., True
        bound = (self.pitch-WALL_THICKNESS_M)/2-RADIUS_M-LANE_MARGIN_M
        blend = np.clip(ahead/LANE_BLEND_M, 0., 1.)

        def clear(lane):
            points = a+np.outer(s+ahead, u)+np.outer(off+(lane-off)*blend, n)
            return bool((self._gap(points, robot_xy, robot_yaw, others) >= PASS_GAP_M).all())
        if clear(0.):
            return 0., True
        for lane in sorted(np.linspace(-bound, bound, LANE_SAMPLES), key=lambda v: (abs(v-off), abs(v))):
            if clear(float(lane)):
                return float(lane), True
        return off, False

    def step(self, dt, robot_xy, robot_yaw=None, others=()):
        """Advance by dt seconds; returns the new position (x, y)."""
        if self.pause_left > 0:
            self.pause_left = max(0., self.pause_left-dt)
            return self.xy
        self.decide_left -= dt
        if self.decide_left <= 0:
            self.decide_left = DECIDE_S
            self.lane, self.clear = self._decide(robot_xy, robot_yaw, others)
        a, u, n, length = self._frame()
        off = float((self.xy-a)@n)
        side = LATERAL_FRACTION*self.speed*dt
        lateral, forward = float(np.clip(self.lane-off, -side, side)), self.speed*dt
        limit = YIELD_GAP_M if self.yields and not self.clear else PASS_GAP_M
        now = float(self._gap(self.xy, robot_xy, robot_yaw, others)[0])
        advanced = False
        for f, l in ((forward, lateral), (0., lateral), (forward, 0.)):
            if f == 0. and l == 0.:
                continue
            new = self.xy+f*u+l*n
            gap = float(self._gap(new, robot_xy, robot_yaw, others)[0])
            if gap >= limit or gap > now:                          # never closer than its gap; away is always allowed
                self.xy, advanced = new, f > 0.
                break
        if advanced:
            self.blocked_for = 0.
            if float((self.xy-a)@u) >= length:                     # reached the next cell
                self.came_from, self.cell = self.cell, self.target
                self._choose()
                self.lane, self.decide_left = 0., 0.
        else:
            self.blocked_for += dt
            if self.blocked_for >= self.patience:                  # turn back to the cell it left
                self.cell, self.target, self.blocked_for = self.target, self.cell, 0.
                self.lane, self.decide_left, self.turned = 0., 0., self.turned+1
        return self.xy


class ObstacleField:
    """The obstacles of one mission in the Genesis scene: `entities` are the cylinders (built by `sim.scene`)."""

    def __init__(self, spec, placement, entities):
        neighbours = passages(spec)
        self.placement, self.entities, self.pitch = placement, entities, pitch_of(spec)

        def walker(o):
            args = (neighbours, tuple(o['start_cell']), o['speed_m_s'], np.random.default_rng(o['walk_seed']))
            if o.get('behaviour') == 'reactive':
                return ReactiveWalker(*args, patience=float(o['patience_s']), yields=bool(o['yields']), pitch=self.pitch)
            return Walker(*args, polite=o.get('polite', True), pitch=self.pitch)
        self.walkers = [walker(o) for o in placement]
        self.heights = [o['height_m'] for o in placement]
        self.trace = []    # per physics step after settling: (timestamp_s, [(x, y), ...])

    def initial_positions(self):
        return [(*np.asarray(o['start_cell'], float)*self.pitch, o['height_m']/2) for o in self.placement]

    def update(self, dt, robot_xy, timestamp_s, robot_yaw=None):
        """Move every obstacle by dt (before a physics step) and record the positions. Reactive walkers also see the
        robot's yaw and the other walkers."""
        positions = []
        for k, (walker, entity, height) in enumerate(zip(self.walkers, self.entities, self.heights)):
            if isinstance(walker, ReactiveWalker):
                others = [w.xy for j, w in enumerate(self.walkers) if j != k]
                x, y = walker.step(dt, robot_xy, robot_yaw, others)
            else:
                x, y = walker.step(dt, robot_xy)
            entity.set_pos(np.array([[x, y, height/2]], np.float32), envs_idx=[0], zero_velocity=True)
            positions.append((float(x), float(y)))
        self.trace.append((float(timestamp_s), positions))
        return positions

    def state(self):
        """The walk state (for simulator snapshots; the cylinders' poses are solver state)."""
        return copy.deepcopy(self.walkers), len(self.trace)

    def restore(self, state):
        walkers, length = state
        self.walkers, self.trace = copy.deepcopy(walkers), self.trace[:length]

    def arrays(self):
        """The trace as arrays: timestamp_s (n,), xy (n, obstacles, 2)."""
        if not self.trace:
            return dict(timestamp_s=np.zeros(0), xy=np.zeros((0, len(self.walkers), 2)))
        return dict(timestamp_s=np.array([t for t, _ in self.trace]), xy=np.array([p for _, p in self.trace]))
