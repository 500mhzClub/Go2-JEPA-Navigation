"""Moving obstacles (docs/dynamic/PLAN.md, P2): scripted kinematic cylinders that walk the maze's passages.

**Placement** (`place_obstacles`), seeded per set, maze, episode and count: each obstacle starts at the centre of a cell
at least `MIN_START_HOPS` passages from the home cell and not the beacon cell (distinct cells), with a walking speed
drawn from `SPEED_M_S` and its own walk seed. Obstacles are cylinders `RADIUS_M` across and `HEIGHT_M` tall, so the
depth layer (3-65 cm above the floor) sees them.

**Walk** (`ObstacleField.update`, every physics step after settling): from cell centre to a neighbouring cell centre
along a passage, never straight back unless the cell is a dead end; at each cell a pause of `PAUSE_S` with probability
`PAUSE_PROBABILITY`. **Polite** (the first condition): an obstacle stops while the robot's base is within `YIELD_M`
ahead of it (in front of its heading and within `YIELD_HALF_WIDTH_M` of its line of travel), and after `GIVE_UP_S`
stopped it turns back to the cell it came from. Every draw comes from the obstacle's own generator, so a mission
replays exactly.

The obstacles are fixed rigid entities moved by `set_pos` (kinematic: they push the robot and are not pushed). Any
contact with one is a disallowed contact (`go2nav.sim.contacts`); the evaluator attributes it from the trace.
"""
from collections import deque
import copy
from dataclasses import dataclass, field
import hashlib

import numpy as np

from go2nav.mazes.grid import PITCH_M

RADIUS_M, HEIGHT_M = .175, .9
SPEED_M_S = (.15, .35)
MIN_START_HOPS = 3
PAUSE_PROBABILITY, PAUSE_S = .3, (1., 4.)
YIELD_M, YIELD_HALF_WIDTH_M, GIVE_UP_S = .8, .6, 3.
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


def cell_of(xy):
    return tuple(int(round(v/PITCH_M)) for v in xy[:2])


def place_obstacles(spec, episode, count, seed, speed=SPEED_M_S):
    """The obstacles of an episode: [dict(start_cell, speed_m_s, walk_seed, radius_m, height_m, colour)], speeds drawn
    from `speed` (the same draws for any range, so a range changes only the speeds)."""
    rng = np.random.default_rng(seed)
    neighbours = passages(spec)
    hops = hops_from(neighbours, cell_of(episode['home_se2_world']))
    beacon = cell_of(episode['beacon_xy_world'])
    candidates = sorted(c for c, h in hops.items() if h >= MIN_START_HOPS and c != beacon)
    if len(candidates) < count:
        raise ValueError(f'{len(candidates)} cells can hold an obstacle, {count} asked')
    starts = [candidates[i] for i in rng.choice(len(candidates), size=count, replace=False)]
    return [dict(start_cell=list(c), speed_m_s=float(rng.uniform(*speed)), walk_seed=int(rng.integers(2**62)),
                 radius_m=RADIUS_M, height_m=HEIGHT_M, colour=list(COLOURS[k % len(COLOURS)]))
            for k, c in enumerate(starts)]


@dataclass
class Walker:
    """One obstacle's walk state (see the module docstring)."""
    neighbours: dict
    cell: tuple
    speed: float
    rng: np.random.Generator
    came_from: tuple = None
    target: tuple = None
    pause_left: float = 0.
    blocked_for: float = 0.
    xy: np.ndarray = field(default=None)

    def __post_init__(self):
        self.xy = np.asarray(self.cell, float)*PITCH_M
        self._choose()

    def _choose(self):
        options = [n for n in self.neighbours[self.cell] if n != self.came_from] or list(self.neighbours[self.cell])
        self.target = options[int(self.rng.integers(len(options)))]
        if self.rng.random() < PAUSE_PROBABILITY:
            self.pause_left = float(self.rng.uniform(*PAUSE_S))

    def heading(self):
        d = np.asarray(self.target, float)*PITCH_M-self.xy
        n = np.linalg.norm(d)
        return d/n if n > 1e-9 else np.zeros(2)

    def yields_to(self, robot_xy):
        if robot_xy is None:
            return False
        h, r = self.heading(), np.asarray(robot_xy[:2], float)-self.xy
        ahead, across = float(h@r), abs(float(h[0]*r[1]-h[1]*r[0]))
        return 0. < ahead < YIELD_M and across < YIELD_HALF_WIDTH_M

    def step(self, dt, robot_xy):
        """Advance by dt seconds; returns the new position (x, y)."""
        if self.pause_left > 0:
            self.pause_left = max(0., self.pause_left-dt)
            return self.xy
        if self.yields_to(robot_xy):
            self.blocked_for += dt
            if self.blocked_for >= GIVE_UP_S:   # turn back to the cell it left
                self.cell, self.target, self.blocked_for = self.target, self.cell, 0.
            return self.xy
        self.blocked_for = 0.
        goal = np.asarray(self.target, float)*PITCH_M
        d = goal-self.xy
        distance, travel = float(np.linalg.norm(d)), self.speed*dt
        if travel >= distance:
            self.xy = goal
            self.came_from, self.cell = self.cell, self.target
            self._choose()
        else:
            self.xy = self.xy+d/distance*travel
        return self.xy


class ObstacleField:
    """The obstacles of one mission in the Genesis scene: `entities` are the cylinders (built by `sim.scene`)."""

    def __init__(self, spec, placement, entities):
        neighbours = passages(spec)
        self.placement, self.entities = placement, entities
        self.walkers = [Walker(neighbours, tuple(o['start_cell']), o['speed_m_s'], np.random.default_rng(o['walk_seed']))
                        for o in placement]
        self.heights = [o['height_m'] for o in placement]
        self.trace = []    # per physics step after settling: (timestamp_s, [(x, y), ...])

    def initial_positions(self):
        return [(*np.asarray(o['start_cell'], float)*PITCH_M, o['height_m']/2) for o in self.placement]

    def update(self, dt, robot_xy, timestamp_s):
        """Move every obstacle by dt (before a physics step) and record the positions."""
        positions = []
        for walker, entity, height in zip(self.walkers, self.entities, self.heights):
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
