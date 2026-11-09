"""A sampling planner over command tapes with an intrinsic cost (docs/dynamic/PLAN.md P17), first on the true model.

Every `REPLAN_FRAMES` frames `Planner.plan` proposes tapes of requested commands over its horizon, predicts each one's
outcome with its model, and returns the cheapest tape's first frames.

**Proposals.** A prior over tapes, built from the layout's distances (`planning.fields`) and holding nothing about
walkers:
- make for the target, forwards only or backing where that turns less, at `SPEEDS`;
- hold;
- walk to the centre of a cell within `HOPS` passages, wait there one of `WAITS_S`, then make for the target (forwards
  only, or backing);
- Gaussian perturbations (at knots `KNOT_FRAMES` apart) of the last plan and of the first proposal.
A tape that makes for a point follows the steepest descent of the point's distance field over a ring `RING_M` round the
robot, turning towards it and driving as its heading allows; within `NEAR_M` of the target it closes on it with forward
and lateral commands.

**Model** (`TrueModel`; P18 replaces the robot's part with a learned one): the robot's path is the tape's, through the
continuous block's limits, at 100 ms; the walkers are copies of the world's, stepped against that path, random
generators included, so the model knows their future draws.

**Cost,** in seconds:
- time: 0.1 s for each step further than `ARRIVE_M` from the target;
- clearance: `CLEARANCE_RATE` seconds a second as the body outline's clearance falls from `SOFT_M` to `FLOOR_M`; a tape
  that goes under `FLOOR_M` is discarded (`DISCARD`, less for a later breach, so that if every tape breaches the latest
  is taken);
- progress: the distance left from the tape's end over `V_REF`. `terminal='static'`: through the walls alone.
  `terminal='walkers'`: with the walkers, where the tape leaves them, as obstacles; a sealed way costs `SEALED_S` more.
"""
import copy

import numpy as np

from go2nav.locomotion.safety import BLOCKS
from go2nav.planning.fields import Fields
from go2nav.sim import kinematic as kin
from go2nav.sim import obstacles as moving

FRAME_S = kin.FRAME_S
REPLAN_FRAMES, KNOT_FRAMES = 4, 8
V_REF, ARRIVE_M, NEAR_M = .3, .03, .4
FLOOR_M, SOFT_M, CLEARANCE_RATE, SEALED_S, DISCARD = .08, .20, 2., 200., 1e4
HOPS, WAITS_S, SPEEDS, REVERSE_M_S = 3, (2., 5., 10., 20.), (.4, .25, .12), .25
RING_M, RING, AT_CELL_M, TURN_GAIN, ALIGNED_RAD = .3, 16, .15, 2., .8
GAUSSIANS, SIGMA = 16, (.12, .06, .25)
DIRECTIONS = np.stack([np.cos(np.arange(RING)*2*np.pi/RING), np.sin(np.arange(RING)*2*np.pi/RING)], 1)
ANGLES = np.arange(RING)*2*np.pi/RING


def wrap(angle):
    return (angle+np.pi) % (2*np.pi)-np.pi


def limit(requested, previous):
    """Executed commands (K, 3) of requests (K, 3) after the previous executed ones (K, 3)."""
    lower, upper, delta = BLOCKS['continuous']
    return np.clip(np.clip(requested, lower, upper), previous-delta, previous+delta)


def advance(xy, yaw, executed):
    """One 100-ms step of poses (K, 2), (K,) under executed commands (K, 3), turning through the step's middle."""
    middle = yaw+executed[:, 2]*FRAME_S/2
    c, s = np.cos(middle), np.sin(middle)
    step = np.stack((c*executed[:, 0]-s*executed[:, 1], s*executed[:, 0]+c*executed[:, 1]), 1)*FRAME_S
    return xy+step, yaw+executed[:, 2]*FRAME_S


def integrate(start, requested):
    """The poses a tape leads through: `start` (xy, yaw, executed) and requests (K, T, 3) to xy (K, T, 2), yaw (K, T)."""
    count, frames = requested.shape[:2]
    xy, yaw, executed = np.tile(start['xy'], (count, 1)), np.full(count, start['yaw']), np.tile(start['executed'], (count, 1))
    path, heading = np.empty((count, frames, 2)), np.empty((count, frames))
    for t in range(frames):
        executed = limit(requested[:, t], executed)
        xy, yaw = advance(xy, yaw, executed)
        path[:, t], heading[:, t] = xy, yaw
    return path, heading


class TrueModel:
    """The walkers' part of the model: copies of the world's walkers stepped against a robot path at 100 ms."""

    def walkers(self, snapshot, xy, yaw, frames):
        """Walker positions (K, T, n, 2) along paths xy (K, T, 2), yaw (K, T), each stepped for its first `frames[k]`
        frames (the rest repeat the last position)."""
        count, horizon = xy.shape[:2]
        out = np.zeros((count, horizon, len(snapshot['walkers']), 2))
        if not snapshot['walkers']:
            return out
        for k in range(count):
            walkers = copy.deepcopy(snapshot['walkers'])
            at, heading = snapshot['xy'], snapshot['yaw']
            for t in range(int(frames[k])):
                kin.step_walkers(walkers, FRAME_S, at, heading)      # against the pose the frame starts from
                out[k, t] = [w.xy for w in walkers]
                at, heading = xy[k, t], float(yaw[k, t])
            out[k, int(frames[k]):] = out[k, max(int(frames[k])-1, 0)]
        return out


class Planner:
    def __init__(self, geometry, *, horizon_s, terminal='static', seed=0, model=None):
        if terminal not in ('static', 'walkers'):
            raise ValueError("terminal is 'static' or 'walkers'")
        self.geometry, self.fields, self.terminal = geometry, Fields(geometry), terminal
        self.frames = int(round(horizon_s/FRAME_S))
        self.model = model or TrueModel()
        self.rng = np.random.default_rng(seed)
        self.previous, self.target = None, None
        self.cells = {cell: np.asarray(cell, float)*geometry.pitch for cell in geometry.neighbours}

    def near_cells(self, xy):
        """The cells within `HOPS` passages of the robot's (the nearest cell's centre by distance)."""
        if not self.cells:
            return []
        here = min(self.cells, key=lambda c: float(np.linalg.norm(self.cells[c]-xy)))
        hops = moving.hops_from(self.geometry.neighbours, here)
        return sorted(c for c, h in hops.items() if h <= HOPS)

    def follow(self, start, target, programmes):
        """The tapes of `programmes`, each (cell or None, wait frames, backing allowed, speed): walk to the cell's
        centre if one is given, wait, then make for the target. Returns requests (K, T, 3)."""
        count, frames = len(programmes), self.frames
        stack = [self.fields.static(target)]
        index = {}
        for cell, *_ in programmes:
            if cell is not None and cell not in index:
                index[cell] = len(stack)
                stack.append(self.fields.static(self.cells[cell]))
        stack = np.stack(stack)
        first = np.array([index[p[0]] if p[0] is not None else 0 for p in programmes])
        waits = np.array([p[1] for p in programmes], float)
        backing = np.array([p[2] for p in programmes], bool)
        speed = np.array([p[3] for p in programmes], float)
        mode = np.where(first > 0, 0, np.where(waits > 0, 1, 2))      # 0 to the cell, 1 waiting, 2 to the target
        xy, yaw = np.tile(start['xy'], (count, 1)), np.full(count, float(start['yaw']))
        executed = np.tile(start['executed'], (count, 1)).astype(float)
        requested, rows = np.zeros((count, frames, 3)), np.arange(count)
        target = np.asarray(target, float)
        for t in range(frames):
            field = np.where(mode == 0, first, 0)
            node = self.fields.index(xy)
            here = np.full(count, np.inf)
            for dx in (-1, 0, 1):                                    # the least over the 3 x 3 nodes, as `Fields.value`
                for dy in (-1, 0, 1):
                    j = np.clip(node+(dx, dy), 0, np.array(self.fields.shape)-1)
                    here = np.minimum(here, stack[field, j[:, 0], j[:, 1]])
            ring = self.fields.index(xy[:, None]+RING_M*DIRECTIONS)
            values = stack[field[:, None], ring[..., 0], ring[..., 1]]
            best = values.argmin(1)
            descends = values[rows, best] < here
            mode = np.where((mode == 0) & (here < AT_CELL_M), 1, mode)
            error = wrap(ANGLES[best]-yaw)
            back = backing & (np.abs(error) > 1.75)
            turn = np.where(back, wrap(error-np.pi), error)
            aligned = np.abs(turn) < ALIGNED_RAD
            forward = np.where(back, -REVERSE_M_S, speed)*np.cos(turn)**2*aligned
            command = np.stack((forward, np.zeros(count), np.clip(TURN_GAIN*turn, -.8, .8)), 1)
            command[~descends | ~np.isfinite(here)] = 0.
            offset = target-xy
            distance = np.linalg.norm(offset, axis=1)
            close = (mode == 2) & (distance < NEAR_M) & (here < 2*NEAR_M)
            c, s = np.cos(yaw), np.sin(yaw)
            body = np.stack((c*offset[:, 0]+s*offset[:, 1], -s*offset[:, 0]+c*offset[:, 1]), 1)
            approach = np.stack((np.clip(2*body[:, 0], -REVERSE_M_S, speed), np.clip(2*body[:, 1], -.2, .2),
                                 np.zeros(count)), 1)
            approach[distance < ARRIVE_M/2] = 0.
            command = np.where(close[:, None], approach, command)
            command[mode == 1] = 0.
            waits = np.where(mode == 1, waits-1, waits)
            mode = np.where((mode == 1) & (waits <= 0), 2, mode)
            requested[:, t] = command
            executed = limit(command, executed)
            xy, yaw = advance(xy, yaw, executed)
        return requested

    def proposals(self, snapshot, target):
        """The proposed tapes (K, T, 3) of requests and a name for each."""
        frames = self.frames
        programmes = [(None, 0, backing, speed) for speed in SPEEDS for backing in (False, True)]
        names = [f'target{"_backing" if backing else ""}@{speed}' for speed in SPEEDS for backing in (False, True)]
        programmes.append((None, frames, False, SPEEDS[0]))
        names.append('hold')
        waits = sorted({min(int(round(w/FRAME_S)), frames) for w in WAITS_S})
        for cell in self.near_cells(snapshot['xy']):
            for wait in waits:
                for backing in (False, True):
                    programmes.append((cell, wait, backing, SPEEDS[0]))
                    names.append(f'cell{cell}_wait{wait*FRAME_S:g}{"_backing" if backing else ""}')
        tapes = [self.follow(snapshot, target, programmes)]
        bases = [tapes[0][0]]
        if self.previous is not None:
            shifted = np.concatenate((self.previous[REPLAN_FRAMES:], np.repeat(self.previous[-1:], REPLAN_FRAMES, 0)))
            tapes.append(shifted[None])
            names.append('previous')
            bases.append(shifted)
        knots = np.arange(0, frames+KNOT_FRAMES, KNOT_FRAMES)
        noise = self.rng.normal(size=(GAUSSIANS, len(knots), 3))*SIGMA
        steps = np.arange(frames)
        smooth = np.stack([np.stack([np.interp(steps, knots, noise[g, :, a]) for a in range(3)], 1)
                           for g in range(GAUSSIANS)])
        tapes.append(np.stack([bases[g % len(bases)] for g in range(GAUSSIANS)])+smooth)
        names += [f'gaussian{g}' for g in range(GAUSSIANS)]
        return np.concatenate(tapes), names

    def costs(self, snapshot, target, requested):
        """Each tape's cost and its terms (see the module docstring)."""
        count, frames = requested.shape[:2]
        xy, yaw = integrate(snapshot, requested)
        walls = self.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1)                    # (K, T)
        breach = walls < FLOOR_M
        alive = np.where(breach.any(1), breach.argmax(1)+1, frames)          # frames worth stepping the walkers for
        walkers = self.model.walkers(snapshot, xy, yaw, alive)
        clearance = walls
        if walkers.shape[2]:
            gap = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None]).min(-1)-moving.RADIUS_M
            clearance = np.minimum(walls, gap)
        breach = clearance < FLOOR_M
        breached = breach.any(1)
        first = np.where(breached, breach.argmax(1), frames)
        counted = np.arange(frames)[None] < first[:, None]
        away = np.linalg.norm(xy-np.asarray(target, float), axis=-1) > ARRIVE_M
        time = (away & counted).sum(1)*FRAME_S
        soft = (np.clip((SOFT_M-clearance)/(SOFT_M-FLOOR_M), 0., 1.)*counted).sum(1)*FRAME_S*CLEARANCE_RATE
        end = xy[:, -1]
        static = self.fields.value(self.fields.static(target), end)
        left, sealed = static.copy(), np.zeros(count, bool)
        if self.terminal == 'walkers' and walkers.shape[2]:
            for k in np.flatnonzero(~breached):
                value = float(self.fields.value(self.fields.with_walkers(target, walkers[k, -1]), end[k:k+1])[0])
                sealed[k] = not np.isfinite(value)
                left[k] = static[k] if sealed[k] else value
        left = np.where(np.isfinite(left), left, 1e3)
        progress = left/V_REF+SEALED_S*sealed
        discard = np.where(breached, DISCARD*(2-first/frames), 0.)
        total = time+soft+progress+discard
        return total, dict(time=time, clearance=soft, progress=progress, sealed=sealed, breached=breached,
                           minimum_clearance=np.where(counted, clearance, np.inf).min(1))

    def plan(self, snapshot, target):
        """The next `REPLAN_FRAMES` requests (n, 3) from the world's snapshot towards `target`, and what was chosen."""
        target = np.asarray(target, float)
        if self.target is None or not np.array_equal(target, self.target):
            self.previous, self.target = None, target.copy()
        requested, names = self.proposals(snapshot, target)
        total, terms = self.costs(snapshot, target, requested)
        k = int(total.argmin())
        self.previous = requested[k]
        return requested[k, :REPLAN_FRAMES].astype(np.float32), dict(
            chosen=names[k], cost=float(total[k]), proposals=len(names), sealed=bool(terms['sealed'][k]),
            breached=bool(terms['breached'][k]), all_breached=bool(terms['breached'].all()),
            **{key: float(terms[key][k]) for key in ('time', 'clearance', 'progress')})
