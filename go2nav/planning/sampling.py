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

**Model** (P18 replaces the robot's part with a learned one): the robot's path is the tape's, through the continuous
block's limits, at 100 ms. The walkers' part is one of `MODELS`: `true` (copies of the world's walkers stepped against
that path, random generators included, so the model knows their future draws), `fresh` (the same with new dice each
plan), `velocity`, `frozen` and `frozen_seen` (`SimpleModel`: walkers that do not react), or `ring_frozen` and
`ring_velocity` (`RingModel`, P20: the same two, with the walkers and their velocities taken from the range ring by
`perception.ring_walkers`; the mission hands such a model every frame's scan through `observe`), or
`ring_model:<path>` (P21: the learned ring world model, `planning.ring_model.LearnedRingModel`, rolled out under each
tape: it is also told each frame's request through `record`, and given the tapes), or `objects:<path>` (P22: the
object-level predictor over the ring tracker's objects, `planning.object_model.ObjectModel`, likewise), or
`modes:<path>` (P22 item 8: several futures an object, `planning.object_model.ModesModel`), or `mix:<parts>` (P22
item 12: the true future beside false ones, `planning.mixture.MixtureModel`).

**Cost,** in seconds:
- time: 0.1 s for each step further than `ARRIVE_M` from the target;
- clearance: `CLEARANCE_RATE` seconds a second as the body outline's clearance falls from `SOFT_M` to `FLOOR_M`; a tape
  that goes under `FLOOR_M` is discarded (`DISCARD`, less for a later breach, so that if every tape breaches the latest
  is taken);
- progress: the distance left from the tape's end over `V_REF`. `terminal='static'`: through the walls alone.
  `terminal='walkers'`: with the walkers, where the tape leaves them, as obstacles; a sealed way costs `SEALED_S` more.
"""
import copy
import os

import numpy as np

from go2nav.locomotion.safety import BLOCKS
from go2nav.perception import ring_walkers
from go2nav.planning.fields import Fields
from go2nav.sim import kinematic as kin
from go2nav.sim import obstacles as moving

FRAME_S = kin.FRAME_S
REPLAN_FRAMES, KNOT_FRAMES = 4, 8
V_REF, ARRIVE_M, NEAR_M = .3, .03, .4
FLOOR_M, SOFT_M, CLEARANCE_RATE, SEALED_S, DISCARD = .08, .20, 2., 200., 1e4
CONTACT_S = 2000.          # what a contact costs a tape, times its probability (a model of several futures)
CONTACT_CAP = None         # P22 item 12: a tape whose chance of contact passes this is discarded as a breach (None: priced)
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


RING_MOUNT_M = .30          # the range sensor's mount, ahead of the body's centre (`sim.range_sensor`)
MODELS = ('true', 'fresh', 'velocity', 'frozen', 'frozen_seen', 'ring_frozen', 'ring_velocity')


def carried(at, velocity, count, horizon, geometry):
    """Walkers at `at` (n, 2) carried on at `velocity` (n, 2) for `horizon` frames, each stopping where a wall would
    be within its radius: (count, horizon, n, 2), the same for every tape."""
    path = at[None]+velocity[None]*(np.arange(1, horizon+1)*FRAME_S)[:, None, None]          # (T, n, 2)
    if len(at) and velocity.any():
        free = geometry.wall_distance(path) >= moving.RADIUS_M
        reached = np.minimum.accumulate(free, 0)                       # until the first frame at a wall
        last = np.maximum(reached.sum(0)-1, 0)                          # (n,) the last free frame
        stopped = np.where(reached.any(0)[:, None], path[last, np.arange(len(at))], at)
        path = np.where(reached[..., None], path, stopped[None])
    return np.broadcast_to(path[None], (count, horizon, len(at), 2)).copy()


class TrueModel:
    """The walkers' part of the model: copies of the world's walkers stepped against a robot path at 100 ms. With
    `fresh` dice each plan's copies get a new random generator, the same for every tape of the plan: the model knows
    the walkers' rules and their state, not their future draws."""

    def __init__(self, fresh=False, seed=0):
        self.fresh, self.rng = fresh, np.random.default_rng([seed, 17])

    def walkers(self, snapshot, xy, yaw, frames, geometry=None):
        """Walker positions (K, T, n, 2) along paths xy (K, T, 2), yaw (K, T), each stepped for its first `frames[k]`
        frames (the rest repeat the last position)."""
        count, horizon = xy.shape[:2]
        out = np.zeros((count, horizon, len(snapshot['walkers']), 2))
        if not snapshot['walkers']:
            return out
        seeds = self.rng.integers(2**62, size=len(snapshot['walkers'])) if self.fresh else None
        for k in range(count):
            walkers = copy.deepcopy(snapshot['walkers'])
            if self.fresh:
                for walker, seed in zip(walkers, seeds):
                    walker.rng = np.random.default_rng(int(seed))
            at, heading = snapshot['xy'], snapshot['yaw']
            for t in range(int(frames[k])):
                kin.step_walkers(walkers, FRAME_S, at, heading)      # against the pose the frame starts from
                out[k, t] = [w.xy for w in walkers]
                at, heading = xy[k, t], float(yaw[k, t])
            out[k, int(frames[k]):] = out[k, max(int(frames[k])-1, 0)]
        return out


class SimpleModel:
    """Walkers that do not react to the robot (P17, the ladder of models):
    - `velocity`: each walker carried on at its velocity over the last frame, stopping where a wall would be within
      its radius;
    - `frozen`: each walker where it is now;
    - `frozen_seen`: only what a range ring shows: a walker is where it was last in the sensor's line of sight (the
      mount, `RING_MOUNT_M` ahead of the centre), and is forgotten once that place is in sight and empty. One never
      seen is not in the model."""

    def __init__(self, kind):
        self.kind, self.remembered = kind, {}

    def believed(self, snapshot, geometry):
        """The walkers' positions and velocities the model starts from."""
        xy, velocity = snapshot['walkers_xy'], snapshot['walkers_velocity']
        if self.kind != 'frozen_seen':
            return xy, velocity if self.kind == 'velocity' else np.zeros_like(xy)
        mount = snapshot['xy']+RING_MOUNT_M*np.array([np.cos(snapshot['yaw']), np.sin(snapshot['yaw'])])
        for k, at in enumerate(xy):
            if geometry.visible(mount, at):
                self.remembered[k] = at.copy()
            elif k in self.remembered and geometry.visible(mount, self.remembered[k]):
                del self.remembered[k]
        kept = np.array([self.remembered[k] for k in sorted(self.remembered)], float).reshape(-1, 2)
        return kept, np.zeros_like(kept)

    def walkers(self, snapshot, xy, yaw, frames, geometry=None):
        at, velocity = self.believed(snapshot, geometry)
        return carried(at, velocity, *xy.shape[:2], geometry)


class RingModel:
    """Walkers that do not react, known from the range ring alone (P20): `ring_frozen` where the tracker has them,
    `ring_velocity` carried on at the tracker's velocity. `observe` takes each frame's scan and pose; the snapshot's
    walkers are not read."""

    def __init__(self, kind):
        self.kind, self.tracker, self.fields = kind, ring_walkers.Tracker(), None

    def bind(self, fields):
        """The layout's wall distances (`planning.fields.Fields`), given by the planner."""
        self.fields = fields

    def observe(self, t, scan, xy, yaw, geometry):
        found = ring_walkers.detect(*scan, xy, yaw, self.fields.wall_clearance)
        self.tracker.update(t, found, ring_walkers.mount_of(xy, yaw), geometry)

    def walkers(self, snapshot, xy, yaw, frames, geometry=None):
        at, velocity = self.tracker.state()
        return carried(at, velocity if self.kind == 'ring_velocity' else np.zeros_like(at), *xy.shape[:2], geometry)


def make_model(name, seed=0):
    if name.startswith('ring_model:'):
        from go2nav.planning.ring_model import LearnedRingModel
        return LearnedRingModel(name.partition(':')[2])
    if name.startswith('objects:'):
        from go2nav.planning.object_model import ObjectModel
        return ObjectModel(name.partition(':')[2])
    if name.startswith('modes:') or name.startswith('expect:'):     # expect: a contact priced by its probability
        from go2nav.planning.object_model import ModesModel
        model = ModesModel(name.partition(':')[2])
        model.EXPECT = name.startswith('expect:')
        import os                                             # development switches (STATUS P22)
        model.W_STILL = float(os.environ.get('GO2NAV_W_STILL', model.W_STILL))
        model.STILL_FRAMES = int(os.environ.get('GO2NAV_STILL_FRAMES', model.STILL_FRAMES))
        return model
    if name.startswith('mix:'):                                       # P22 item 12: the true future beside false ones
        from go2nav.planning.mixture import MixtureModel
        return MixtureModel(name.partition(':')[2], seed=seed)
    if name not in MODELS:
        raise ValueError(f'the model is one of {MODELS} or ring_model:<path>')
    if name.startswith('ring_'):
        return RingModel(name)
    return TrueModel(fresh=name == 'fresh', seed=seed) if name in ('true', 'fresh') else SimpleModel(name)


class Planner:
    def __init__(self, geometry, *, horizon_s, terminal='static', seed=0, model='true', max_speed=None, cells=True):
        if terminal not in ('static', 'walkers'):
            raise ValueError("terminal is 'static' or 'walkers'")
        self.geometry, self.fields, self.terminal = geometry, Fields(geometry), terminal
        self.frames = int(round(horizon_s/FRAME_S))
        self.model = make_model(model, seed) if isinstance(model, str) else model
        if hasattr(self.model, 'bind'):
            self.model.bind(self.fields)
        self.speeds = SPEEDS if max_speed is None else tuple(v*max_speed/SPEEDS[0] for v in SPEEDS)
        self.max_speed, self.use_cells = max_speed, cells
        self.rng = np.random.default_rng(seed)
        self.contact_cap = None                     # overrides CONTACT_CAP (the suites set it by GO2NAV_CONTACT_CAP)
        if os.environ.get('GO2NAV_CONTACT_CAP'):
            self.contact_cap = float(os.environ['GO2NAV_CONTACT_CAP'])
        self.floor_exempt = os.environ.get('GO2NAV_FLOOR_EXEMPT', '1') != '0'
        self.scenarios = int(os.environ.get('GO2NAV_SCENARIOS', '0'))    # P22 item 12: `scenario_costs` when set
        self.scenario_rng = np.random.default_rng([seed, 31])           # not `rng`: the proposals stay as they were
        self.exact = os.environ.get('GO2NAV_EXACT', '0') == '1'          # P22 item 12: `expected_costs` when set
        self.recover = os.environ.get('GO2NAV_RECOVER', '0') == '1'      # ... with a way out from under the floor
        self.walker_price = float(os.environ['GO2NAV_WALKER_PRICE']) if os.environ.get('GO2NAV_WALKER_PRICE') else None
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
        speeds = self.speeds
        programmes = [(None, 0, backing, speed) for speed in speeds for backing in (False, True)]
        names = [f'target{"_backing" if backing else ""}@{speed:g}' for speed in speeds for backing in (False, True)]
        programmes.append((None, frames, False, speeds[0]))
        names.append('hold')
        waits = sorted({min(int(round(w/FRAME_S)), frames) for w in WAITS_S})
        for cell in self.near_cells(snapshot['xy']) if self.use_cells else ():
            for wait in waits:
                for backing in (False, True):
                    programmes.append((cell, wait, backing, speeds[0]))
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
        requested = np.concatenate(tapes)
        if self.max_speed is not None:                                 # the handicapped robot: no faster than this
            requested[..., 0] = np.clip(requested[..., 0], -min(REVERSE_M_S, self.max_speed), self.max_speed)
        return requested, names

    def costs(self, snapshot, target, requested):
        """Each tape's cost and its terms (see the module docstring)."""
        count, frames = requested.shape[:2]
        xy, yaw = integrate(snapshot, requested)
        walls = self.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1)                    # (K, T)
        breach = walls < FLOOR_M
        alive = np.where(breach.any(1), breach.argmax(1)+1, frames)          # frames worth stepping the walkers for
        extra = dict(requested=requested) if getattr(self.model, 'takes_tapes', False) else {}
        walkers = self.model.walkers(snapshot, xy, yaw, alive, self.geometry, **extra)
        clearance = walls
        weights, risk, expected_soft = getattr(self.model, 'weights', None), 0., 0.
        if walkers.shape[2] and self.exact:            # a model of one future: each walker's, at weight 1 (P22 item 13)
            groups = self.model.groups if weights is not None else np.arange(walkers.shape[2])
            weights = weights if weights is not None else np.ones((count, walkers.shape[2]))
            return self.expected_costs(snapshot, target, xy, yaw, walls, walkers, weights, groups)
        if weights is not None and walkers.shape[2] and self.scenarios:
            return self.scenario_costs(target, xy, yaw, walls, walkers, weights, self.model.groups)
        capped = None
        if weights is not None and walkers.shape[2]:
            # a model of several futures with probabilities: a contact is priced by its probability, not ruled out
            gaps = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None])-moving.RADIUS_M       # (K, T, n)
            now = kin.rectangle_gap(walkers[:, 0], snapshot['xy'], snapshot['yaw'])-moving.RADIUS_M   # (K, n)
            floor = np.minimum(FLOOR_M, now-.005)      # already under the floor: a tape may still not go nearer
            if not self.floor_exempt:                  # P22 item 12: as the true model's rule, under it is a breach
                floor = np.full_like(now, FLOOR_M)
            hits = np.maximum.accumulate(gaps < floor[:, None], 1)                    # (K, T, n): a contact by then
            groups = self.model.groups
            clear = np.ones((count, frames))
            for g in np.unique(groups):
                clear = clear*(1.-np.clip((weights[:, None, groups == g]*hits[..., groups == g]).sum(-1), 0., 1.))
            chance = 1.-clear                                                         # (K, T): of a contact by then
            cap = CONTACT_CAP if self.contact_cap is None else self.contact_cap
            if cap is not None:
                capped = chance > cap                     # past the cap: a breach from then, as the true model's
                chance = np.where(capped.any(1)[:, None], 0., chance)
            risk = CONTACT_S*chance[:, -1]
            softs = np.clip((SOFT_M-gaps)/(SOFT_M-FLOOR_M), 0., 1.).sum(1)*FRAME_S*CLEARANCE_RATE     # (K, n)
            expected_soft = (weights*softs).sum(1)
        elif walkers.shape[2]:
            gap = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None]).min(-1)-moving.RADIUS_M
            clearance = np.minimum(walls, gap)
        breach = clearance < FLOOR_M
        if capped is not None:
            breach = breach | capped
        breached = breach.any(1)
        first = np.where(breached, breach.argmax(1), frames)
        counted = np.arange(frames)[None] < first[:, None]
        away = np.linalg.norm(xy-np.asarray(target, float), axis=-1) > ARRIVE_M
        time = (away & counted).sum(1)*FRAME_S
        soft = (np.clip((SOFT_M-clearance)/(SOFT_M-FLOOR_M), 0., 1.)*counted).sum(1)*FRAME_S*CLEARANCE_RATE
        end = xy[:, -1]
        static = self.fields.value(self.fields.static(target), end)
        left, sealed = static.copy(), np.zeros(count, bool)
        ends = getattr(self.model, 'ends', None)             # a model of several futures: each walker at its likeliest
        ends = walkers[:, -1] if ends is None else ends
        if self.terminal == 'walkers' and walkers.shape[2]:
            for k in np.flatnonzero(~breached):
                value = float(self.fields.value(self.fields.with_walkers(target, ends[k]), end[k:k+1])[0])
                sealed[k] = not np.isfinite(value)
                left[k] = static[k] if sealed[k] else value
        left = np.where(np.isfinite(left), left, 1e3)
        progress = left/V_REF+SEALED_S*sealed
        discard = np.where(breached, DISCARD*(2-first/frames), 0.)
        total = time+soft+progress+discard+risk+expected_soft
        return total, dict(time=time, clearance=soft, progress=progress, sealed=sealed, breached=breached,
                           minimum_clearance=np.where(counted, clearance, np.inf).min(1))

    def scenario_costs(self, target, xy, yaw, walls, walkers, weights, groups):
        """P22 item 12: a model of several futures through the true model's rule. `scenarios` joint futures are drawn,
        each walker (`groups`) taking one of its futures by its probability under each tape (the same uniform draws for
        every tape), and each tape is costed on each as the true model's are (a walker under `FLOOR_M` is a breach,
        the later the cheaper; clearance from the nearest; the distance left with the walkers where the model's
        likeliest futures leave them, `ends`); the cost is the mean. With one future, the true model's rule exactly."""
        count, frames = xy.shape[:2]
        ids = np.unique(groups)
        draws = self.scenario_rng.random((self.scenarios, len(ids)))
        chosen = np.empty((self.scenarios, count, len(ids)), int)
        for i, g in enumerate(ids):
            members = np.flatnonzero(groups == g)
            w = np.clip(weights[:, members], 0., None)
            total = w.sum(1, keepdims=True)
            cdf = np.cumsum(np.where(total > 0, w/np.where(total > 0, total, 1.), 1./len(members)), 1)   # (K, m)
            pick = (draws[:, i, None, None] > cdf[None]).sum(-1)                                          # (S, K)
            chosen[:, :, i] = members[np.minimum(pick, len(members)-1)]
        end = xy[:, -1]
        static = self.fields.value(self.fields.static(target), end)
        aware, sealed_aware = static.copy(), np.zeros(count, bool)
        if self.terminal == 'walkers':
            ends = getattr(self.model, 'ends', None)
            ends = walkers[:, -1] if ends is None else ends
            for k in range(count):
                value = float(self.fields.value(self.fields.with_walkers(target, ends[k]), end[k:k+1])[0])
                sealed_aware[k] = not np.isfinite(value)
                aware[k] = static[k] if sealed_aware[k] else value
        away = np.linalg.norm(xy-np.asarray(target, float), axis=-1) > ARRIVE_M
        sums = dict(total=0., time=0., clearance=0., progress=0., sealed=0., breached=0.)
        minimum = np.full(count, np.inf)
        for s in range(self.scenarios):
            paths = np.take_along_axis(walkers, chosen[s][:, None, :, None], 2)                           # (K, T, n, 2)
            gap = kin.rectangle_gap(paths, xy[:, :, None], yaw[:, :, None]).min(-1)-moving.RADIUS_M
            clearance = np.minimum(walls, gap)
            breach = clearance < FLOOR_M
            breached = breach.any(1)
            first = np.where(breached, breach.argmax(1), frames)
            counted = np.arange(frames)[None] < first[:, None]
            time = (away & counted).sum(1)*FRAME_S
            soft = (np.clip((SOFT_M-clearance)/(SOFT_M-FLOOR_M), 0., 1.)*counted).sum(1)*FRAME_S*CLEARANCE_RATE
            sealed = ~breached & sealed_aware
            left = np.where(breached, static, aware)
            left = np.where(np.isfinite(left), left, 1e3)
            progress = left/V_REF+SEALED_S*sealed
            discard = np.where(breached, DISCARD*(2-first/frames), 0.)
            for key, value in (('total', time+soft+progress+discard), ('time', time), ('clearance', soft),
                               ('progress', progress), ('sealed', sealed), ('breached', breached)):
                sums[key] = sums[key]+value
            minimum = np.minimum(minimum, np.where(counted, clearance, np.inf).min(1))
        mean = {key: value/self.scenarios for key, value in sums.items()}
        return mean['total'], dict(time=mean['time'], clearance=mean['clearance'], progress=mean['progress'],
                                   sealed=mean['sealed'] > .5, breached=mean['breached'] >= 1., minimum_clearance=minimum)

    def expected_costs(self, snapshot, target, xy, yaw, walls, walkers, weights, groups, ends=None):
        """P22 item 12: the true model's rule in expectation over a model of several futures, worked out rather than
        drawn (`scenario_costs` with every joint future, the walkers taken as independent). The breach comes at the
        wall's first frame or a walker's, whichever is first; S(t), the chance that none has come by frame t, is the
        product over walkers of one less the weight of their futures within the floor by then. In expectation: the
        breach cost over its frame (`DISCARD`, the later the cheaper), the time and clearance of each frame times S(t)
        (clearance from the nearest: a frame's worst of the walls and each walker's expected nearness), and the
        distance left, through the walkers where the likeliest futures leave them if none breaches (`ends`), else
        through the walls alone. With one future, the true model's rule exactly. With `recover`, a walker already under
        the floor counts only when a tape goes nearer (the expected-cost rule's way out). `ends` (K, n, 2): where the
        walkers end up for the distance left, in place of the model's (`eval.contact_calibration` costs the truth)."""
        count, frames = xy.shape[:2]
        gaps = kin.rectangle_gap(walkers, xy[:, :, None], yaw[:, :, None])-moving.RADIUS_M          # (K, T, J)
        if self.recover:
            now = kin.rectangle_gap(walkers[:, 0], snapshot['xy'], snapshot['yaw'])-moving.RADIUS_M  # (K, J)
            floor = np.minimum(FLOOR_M, now-.005)[:, None]
        else:
            floor = FLOOR_M
        hits = np.maximum.accumulate(gaps < floor, 1)
        w = np.clip(np.asarray(weights, float), 0., None)
        softs = np.clip((SOFT_M-gaps)/(SOFT_M-FLOOR_M), 0., 1.)                                    # (K, T, J)
        clear, nearness = np.ones((count, frames)), np.zeros((count, frames))
        for g in np.unique(groups):
            members = groups == g
            total = w[:, members].sum(1, keepdims=True)
            share = np.where(total > 0, w[:, members]/np.where(total > 0, total, 1.), 0.)            # (K, m)
            clear = clear*(1.-np.clip((share[:, None]*hits[..., members]).sum(-1), 0., 1.))
            nearness = np.maximum(nearness, (share[:, None]*softs[..., members]).sum(-1))
        wall_breach = walls < FLOOR_M
        wall_first = np.where(wall_breach.any(1), wall_breach.argmax(1), frames)
        survive = np.where(np.arange(frames)[None] < wall_first[:, None], clear, 0.)                # S(t)
        before = np.concatenate((np.ones((count, 1)), survive[:, :-1]), 1)
        breach_at = before-survive                                                                  # P(first at t)
        late = (2-np.arange(frames)/frames)[None]
        walker = np.arange(frames)[None] < wall_first[:, None]           # before the wall's frame: a walker's breach
        price = DISCARD if self.walker_price is None else self.walker_price
        discard = (np.where(walker, price, DISCARD)*breach_at*late).sum(1)
        away = np.linalg.norm(xy-np.asarray(target, float), axis=-1) > ARRIVE_M
        time = (survive*away).sum(1)*FRAME_S
        wall_soft = np.clip((SOFT_M-walls)/(SOFT_M-FLOOR_M), 0., 1.)
        soft = (survive*np.maximum(wall_soft, nearness)).sum(1)*FRAME_S*CLEARANCE_RATE
        end = xy[:, -1]
        static = self.fields.value(self.fields.static(target), end)
        aware, sealed_aware = static.copy(), np.zeros(count, bool)
        if self.terminal == 'walkers':
            ends = getattr(self.model, 'ends', None) if ends is None else ends
            ends = walkers[:, -1] if ends is None else ends
            for k in range(count):
                value = float(self.fields.value(self.fields.with_walkers(target, ends[k]), end[k:k+1])[0])
                sealed_aware[k] = not np.isfinite(value)
                aware[k] = static[k] if sealed_aware[k] else value
        static, aware = np.where(np.isfinite(static), static, 1e3), np.where(np.isfinite(aware), aware, 1e3)
        clean = survive[:, -1]                                                                      # no breach at all
        progress = (1.-clean)*static/V_REF+clean*(aware/V_REF+SEALED_S*sealed_aware)
        breached = clean < .5
        minimum = np.minimum(walls, np.where(w[:, None] > 0, gaps, np.inf).min(-1)).min(1)
        return time+soft+progress+discard, dict(time=time, clearance=soft, progress=progress,
                                                sealed=sealed_aware & ~breached, breached=breached,
                                                minimum_clearance=minimum)

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
