"""P27's world (docs/dynamic/PLAN.md P27 item 1): reactive balls. A red disc the agent moves, and three obstacles of
their own colours that cruise, bounce off the walls and react to the agent, in 64 x 64 RGB frames.

- An action is the agent's displacement (dx, dy), its norm clipped to 3 px; the agent stops at the walls.
- An obstacle cruises at 0.5-2 px a step and bounces off the walls. Obstacles pass through each other.
- Its kind is drawn per episode: INDIFFERENT cruises whatever the agent does; YIELDING moves at its cruise velocity
  plus FLEE px a step straight away from the agent while the agent is within REACT px; PAUSING stops while the agent
  is within REACT, then waits a hidden delay of 5-15 steps before cruising again.
- Contact: an obstacle whose centre is under 2 radii from the agent's is pushed along the contact normal until they
  are 2 radii apart; if a wall stops it, the agent is moved back instead. Its cruise velocity is unchanged.
- T1 draws a ring round each obstacle that shows its kind (`render(..., kinds)`); T2 does not.

A step is a pure function of the state and the action: the pause delays come from a table drawn at reset and kept in
the state. A copy of the state under another tape is therefore an exact counterfactual (`branches`).

Positions are (x, y) in pixels, pixel k's centre at k + 0.5, as `sim.balls`. Everything is batched over episodes, in
numpy.
"""
from dataclasses import dataclass, replace

import numpy as np

from go2nav.sim.balls import LOW, HIGH, MAX_ACTION, RADIUS, SIZE, clip_actions

K = 3                                  # obstacles
SPEED = (.5, 2.)                       # px a step, an obstacle's cruise speed (uniform)
REACT, FLEE = 12., 2.                  # px: the reaction radius; px a step: a yielding obstacle's added speed
DELAY = (5, 15)                        # steps a pausing obstacle waits after the agent leaves (uniform, inclusive)
DELAYS = 16                            # delays drawn per obstacle at reset (used in turn, cyclically)
INDIFFERENT, YIELDING, PAUSING = 0, 1, 2
KINDS = ('indifferent', 'yielding', 'pausing')
AGENT = np.array((255., 0., 0.))
COLOURS = np.array(((0., 220., 0.), (40., 90., 255.), (235., 215., 0.)))       # green, blue, yellow
RINGS = np.array(((0., 0., 0.), (255., 255., 255.), (255., 0., 255.)))         # by kind: none, white, magenta
RING_RADIUS = 5.
TAPES = ('hold', 'approach', 'retreat', 'random')
BRANCH_STEPS = 16


@dataclass
class State:
    agent: np.ndarray      # (B, 2)
    obstacle: np.ndarray   # (B, K, 2)
    cruise: np.ndarray     # (B, K, 2) px a step: the velocity an obstacle cruises at
    velocity: np.ndarray   # (B, K, 2) px: each obstacle's displacement over the last step (its cruise at reset)
    kind: np.ndarray       # (B, K) int
    timer: np.ndarray      # (B, K) int: steps a pausing obstacle still waits once the agent is away
    delays: np.ndarray     # (B, K, DELAYS) int: the pause delays, used in turn
    pauses: np.ndarray     # (B, K) int: pauses begun so far (indexes `delays`)
    near: np.ndarray       # (B, K) bool: the agent was within REACT at the last step
    contact: np.ndarray    # (B, K) bool: the agent pushed the obstacle at the last step
    bounce: np.ndarray     # (B, K) bool: the obstacle met a wall at the last step

    def take(self, index):
        return State(**{k: v[index].copy() for k, v in vars(self).items()})


def reset(rng, batch):
    agent = rng.uniform(LOW, HIGH, (batch, 2))
    obstacle = rng.uniform(LOW, HIGH, (batch, K, 2))
    speed, angle = rng.uniform(*SPEED, (batch, K)), rng.uniform(0, 2*np.pi, (batch, K))
    cruise = np.stack((speed*np.cos(angle), speed*np.sin(angle)), -1)
    flags = np.zeros((batch, K), bool)
    return State(agent, obstacle, cruise, cruise.copy(), rng.integers(0, 3, (batch, K)), np.zeros((batch, K), int),
                 rng.integers(DELAY[0], DELAY[1]+1, (batch, K, DELAYS)), np.zeros((batch, K), int), flags, flags.copy(),
                 flags.copy())


def step(state, actions):
    """One step; see the module's description. Returns a new state."""
    agent = np.clip(state.agent+clip_actions(actions), LOW, HIGH)
    away = state.obstacle-agent[:, None]
    distance = np.linalg.norm(away, axis=-1)
    normal = away/np.maximum(distance, 1e-9)[..., None]
    near = distance < REACT
    yielding, pausing = state.kind == YIELDING, state.kind == PAUSING
    # the pause: a new one takes the next delay from the table; the timer runs once the agent is away
    began = pausing & near & ~state.near
    pauses = state.pauses+began
    delay = np.take_along_axis(state.delays, ((pauses-1) % DELAYS)[..., None], -1)[..., 0]
    waiting = pausing & ~near & (state.timer > 0)
    timer = np.where(pausing & near, delay, np.where(waiting, state.timer-1, state.timer))
    stopped = pausing & (near | waiting)
    move = np.where(stopped[..., None], 0., state.cruise+np.where((yielding & near)[..., None], FLEE*normal, 0.))
    obstacle, cruise = state.obstacle+move, state.cruise.copy()
    below, above = obstacle < LOW, obstacle > HIGH
    obstacle = np.where(below, 2*LOW-obstacle, np.where(above, 2*HIGH-obstacle, obstacle))
    cruise = np.where(below, np.abs(cruise), np.where(above, -np.abs(cruise), cruise))
    # contact: the obstacle is pushed out along the normal; where a wall stops it, the agent gives way
    away = obstacle-agent[:, None]
    distance = np.linalg.norm(away, axis=-1)
    contact = distance < 2*RADIUS
    normal = np.where((distance > 1e-9)[..., None], away/np.maximum(distance, 1e-9)[..., None], np.array((1., 0.)))
    pushed = np.clip(agent[:, None]+2*RADIUS*normal, LOW, HIGH)
    obstacle = np.where(contact[..., None], pushed, obstacle)
    for k in range(K):                                                   # the agent backs off an obstacle held by a wall
        gap = obstacle[:, k]-agent
        d = np.linalg.norm(gap, axis=-1)
        back = contact[:, k] & (d < 2*RADIUS-1e-9)
        n = gap/np.maximum(d, 1e-9)[:, None]
        agent = np.where(back[:, None], np.clip(obstacle[:, k]-2*RADIUS*n, LOW, HIGH), agent)
    return replace(state, agent=agent, obstacle=obstacle, cruise=cruise, velocity=obstacle-state.obstacle, timer=timer,
                   pauses=pauses, near=near, contact=contact, bounce=(below | above).any(-1))


def render(agent, obstacle, kinds=None):
    """Frames (B, 64, 64, 3) uint8: the obstacles in order, then the agent over them; each disc covers a pixel by
    clip(radius + 1/2 - distance, 0, 1). With `kinds` (B, K), T1: a ring of width 1 at RING_RADIUS in the kind's
    colour, drawn under its disc (INDIFFERENT has none)."""
    centres = np.arange(SIZE)+.5
    frame = np.zeros((len(agent), SIZE, SIZE, 3))

    def distances(xy):
        return np.hypot(centres[None, None, :]-xy[:, 0, None, None], centres[None, :, None]-xy[:, 1, None, None])

    for k in range(obstacle.shape[1]):
        d = distances(obstacle[:, k])
        if kinds is not None:
            ring = (np.clip(1.-np.abs(d-RING_RADIUS), 0., 1.)*(kinds[:, k] != INDIFFERENT)[:, None, None])[..., None]
            frame = ring*RINGS[kinds[:, k]][:, None, None]+(1-ring)*frame
        alpha = np.clip(RADIUS+.5-d, 0., 1.)[..., None]
        frame = alpha*COLOURS[k]+(1-alpha)*frame
    alpha = np.clip(RADIUS+.5-distances(agent), 0., 1.)[..., None]
    return np.round(alpha*AGENT+(1-alpha)*frame).astype(np.uint8)


# -- policies ---------------------------------------------------------------------------------------------------------

class Interactive:
    """The data's policy: segments of 5-20 steps. In about half, the agent heads for a random obstacle at a speed
    uniform in 1-3 px a step, with Gaussian noise of 0.7 px; in the rest it takes a random action (direction uniform,
    norm uniform in 0-3) redrawn every 1-5 steps. Neither depends on the obstacles' kinds."""

    def __init__(self, rng, batch):
        self.rng, self.left = rng, np.zeros(batch, int)
        self.chase, self.target, self.speed = np.zeros(batch, bool), np.zeros(batch, int), np.zeros(batch)
        self.hold, self.action = np.zeros(batch, int), np.zeros((batch, 2))

    def __call__(self, state):
        rng, n = self.rng, len(self.left)
        new = self.left <= 0
        self.left = np.where(new, rng.integers(5, 21, n), self.left)-1
        self.chase = np.where(new, rng.random(n) < .5, self.chase)
        self.target = np.where(new, rng.integers(0, K, n), self.target)
        self.speed = np.where(new, rng.uniform(1., MAX_ACTION, n), self.speed)
        redraw = new | (self.hold <= 0)
        self.hold = np.where(redraw, rng.integers(1, 6, n), self.hold)-1
        angle, norm = rng.uniform(0, 2*np.pi, n), rng.uniform(0, MAX_ACTION, n)
        self.action = np.where(redraw[:, None], np.stack((norm*np.cos(angle), norm*np.sin(angle)), -1), self.action)
        to = state.obstacle[np.arange(n), self.target]-state.agent
        heading = self.speed[:, None]*to/np.maximum(np.linalg.norm(to, axis=-1), 1e-9)[:, None]+rng.normal(0, .7, (n, 2))
        return clip_actions(np.where(self.chase[:, None], heading, self.action))


class Confounded:
    """The confound test's policy (R3 only): a scripted avoider. It heads for a waypoint (redrawn on arrival or after
    30 steps) at 3 px a step. With an obstacle within 18 px it slows in proportion to the nearest one's distance and
    turns away from it, so its actions depend on where the obstacles are, as a planner's do."""

    def __init__(self, rng, batch):
        self.rng, self.goal, self.left = rng, rng.uniform(LOW, HIGH, (batch, 2)), np.full(batch, 30)

    def __call__(self, state):
        rng, n = self.rng, len(self.left)
        to = self.goal-state.agent
        new = (np.linalg.norm(to, axis=-1) < 4.) | (self.left <= 0)
        self.goal = np.where(new[:, None], rng.uniform(LOW, HIGH, (n, 2)), self.goal)
        self.left = np.where(new, 30, self.left)-1
        to = self.goal-state.agent
        heading = to/np.maximum(np.linalg.norm(to, axis=-1), 1e-9)[:, None]
        away = state.agent[:, None]-state.obstacle
        distance = np.linalg.norm(away, axis=-1)
        nearest = distance.argmin(-1)
        d = distance[np.arange(n), nearest]
        flee = away[np.arange(n), nearest]/np.maximum(d, 1e-9)[:, None]
        w = np.clip((18.-d)/12., 0., 1.)[:, None]                         # 0 beyond 18 px, 1 within 6
        direction = (1-w)*heading+w*flee
        direction = direction/np.maximum(np.linalg.norm(direction, axis=-1), 1e-9)[:, None]
        return clip_actions(MAX_ACTION*(1-.7*w)*direction+rng.normal(0, .3, (n, 2)))


POLICIES = dict(interactive=Interactive, confounded=Confounded)


# -- episodes and exact branches --------------------------------------------------------------------------------------

def record(state):
    return dict(agent=state.agent, obstacle=state.obstacle, velocity=state.velocity, timer=state.timer,
                contact=state.contact, bounce=state.bounce, near=state.near)


def stack(rows):
    return {k: np.stack([r[k] for r in rows], 1) for k in rows[0]}


def branches(rng, state, steps=BRANCH_STEPS):
    """The 4 tapes of TAPES from one state, each on its own copy: hold (no motion); approach, towards the obstacle
    nearest at the start at 3 px a step; retreat, away from it; random, a random action redrawn every 1-5 steps.
    Returns the executed actions (B, 4, steps, 2), the target obstacle (B,) and the states after each step, each
    (B, 4, steps, ...)."""
    n = len(state.agent)
    target = np.linalg.norm(state.obstacle-state.agent[:, None], axis=-1).argmin(-1)
    actions, rows = [], []
    for tape in TAPES:
        s, tape_actions, tape_rows = state.take(slice(None)), [], []
        hold, action = np.zeros(n, int), np.zeros((n, 2))
        for _ in range(steps):
            to = s.obstacle[np.arange(n), target]-s.agent
            unit = to/np.maximum(np.linalg.norm(to, axis=-1), 1e-9)[:, None]
            if tape == 'hold':
                a = np.zeros((n, 2))
            elif tape in ('approach', 'retreat'):
                a = (MAX_ACTION if tape == 'approach' else -MAX_ACTION)*unit
            else:
                redraw = hold <= 0
                hold = np.where(redraw, rng.integers(1, 6, n), hold)-1
                angle, norm = rng.uniform(0, 2*np.pi, n), rng.uniform(0, MAX_ACTION, n)
                action = np.where(redraw[:, None], np.stack((norm*np.cos(angle), norm*np.sin(angle)), -1), action)
                a = action
            a = clip_actions(a)
            s = step(s, a)
            tape_actions.append(a), tape_rows.append(record(s))
        actions.append(np.stack(tape_actions, 1)), rows.append(stack(tape_rows))
    return np.stack(actions, 1), target, {k: np.stack([r[k] for r in rows], 1) for k in rows[0]}


def episodes(rng, batch, frames, policy='interactive', moments=0, branch_frames=False):
    """`batch` episodes of `frames` frames under a policy, with `moments` branch moments each.

    Returns (episode, branch). `episode`: the T2 frames `frames` and T1 frames `frames_t1` (B, T, 64, 64, 3) uint8, the
    actions (B, T-1, 2) as executed, `kind` (B, K), and per frame the agent (B, T, 2), obstacle and velocity (B, T, K,
    2), and timer, contact, bounce and near (B, T, K). `branch` (None without moments): `t0` (B, M), the frame each
    branch starts from, sorted; `target` (B, M); `actions` (B, M, 4, 16, 2); the states after each step (B, M, 4, 16,
    ...); and with `branch_frames`, the frames of both variants (B, M, 4, 16, 64, 64, 3)."""
    state, drive = reset(rng, batch), POLICIES[policy](rng, batch)
    t0 = np.sort(np.stack([rng.choice(np.arange(3, frames-BRANCH_STEPS), moments, replace=False) for _ in range(batch)]),
                 -1) if moments else None
    rows, actions, parts = [record(state)], [], [[None]*moments for _ in range(batch)]
    for t in range(frames):
        if moments:
            b, m = np.nonzero(t0 == t)
            if len(b):
                acts, target, out = branches(rng, state.take(b))
                for i, (bi, mi) in enumerate(zip(b, m)):
                    parts[bi][mi] = dict(actions=acts[i], target=target[i], **{k: v[i] for k, v in out.items()})
        if t == frames-1:
            break
        a = drive(state)
        state = step(state, a)
        actions.append(a), rows.append(record(state))
    ep = stack(rows)
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    kinds = np.repeat(state.kind, frames, 0)
    shape = (batch, frames, SIZE, SIZE, 3)
    episode = dict(frames=render(flat(ep['agent']), flat(ep['obstacle'])).reshape(shape),
                   frames_t1=render(flat(ep['agent']), flat(ep['obstacle']), kinds).reshape(shape),
                   actions=np.stack(actions, 1).astype(np.float32), kind=state.kind.astype(np.int8),
                   **{k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in ep.items()})
    if not moments:
        return episode, None
    branch = {k: np.stack([np.stack([parts[b][m][k] for m in range(moments)]) for b in range(batch)])
              for k in parts[0][0]}
    if branch_frames:
        a, o = branch['agent'].reshape(-1, 2), branch['obstacle'].reshape(-1, K, 2)
        shape = (*branch['agent'].shape[:-1], SIZE, SIZE, 3)
        kinds = np.repeat(state.kind, moments*len(TAPES)*BRANCH_STEPS, 0)
        branch['frames'], branch['frames_t1'] = render(a, o).reshape(shape), render(a, o, kinds).reshape(shape)
    branch = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in branch.items()}
    return episode, dict(t0=t0.astype(np.int16), **branch)
