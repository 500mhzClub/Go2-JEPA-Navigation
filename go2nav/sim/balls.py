"""P25's second world (docs/dynamic/PLAN.md P25): Zhu et al.'s bouncing balls (arXiv 2609.04264), rebuilt.

A red ball the agent moves and n white balls that move on their own, in 64 x 64 RGB frames on a black ground. What the
paper leaves out is chosen in PLAN P25 item 1:
- radius 3 px;
- an action is the red ball's displacement for one step (dx, dy) in pixels, its norm clipped to 3 (the paper's clip);
  the red ball stops at the walls, its centre kept a radius from them;
- white balls start at 1-3 px a step in a random direction and bounce elastically off the walls; they pass through
  each other and through the red ball;
- one step a frame.

Positions are (x, y) in pixels, x along the image's columns and y down its rows, with pixel k's centre at k + 0.5. Every
function is batched over episodes (the leading axis), in numpy; `red_rollout` is the red ball's motion in torch, for the
planner's ceiling (the white balls do not touch the red one, so its motion is its actions' alone).
"""
from dataclasses import dataclass

import numpy as np

SIZE, RADIUS, MAX_ACTION = 64, 3., 3.
SPEED = (1., 3.)                      # px a step, a white ball's speed (uniform)
HOLD = (1, 5)                         # steps the data's random policy holds an action (uniform, inclusive)
RED, WHITE = np.array((255., 0., 0.)), np.array((255., 255., 255.))
LOW, HIGH = RADIUS, SIZE-RADIUS       # where a centre may be


@dataclass
class State:
    red: np.ndarray        # (B, 2)
    white: np.ndarray      # (B, n, 2)
    velocity: np.ndarray   # (B, n, 2), px a step


def clip_actions(actions):
    """Actions (..., 2) with their norms clipped to MAX_ACTION."""
    a = np.asarray(actions, np.float64)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    return a*np.minimum(1., MAX_ACTION/np.maximum(norm, 1e-12))


def reset(rng, batch, n):
    """Every ball anywhere its centre may be; each white ball at a speed uniform in SPEED in a random direction."""
    red = rng.uniform(LOW, HIGH, (batch, 2))
    white = rng.uniform(LOW, HIGH, (batch, n, 2))
    speed, angle = rng.uniform(*SPEED, (batch, n)), rng.uniform(0, 2*np.pi, (batch, n))
    return State(red, white, np.stack((speed*np.cos(angle), speed*np.sin(angle)), -1))


def step(state, actions):
    """One step: the red ball by its clipped action, stopped at the walls; each white ball by its velocity, reflected
    elastically at the walls (a speed under the walls' gap needs one reflection at most)."""
    red = np.clip(state.red+clip_actions(actions), LOW, HIGH)
    white, velocity = state.white+state.velocity, state.velocity.copy()
    below, above = white < LOW, white > HIGH
    white = np.where(below, 2*LOW-white, np.where(above, 2*HIGH-white, white))
    velocity[below | above] *= -1
    return State(red, white, velocity)


def render(red, white):
    """Frames (B, 64, 64, 3) uint8 of balls at red (B, 2) and white (B, n, 2): the white balls first and the red one
    over them, each disc covering a pixel by clip(radius + 1/2 - distance to the pixel's centre, 0, 1)."""
    centres = np.arange(SIZE)+.5
    frame = np.zeros((len(red), SIZE, SIZE, 3))
    balls = [(white[:, k], WHITE) for k in range(white.shape[1])]+[(red, RED)]
    for xy, colour in balls:
        dx, dy = centres[None, None, :]-xy[:, 0, None, None], centres[None, :, None]-xy[:, 1, None, None]
        alpha = np.clip(RADIUS+.5-np.hypot(dx, dy), 0., 1.)[..., None]
        frame = alpha*colour+(1-alpha)*frame
    return np.round(frame).astype(np.uint8)


def random_actions(rng, batch, steps):
    """The data's policy: actions (B, steps, 2), each held 1-5 steps, its direction uniform and norm uniform in 0-3."""
    out = np.zeros((batch, steps, 2))
    for b in range(batch):
        t = 0
        while t < steps:
            hold = rng.integers(HOLD[0], HOLD[1]+1)
            angle, norm = rng.uniform(0, 2*np.pi), rng.uniform(0, MAX_ACTION)
            out[b, t:t+hold] = norm*np.cos(angle), norm*np.sin(angle)
            t += hold
    return out


def episodes(rng, batch, n, frames):
    """`batch` episodes of `frames` frames under the random policy: a dict of the frames (B, T, 64, 64, 3) uint8, the
    actions (B, T-1, 2) as executed (clipped), and the red (B, T, 2), white (B, T, n, 2) and velocity (B, T, n, 2)."""
    state, actions = reset(rng, batch, n), clip_actions(random_actions(rng, batch, frames-1))
    red, white, velocity = [state.red], [state.white], [state.velocity]
    for t in range(frames-1):
        state = step(state, actions[:, t])
        red.append(state.red), white.append(state.white), velocity.append(state.velocity)
    red, white, velocity = np.stack(red, 1), np.stack(white, 1), np.stack(velocity, 1)
    pixels = render(red.reshape(-1, 2), white.reshape(-1, n, 2)).reshape(batch, frames, SIZE, SIZE, 3)
    return dict(frames=pixels, actions=actions.astype(np.float32), red=red.astype(np.float32),
                white=white.astype(np.float32), velocity=velocity.astype(np.float32))


def red_rollout(red, actions):
    """The red ball's positions after each of a tape's actions, in torch: red (..., 2) and actions (..., H, 2) to
    (..., H, 2), norms clipped and the walls applied each step, as `step` does."""
    import torch
    norm = actions.norm(dim=-1, keepdim=True)
    actions = actions*torch.clamp(MAX_ACTION/norm.clamp_min(1e-12), max=1.)
    out, xy = [], red
    for k in range(actions.shape[-2]):
        xy = (xy+actions[..., k, :]).clamp(LOW, HIGH)
        out.append(xy)
    return torch.stack(out, -2)
