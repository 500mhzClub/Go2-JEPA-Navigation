"""An object-level action-conditioned predictor (docs/dynamic/PLAN.md P22): C-JEPA's predictor (Nam et al., arXiv
2602.11389) over a robot token and object tokens, with LeWM's AdaLN-zero transformer (`models.lewm.ARPredictor`).

**Steps** are 0.4 s (`STEP_FRAMES` frames, the plan's period). Every coordinate is in the robot's body frame at the
step the prediction starts from (x forward, y left).

**Tokens** of a step (`object_features`, `robot_features`):
- an object: its position, whether it is seen at that step (not only remembered), whether the slot is filled, and the
  walls' distance on a `OBJECT_GRID` grid `OBJECT_SPACING` apart round it, aligned with the start frame, clipped at
  `CLIP_M`;
- the robot: its position and heading (cos, sin), and the walls on a `ROBOT_GRID` grid `ROBOT_SPACING` apart round it,
  aligned with its heading;
- a step's action: the `STEP_FRAMES` requested commands after it (scaled by the continuous block's bounds), which
  condition every token of the step (AdaLN-zero).
The predictor attends block-causally over `history` steps (a token sees every token of its own step and the earlier
ones: interactions within a step); each token keeps a learned per-slot embedding, and training shuffles the slots, so
the embedding marks an object's identity, not its place.

**Output:** each object's displacement to the next step (`head`), from the last position the predictor was shown:
its position at that step, or for a masked object its first one. With `residual` (run M2 on) the head's output is a
correction to the object carried on at its velocity (`advance`), the object's token also holds that velocity (the
tracker's in the history, the last predicted step's in a rollout), and the head starts at zero: the model starts as
constant velocity and learns what departs from it. The first runs (M0, M1) predicted the whole displacement and lost
to constant velocity at 1.6 s, the known result of trajectory prediction (Schoeller et al. 2020, arXiv 1903.07933).
**Object masking** (`mask`): a masked object's
tokens after the first history step are replaced by a projection of its first token plus a learned embedding, so the
predictor must infer where it is from the others and the robot's actions (C-JEPA).

`rollout` steps the predictor on its own predictions: each predicted position becomes the object's next token (seen,
its walls recomputed by `walls`, a lookup), with the robot's next token given (its path is the tape's).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from go2nav.locomotion.safety import BLOCKS
from go2nav.models.lewm import MLP, ARPredictor, Embedder
from go2nav.planning.wall_raster import grid_offsets

STEP_FRAMES = 4
STEP_S = STEP_FRAMES*.1
OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING, CLIP_M = 5, .25, 7, .3, 1.
COMMAND_SCALE = np.maximum(np.abs(BLOCKS['continuous'][0]), np.abs(BLOCKS['continuous'][1])).astype(np.float32)
OBJECT_DIM = 4+OBJECT_GRID**2
ROBOT_DIM = 4+ROBOT_GRID**2
ACTION_DIM = 3*STEP_FRAMES


@dataclass(frozen=True)
class ObjectConfig:
    slots: int = 6
    history: int = 3
    dim: int = 128
    depth: int = 6
    heads: int = 4
    head_dim: int = 32
    mlp: int = 512
    dropout: float = .1
    hidden: int = 256
    residual: bool = False

    def to_dict(self):
        return asdict(self)


def offsets(size, spacing, device=None):
    return torch.as_tensor(grid_offsets(size, spacing), dtype=torch.float32, device=device)


def object_features(xy, seen, present, walls, velocity=None, memory=None):
    """xy (..., 2), seen and present (...,) bool, walls (..., OBJECT_GRID**2) to (..., OBJECT_DIM); with `memory`
    (..., m) (P22 item 10) and `velocity` (..., 2) (m/s, a residual model's) appended, the velocity last."""
    keep = present[..., None].float()
    parts = [torch.nan_to_num(xy)*keep, seen[..., None].float(), keep, walls.clamp(max=CLIP_M)*keep]
    if memory is not None:
        parts.append(memory*keep)
    if velocity is not None:
        parts.append(velocity*keep)
    return torch.cat(parts, -1)


def robot_features(xy, yaw, walls):
    """xy (..., 2), yaw (...,), walls (..., ROBOT_GRID**2) to (..., ROBOT_DIM)."""
    return torch.cat((xy, torch.cos(yaw)[..., None], torch.sin(yaw)[..., None], walls.clamp(max=CLIP_M)), -1)


def actions_input(requested):
    """Requested commands (..., STEP_FRAMES, 3) to a step's action (..., ACTION_DIM)."""
    scale = torch.as_tensor(COMMAND_SCALE, device=requested.device)
    return (requested.float()/scale).flatten(-2)


class ObjectJEPA(nn.Module):
    def __init__(self, config=ObjectConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.object_in = MLP(OBJECT_DIM+(2 if config.residual else 0), config.hidden, d)
        self.robot_in = MLP(ROBOT_DIM, config.hidden, d)
        self.action_in = Embedder(ACTION_DIM, 16, d)
        self.predictor = ARPredictor(num_frames=config.history, depth=config.depth, heads=config.heads,
                                     dim_head=config.head_dim, mlp_dim=config.mlp, dim=d, dropout=config.dropout,
                                     tokens=1+config.slots)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.mask_proj = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, config.hidden), nn.GELU(), nn.Linear(config.hidden, 2))
        if config.residual:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def advance(self, delta, velocity):
        """The displacement to the next step: the head's output, and for a residual model the object carried on at
        `velocity` (..., 2) for one step besides."""
        return delta+velocity*STEP_S if self.config.residual else delta

    def embed(self, objects, robot):
        """objects (B, T, S, OBJECT_DIM), robot (B, T, ROBOT_DIM) to tokens (B, T, 1+S, d)."""
        b, t, s, _ = objects.shape
        o = self.object_in(objects.reshape(b*t*s, -1)).reshape(b, t, s, -1)
        r = self.robot_in(robot.reshape(b*t, -1)).reshape(b, t, 1, -1)
        return torch.cat((r, o), 2)

    def mask(self, tokens, masked):
        """Object masking: `masked` (B, S) bool; the masked objects' tokens after the first step replaced."""
        if masked is None or not masked.any():
            return tokens
        first = tokens[:, :1, 1:]
        replacement = (self.mask_proj(first)+self.mask_token).expand(-1, tokens.shape[1]-1, -1, -1)
        rest = tokens[:, 1:, 1:]
        rest = torch.where(masked[:, None, :, None], replacement, rest)
        return torch.cat((tokens[:, :, :1], torch.cat((tokens[:, :1, 1:], rest), 1)), 2)

    def forward(self, objects, robot, actions, masked=None):
        """Predicted displacement of every object to the step after each history step: (B, T, S, 2)."""
        z = self.mask(self.embed(objects, robot), masked)
        out = self.predictor(z, self.action_in(actions))
        return self.head(out[:, :, 1:])

    def rollout(self, objects, robot, actions, future_robot, walls, steps):
        """From `history` steps of tokens (objects (B, H, S, OBJECT_DIM), robot (B, H, ROBOT_DIM)), the actions of
        those steps and of the steps after them (B, H+steps-1, ACTION_DIM), the robot's tokens at the steps after the
        history (B, steps, ROBOT_DIM) and `walls` (positions (B, S, 2) to (B, S, OBJECT_GRID**2)): every object's
        predicted position at each step (B, steps, S, 2). Objects absent at the last history step stay absent."""
        h = self.config.history
        objects, robot = objects.clone(), robot.clone()
        present = objects[:, -1, :, 3] > .5
        position = objects[:, -1, :, :2]
        velocity = objects[:, -1, :, OBJECT_DIM:OBJECT_DIM+2] if self.config.residual else None
        out = []
        for k in range(steps):
            delta = self.forward(objects[:, -h:], robot[:, -h:], actions[:, k:k+h])[:, -1]
            move = self.advance(delta, velocity)
            position = position+move
            out.append(position)
            if self.config.residual:
                velocity = move/STEP_S
            new = object_features(position, present, present, walls(position), velocity)
            objects = torch.cat((objects, new[:, None]), 1)
            robot = torch.cat((robot, future_robot[:, k:k+1]), 1)
        return torch.stack(out, 1)
