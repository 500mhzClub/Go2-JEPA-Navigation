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
its position at that step, or for a masked object its first one. **Object masking** (`mask`): a masked object's
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

    def to_dict(self):
        return asdict(self)


def offsets(size, spacing, device=None):
    return torch.as_tensor(grid_offsets(size, spacing), dtype=torch.float32, device=device)


def object_features(xy, seen, present, walls):
    """xy (..., 2), seen and present (...,) bool, walls (..., OBJECT_GRID**2) to (..., OBJECT_DIM)."""
    keep = present[..., None].float()
    return torch.cat((torch.nan_to_num(xy)*keep, seen[..., None].float(), keep, walls.clamp(max=CLIP_M)*keep), -1)


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
        self.object_in = MLP(OBJECT_DIM, config.hidden, d)
        self.robot_in = MLP(ROBOT_DIM, config.hidden, d)
        self.action_in = Embedder(ACTION_DIM, 16, d)
        self.predictor = ARPredictor(num_frames=config.history, depth=config.depth, heads=config.heads,
                                     dim_head=config.head_dim, mlp_dim=config.mlp, dim=d, dropout=config.dropout,
                                     tokens=1+config.slots)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.mask_proj = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, config.hidden), nn.GELU(), nn.Linear(config.hidden, 2))

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
        out = []
        for k in range(steps):
            delta = self.forward(objects[:, -h:], robot[:, -h:], actions[:, k:k+h])[:, -1]
            position = position+delta
            out.append(position)
            new = object_features(position, present, present, walls(position))
            objects = torch.cat((objects, new[:, None]), 1)
            robot = torch.cat((robot, future_robot[:, k:k+1]), 1)
        return torch.stack(out, 1)
