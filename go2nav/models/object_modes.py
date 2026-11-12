"""Several futures for each object in one pass, conditioned on the robot's tape (docs/dynamic/PLAN.md P22 item 8).

P22's step-by-step predictor (`models.object_jepa`) gives one future. From observations a walker's future has
branches (a pause that ends at a time nobody sees, a kind that is not visible), and the planner drove into them
(STATUS P22, 12 November). This is the design trajectory prediction uses for that (MTP, Cui et al. 2019; conditioned on
the ego's plan as PiP, Song et al. 2020): `modes` futures an object, each with a probability, over the whole horizon.

**Tokens** (body frame of the robot at the step the prediction starts from, steps of 0.4 s, as `models.object_jepa`):
- over `history` steps, a robot token (position, heading, the walls round it, the requests of the step after it) and
  an object token a slot (position, seen, present, the tracker's velocity, the walls round it on a wider grid,
  `WIDE_GRID` x `WIDE_GRID` at `WIDE_SPACING`: the predictor cannot look the walls up again along the way);
- over the `steps` future steps, a robot token each: where the tape takes the robot.
One transformer over all of them (every token sees every other), with learned embeddings for the step and for the
token's place in a step.

**Output,** from each object's token at the last history step: `modes` futures as corrections to the object carried
on at its velocity (`steps` x 2 each), and a logit a mode. `futures` gives positions and probabilities.

**Loss** (`losses`): for each object, each mode's mean distance to the tracker's later positions (the steps it is seen
at); the nearest mode takes `1 - EPSILON` of the weight and the others share `EPSILON` (so none dies); plus the
cross-entropy of the logits on which mode was nearest.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from go2nav.models.lewm import MLP, ViTBlock
from go2nav.models.norm import LayerNorm
from go2nav.models.object_jepa import ACTION_DIM, ROBOT_DIM, STEP_S

WIDE_GRID, WIDE_SPACING = 9, .3
OBJECT_DIM = 4+WIDE_GRID**2+2
EPSILON = .05


@dataclass(frozen=True)
class ModesConfig:
    slots: int = 6
    history: int = 3
    steps: int = 16
    modes: int = 6
    dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp: int = 512
    hidden: int = 256

    def to_dict(self):
        return asdict(self)


class ObjectModes(nn.Module):
    def __init__(self, config=ModesConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.object_in = MLP(OBJECT_DIM, config.hidden, d)
        self.robot_in = MLP(ROBOT_DIM+ACTION_DIM, config.hidden, d)
        self.step = nn.Parameter(torch.randn(1, config.history+config.steps, 1, d)*.02)
        self.place = nn.Parameter(torch.randn(1, 1, 1+config.slots, d)*.02)
        self.blocks = nn.ModuleList(ViTBlock(d, config.heads, config.mlp) for _ in range(config.depth))
        self.norm = LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, config.hidden), nn.GELU(),
                                  nn.Linear(config.hidden, config.modes*(2*config.steps+1)))
        nn.init.normal_(self.head[-1].weight, std=1e-3)        # near constant velocity, the modes a little apart
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, objects, robot, future_robot):
        """objects (B, H, S, OBJECT_DIM); robot (B, H, ROBOT_DIM+ACTION_DIM); future_robot (B, steps, the same).
        Returns corrections (B, S, modes, steps, 2) and logits (B, S, modes)."""
        c = self.config
        b, h, s, _ = objects.shape
        o = self.object_in(objects.reshape(b*h*s, -1)).reshape(b, h, s, -1)
        r = self.robot_in(torch.cat((robot, future_robot), 1).reshape(b*(h+c.steps), -1)).reshape(b, h+c.steps, 1, -1)
        past = torch.cat((r[:, :h], o), 2)+self.step[:, :h]+self.place                     # (B, H, 1+S, d)
        ahead = r[:, h:]+self.step[:, h:]+self.place[:, :, :1]                              # (B, steps, 1, d)
        x = torch.cat((past.flatten(1, 2), ahead.flatten(1, 2)), 1)
        for block in self.blocks:
            x = block(x)
        tokens = self.norm(x)[:, :h*(1+s)].reshape(b, h, 1+s, -1)[:, -1, 1:]              # the objects at the last step
        out = self.head(tokens).reshape(b, s, c.modes, 2*c.steps+1)
        return out[..., :-1].reshape(b, s, c.modes, c.steps, 2), out[..., -1]

    def futures(self, objects, robot, future_robot):
        """Positions (B, S, modes, steps, 2) and probabilities (B, S, modes): each object carried on at the velocity
        in its last token, plus each mode's correction."""
        correction, logits = self.forward(objects, robot, future_robot)
        return carried(objects, self.config.steps)[:, :, None]+correction, logits.softmax(-1)


def carried(objects, steps):
    """(B, S, steps, 2): each object from its last history position at the velocity in that token."""
    last = objects[:, -1]
    k = torch.arange(1, steps+1, device=objects.device)[None, None, :, None]
    return last[..., None, :2]+last[..., None, -2:]*STEP_S*k


def losses(model, objects, robot, future_robot, target, counted):
    """`target` (B, steps, S, 2) and `counted` (B, steps, S) bool: the tracker's later positions and where they
    count. Returns the loss and its terms."""
    correction, logits = model(objects, robot, future_robot)
    positions = carried(objects, model.config.steps)[:, :, None]+correction               # (B, S, K, T, 2)
    goal, mask = target.permute(0, 2, 1, 3)[:, :, None], counted.permute(0, 2, 1)[:, :, None].float()
    distance = ((positions-goal).pow(2).sum(-1)+1e-8).sqrt()                              # (B, S, K, T)
    error = (distance*mask).sum(-1)/mask.sum(-1).clamp(min=1)                             # (B, S, K)
    valid = mask.sum(-1)[:, :, 0] > 0                                                     # (B, S)
    nearest = error.argmin(-1)
    modes = model.config.modes
    weight = torch.full_like(error, EPSILON/max(modes-1, 1) if modes > 1 else 0.)
    weight.scatter_(-1, nearest[..., None], 1.-EPSILON if modes > 1 else 1.)
    count = valid.sum().clamp(min=1)
    fit = ((error*weight).sum(-1)*valid).sum()/count
    choice = (F.cross_entropy(logits.flatten(0, 1).float(), nearest.flatten(), reduction='none').reshape(valid.shape)*valid).sum()/count
    return fit+choice, dict(fit=fit, choice=choice, nearest=(error.min(-1).values*valid).sum()/count)
