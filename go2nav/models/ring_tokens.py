"""P28 (docs/dynamic/PLAN.md P28 item 2): P27's object-token JEPA (`models.object_tokens`) on the ring world's objects.

Estimation is the ring tracker's; the dynamics are learned in latent space.
- **An object's token:** `models.object_jepa.object_features` of a slot at a step (its position in the robot's frame
  at the branch point, seen, present, the walls' distance on a 5 x 5 grid round it, its velocity) with Fourier
  features of the position, through an MLP and LeWM's BatchNorm projector. **The robot's token:** `robot_features`
  (its pose and the walls round it) likewise. A state is (B, T, 1 + slots, D), the robot first.
- **Empty slots** (about three in four) take one learned token and pass through no BatchNorm, in the encoder and in
  the predictor's output. The first run put them through the projectors: their share of a batch then set its
  statistics, the history's and the future's batches differed, and the tokens of training were not those of
  evaluation (STATUS P28, 16 November).
- **The predictor** is LeWM's (`models.lewm.ARPredictor`, block-causal over the history's steps, every token
  conditioned on the step's requests by AdaLN-zero), with M5's sizes. It predicts the next step's tokens.
- **Rollouts are in latent space:** the objects' tokens are the predictor's own, and the robot's token at each future
  step is the encoding of its pose on the tape's path (as M5 is given it). Nothing is decoded on the way.
- **The objective** (`token_losses`): the MSE of the teacher-forced predictions over the history and of a rollout
  into the future, on the tokens that have a target (a history slot seen at the next step; a matched walker's true
  position at a future step; the robot always), plus SIGReg on the robot's tokens per step and on the present
  objects' tokens pooled into sets of the batch's size.
- **The head** (`PositionHead`, trained afterwards on encoded tokens only) reads an object's position from its token.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from go2nav.models.lewm import ARPredictor, Embedder, MLP
from go2nav.models.object_jepa import ACTION_DIM, OBJECT_DIM, ROBOT_DIM


@dataclass(frozen=True)
class RingTokenConfig:
    slots: int = 6
    history: int = 3
    width: int = 256
    latent: int = 128
    frequencies: int = 8
    position_scale: float = 4.        # m: positions over it go into the Fourier features and out of the head
    projector_hidden: int = 2048
    predictor_depth: int = 6
    predictor_heads: int = 4
    predictor_head_dim: int = 32
    predictor_mlp: int = 512
    residual: bool = False            # P27b Q3: an object's predicted token is its own token plus the predictor's output

    @property
    def tokens(self):
        return 1+self.slots


def body(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))


class RingTokenJEPA(nn.Module):
    def __init__(self, config=RingTokenConfig()):
        super().__init__()
        self.config = c = config
        self.object_in = body(OBJECT_DIM+2+4*c.frequencies, c.width)
        self.robot_in = body(ROBOT_DIM+4*c.frequencies, c.width)
        self.object_projector = MLP(c.width, c.projector_hidden, c.latent)
        self.robot_projector = MLP(c.width, c.projector_hidden, c.latent)
        self.action_encoder = Embedder(ACTION_DIM, 16, c.latent)
        self.predictor = ARPredictor(num_frames=c.history, depth=c.predictor_depth, heads=c.predictor_heads,
                                     dim_head=c.predictor_head_dim, mlp_dim=c.predictor_mlp, dim=c.latent, tokens=c.tokens)
        self.pred_proj = MLP(c.latent, c.projector_hidden, c.latent)
        self.robot_pred_proj = MLP(c.latent, c.projector_hidden, c.latent)
        if c.residual:
            nn.init.zeros_(self.pred_proj.net[-1].weight)
            nn.init.zeros_(self.pred_proj.net[-1].bias)
        self.absent = nn.Parameter(torch.zeros(c.latent))
        self.register_buffer('scales', math.pi*2.**torch.arange(c.frequencies), persistent=False)

    def fourier(self, features):
        """Features (..., F) whose first two values are a position (m): with its Fourier features appended."""
        phase = (features[..., :2]/self.config.position_scale)[..., None]*self.scales
        return torch.cat((features, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)

    def fill(self, projector, x, present):
        """`projector` on the rows of x (..., W) where `present` (...,); the learned empty token elsewhere."""
        out = self.absent.to(x.dtype).expand(*x.shape[:-1], -1).clone()
        if present.sum() > 1:
            out[present] = projector(x[present]).to(out.dtype)
        return out

    def encode_objects(self, objects):
        """`object_features` (..., OBJECT_DIM + 2) to tokens (..., D); a slot is empty where its present flag is 0."""
        x = self.object_in(self.fourier(objects.float()).to(self.object_in[0].weight.dtype))
        return self.fill(self.object_projector, x, objects[..., 3] > .5)

    def encode_robot(self, robot):
        """`robot_features` (..., ROBOT_DIM) to tokens (..., D)."""
        x = self.robot_in(self.fourier(robot.float()).to(self.robot_in[0].weight.dtype))
        return self.robot_projector(x.flatten(0, -2)).unflatten(0, x.shape[:-1])

    def encode(self, objects, robot):
        """objects (B, T, S, ·) and robot (B, T, ·) to the state (B, T, 1 + S, D)."""
        return torch.cat((self.encode_robot(robot)[:, :, None], self.encode_objects(objects)), 2)

    def predict(self, z, actions, present):
        """States (B, T, 1 + S, D), actions (B, T, ACTION_DIM) and the slots present at the branch point (B, S) to the
        predicted next states (B, T, 1 + S, D); a slot empty at the branch point stays empty."""
        out = self.predictor(z, self.action_encoder(actions))
        robot = self.robot_pred_proj(out[:, :, 0].flatten(0, 1)).unflatten(0, out.shape[:2])
        alive = present[:, None].expand(-1, out.size(1), -1)
        objects = self.fill(self.pred_proj, out[:, :, 1:], alive)
        if self.config.residual:
            objects = torch.where(alive[..., None], z[:, :, 1:].to(objects.dtype)+objects, objects)
        return torch.cat((robot[:, :, None].to(objects.dtype), objects), 2)

    def rollout(self, z, actions, future_robot, steps, present):
        """From the history's states (B, H, 1 + S, D), the actions of the history's steps and of those after them
        (B, H - 1 + steps, ·), the robot's tokens at the steps after the history (B, steps, D) and the slots present at
        the branch point (B, S): the predicted states (B, steps, 1 + S, D), each with the robot's token as predicted;
        the robot's given token takes its place in what the next step attends over."""
        h = self.config.history
        sequence, predicted = z, []
        for k in range(steps):
            nxt = self.predict(sequence[:, -h:], actions[:, k:k+h], present)[:, -1:]
            predicted.append(nxt)
            given = torch.cat((future_robot[:, k:k+1, None].to(nxt.dtype), nxt[:, :, 1:]), 2)
            sequence = torch.cat((sequence, given), 1)
        return torch.cat(predicted, 1)


class PositionHead(nn.Module):
    """An object's token (..., D) to its position (..., 2) in the scene's frame, over `position_scale`."""

    def __init__(self, config, hidden=512, layers=2):
        super().__init__()
        widths = [config.latent]+[hidden]*layers
        self.net = nn.Sequential(*[m for a, b in zip(widths, widths[1:]) for m in (nn.Linear(a, b), nn.GELU())],
                                 nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


def token_losses(model, b, *, rollout_steps, sigreg, lam, generator=None):
    """The objective on a batch of `data.ring_scenes.Scenes.batch` (the module's description). Returns the terms:
    `pred`, `rollout`, `sigreg`, `loss`."""
    h = model.config.history
    states = model.encode(torch.cat((b['objects'], b['future_objects'][:, :rollout_steps]), 1),
                          torch.cat((b['robot'], b['future_robot'][:, :rollout_steps]), 1))     # one call: one batch
    z, future = states[:, :h], states[:, h:]
    alive = b['present'][:, -1]
    count = torch.cat((torch.ones_like(b['valid'][:, :1, :1]).expand(-1, h-1+rollout_steps, -1),
                       torch.cat((b['seen'][:, 1:], b['valid'][:, :rollout_steps]), 1)), 2)   # (B, H-1+K, 1+S)
    target = torch.cat((z[:, 1:], future), 1)
    mse = lambda out, tgt, c: ((out-tgt).float().pow(2).mean(-1)*c).sum()/c.sum().clamp(min=1)
    pred = model.predict(z, b['actions'][:, :h], alive)
    rolled = model.rollout(z, b['actions'], future[:, :, 0], rollout_steps, alive)
    terms = dict(pred=mse(pred, target[:, :h], count[:, :h]), rollout=mse(rolled, future, count[:, h-1:]))
    everything = torch.cat((z, future), 1).float()
    present = torch.cat((b['present'], b['valid'][:, :rollout_steps]), 1)
    objects = everything[:, :, 1:][present]
    size = len(z)
    order = torch.randperm(len(objects), generator=generator, device=objects.device)[:len(objects)//size*size]
    terms['sigreg'] = (sigreg(everything[:, :, 0].transpose(0, 1), generator)
                       + sigreg(objects[order].unflatten(0, (-1, size)), generator))/2
    terms['loss'] = terms['pred']+terms['rollout']+lam*terms['sigreg']
    return terms
