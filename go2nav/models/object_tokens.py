"""P27's R4 (docs/dynamic/PLAN.md P27 item 3): an object-token JEPA. Estimation is engineered and dynamics are learned,
in latent space.

- **The detector** (`detect`, no labels, nothing learned): each object has a known colour; its position is the centroid
  of the pixels near that colour. An object that cannot be seen (wholly covered) is placed at the frame's centre.
- **A token** per object and frame: a 16 x 16 crop centred on the detected position (bilinear), flattened through an
  MLP, plus an embedding of the position (Fourier features through an MLP) and a learned embedding of which object it
  is; then LeWM's BatchNorm projector to the latent.
- **The predictor** is LeWM's (`models.lewm.ARPredictor`, block-causal over frames, every token conditioned on the
  frame's action by AdaLN-zero), over the 4 object tokens of each history frame. Its targets are the tokens of the
  next frame: the future crops' and positions' embeddings.
- **The objective** (`token_losses`) is LeWM's: teacher-forced and rolled-out MSE, plus SIGReg on every token (each
  object at each time step a set of samples).

The class has `encode`, `predict` and `rollout` with `models.lewm.LeWorldModel`'s grid shapes, (B, T, 4, D), so the
bench's head and measures run unchanged.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from go2nav.models.keypoints import KeypointConfig, KeypointNet
from go2nav.models.lewm import ARPredictor, Embedder, MLP
from go2nav.sim import reactive_balls as world


@dataclass(frozen=True)
class ObjectTokenConfig:
    size: int = world.SIZE
    crop: int = 16
    width: int = 256
    latent: int = 128
    frequencies: int = 8
    projector_hidden: int = 2048
    history: int = 3
    predictor_depth: int = 4
    predictor_heads: int = 4
    predictor_head_dim: int = 16
    predictor_mlp: int = 256
    action_dim: int = 2
    tokens: int = 1+world.K
    grid: tuple = ()                     # read by code written for `LeWMConfig`
    residual: bool = False               # P27b Q3: the predictor's output is added to the frame's own tokens
    velocity: bool = False               # P27b Q4: and the token's last change (a constant-velocity prior)
    keypoints: int = 0                   # P30: learned keypoints (`models.keypoints`) in place of the colour detector
    velocity_agent: bool = True          # Q4's second round: False keeps the prior off the agent's token, whose motion
                                         # is its action (the ring world's robot token is given, not predicted)


def palette(device=None):
    """The 4 objects' colours (agent, then the obstacles), in [0, 1]."""
    import numpy as np
    return torch.tensor(np.concatenate((world.AGENT[None], world.COLOURS))/255., dtype=torch.float32, device=device)


def detect(pixels, reach=.5):
    """Frames (N, 3, H, W) in [-1, 1] to each object's position (N, 4, 2) in pixels and whether it was seen (N, 4).
    A pixel counts for an object by max(0, 1 - |pixel - colour| / reach), and the position is the weighted centroid."""
    rgb = (pixels.float()+1)/2
    colours = palette(pixels.device)
    weight = (1-(rgb[:, None]-colours[None, :, :, None, None]).norm(dim=2)/reach).clamp_min(0.)        # (N, 4, H, W)
    centres = torch.arange(pixels.shape[-1], device=pixels.device)+.5
    total = weight.sum((2, 3))
    x, y = (weight.sum(2)*centres).sum(-1), (weight.sum(3)*centres).sum(-1)
    seen = total > 1.
    xy = torch.stack((x, y), -1)/total.clamp_min(1e-6)[..., None]
    return torch.where(seen[..., None], xy, torch.full_like(xy, pixels.shape[-1]/2)), seen


def crops(pixels, xy, crop):
    """A crop x crop window of each frame (N, 3, H, W) centred on each position (N, S, 2) px, bilinear, zero beyond the
    frame: (N, S, 3, crop, crop)."""
    n, s = xy.shape[:2]
    size = pixels.shape[-1]
    offsets = (torch.arange(crop, device=pixels.device)+.5-crop/2)
    gx = xy[..., 0, None, None]+offsets[None, None, None, :]                                           # (N, S, 1, crop)
    gy = xy[..., 1, None, None]+offsets[None, None, :, None]                                           # (N, S, crop, 1)
    grid = torch.stack((gx.expand(n, s, crop, crop), gy.expand(n, s, crop, crop)), -1)/size*2-1
    out = F.grid_sample((pixels.float()+1)/2, grid.flatten(1, 2), mode='bilinear', padding_mode='zeros', align_corners=False)
    return out.unflatten(2, (s, crop)).transpose(1, 2)


class ObjectTokenJEPA(nn.Module):
    def __init__(self, config=ObjectTokenConfig()):
        super().__init__()
        self.config = c = config
        self.crop_embed = nn.Sequential(nn.Linear(3*c.crop**2, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.position_embed = nn.Sequential(nn.Linear(4*c.frequencies+3, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.identity = nn.Parameter(torch.randn(c.tokens, c.width)*.02)
        self.mix = nn.Sequential(nn.GELU(), nn.Linear(c.width, c.width))
        self.projector = MLP(c.width, c.projector_hidden, c.latent)
        self.action_encoder = Embedder(c.action_dim, 10, c.latent, 4)
        self.predictor = ARPredictor(num_frames=c.history, depth=c.predictor_depth, heads=c.predictor_heads,
                                     dim_head=c.predictor_head_dim, mlp_dim=c.predictor_mlp, dim=c.latent, tokens=c.tokens)
        self.pred_proj = MLP(c.latent, c.projector_hidden, c.latent)
        if c.residual:                   # starts as a copy of the last frame
            nn.init.zeros_(self.pred_proj.net[-1].weight)
            nn.init.zeros_(self.pred_proj.net[-1].bias)
        self.register_buffer('scales', math.pi*2.**torch.arange(c.frequencies), persistent=False)
        self.locator = None
        if c.keypoints:                  # P30: frozen; `order` is each object's keypoint, a fixed reordering of tokens
            self.locator = KeypointNet(KeypointConfig(keypoints=c.keypoints, size=c.size)).requires_grad_(False)
            self.register_buffer('order', torch.arange(c.tokens))

    def position_features(self, xy, seen):
        """(…, 2) px and (…,) bool to (…, 4 x frequencies + 3): the position in [-1, 1], its Fourier features, and seen."""
        u = xy/self.config.size*2-1
        phase = u[..., None]*self.scales
        return torch.cat((u, phase.sin().flatten(-2), phase.cos().flatten(-2), seen[..., None].float()), -1)

    def encode(self, pixels, located=None):
        """Frames (B, T, 3, H, W) in [-1, 1] to the object tokens (B, T, 4, D). `located` (B, T, 4, 2): the locator's
        positions for these frames, already computed (P30's trainer caches them; the locator is frozen)."""
        b = pixels.size(0)
        flat = pixels.flatten(0, 1)
        with torch.no_grad():
            if located is not None:
                xy = located.flatten(0, 1).float()
                seen = torch.ones(xy.shape[:2], dtype=torch.bool, device=xy.device)
            elif self.locator is not None:
                xy = self.locator(flat)[:, self.order]
                seen = torch.ones(xy.shape[:2], dtype=torch.bool, device=xy.device)
            else:
                xy, seen = detect(flat)
            windows = crops(flat, xy, self.config.crop).flatten(2)
        x = self.crop_embed(windows.to(self.crop_embed[0].weight.dtype))+self.position_embed(self.position_features(xy, seen))
        x = self.mix(x+self.identity)
        return self.projector(x.flatten(0, 1)).unflatten(0, (b, -1, self.config.tokens))

    def predict(self, z, actions):
        """Tokens (B, T, 4, D) and actions (B, T, 2) to the predicted next tokens at every position (B, T, 4, D)."""
        out = self.predictor(z, self.action_encoder(actions))
        out = self.pred_proj(out.flatten(0, 2)).unflatten(0, out.shape[:3])
        if self.config.velocity:         # none at the first step, which has no earlier frame
            change = z-torch.cat((z[:, :1], z[:, :-1]), 1)
            if not self.config.velocity_agent:
                change = torch.cat((torch.zeros_like(change[:, :, :1]), change[:, :, 1:]), 2)
            out = out+change
        return z+out if self.config.residual else out

    def rollout(self, z, actions, steps):
        """As `models.lewm.LeWorldModel.rollout`: the history's tokens (B, N, 4, D) and N-1+steps actions to the `steps`
        predicted frames' tokens (B, steps, 4, D)."""
        n = self.config.history
        if z.size(1) != n or actions.size(1) < n-1+steps:
            raise ValueError(f'{n} history frames and {n-1+steps} actions required')
        sequence, predicted = z, []
        for k in range(steps):
            nxt = self.predict(sequence[:, -n:], actions[:, k:k+n])[:, -1:]
            predicted.append(nxt)
            sequence = torch.cat((sequence, nxt), dim=1)
        return torch.cat(predicted, dim=1)


def token_losses(model, pixels, actions, *, rollout_steps, sigreg, lam, generator=None, z=None):
    """LeWM's objective on token states (B, T, S, D): `pred` (teacher-forced over the history), `rollout` (the
    autoregressive rollout to the next `rollout_steps` frames) and `sigreg` (every token at every time step a set of
    samples across the batch)."""
    n = model.config.history
    z = model.encode(pixels) if z is None else z
    pred, rolled = model.predict(z[:, :n], actions[:, :n]), model.rollout(z[:, :n], actions, rollout_steps)
    terms = dict(pred=(pred-z[:, 1:n+1]).pow(2).mean(), rollout=(rolled-z[:, n:n+rollout_steps]).pow(2).mean(),
                 sigreg=sigreg(z.float().flatten(1, 2).transpose(0, 1), generator))
    terms['loss'] = terms['pred']+terms['rollout']+lam*terms['sigreg']
    return terms
