"""A world model of the range ring (docs/dynamic/PLAN.md P21): LeWM's recipe (`models.lewm`: an encoder and an
action-conditioned latent predictor trained jointly, a next-embedding loss and SIGReg, no stop-gradient or EMA) on one
sensor, the 720-beam ring, in the kinematic world of P20.

**Input** (`scan_input`): a scan as (720, 2): the range on a log scale from `MIN_RANGE_M` (0) to `MAX_RANGE_M` (1),
and whether the beam returned. Nothing else: no command history, no pose.

**Encoder:** a circular 1-D convolution stem over the beams, then the ring cut into `sectors` (24 of 15 degrees), each
sector's features one token, a learned position per sector, and a small transformer; each token through a BatchNorm
projector to the latent (`latent`). So a walker fills a token of its own. With `single_vector` (run R3) the tokens are
averaged into one before the projector.

**Predictor:** LeWM's `ARPredictor` over `history` frames of tokens, block-causal, every token of a frame conditioned
(AdaLN-zero) on that frame's command: the requested (forward, lateral, yaw) of the frame, scaled by the continuous
block's bounds. Position i predicts frame i+1; `rollout` runs it step by step on its own predictions.

**Decoder:** the latent's tokens back to the scan's log range (two transformer blocks and a linear map from each token
to its sector's 30 beams). With `decoder_grad` (the default) its loss reaches the latent: the next-scan auxiliary;
without (run R2) the decoder learns from a detached latent and leaves the latent to the JEPA objective alone. The
planner reads walkers from decoded scans (`planning.sampling.LearnedRingModel`).
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go2nav.locomotion.safety import BLOCKS
from go2nav.models.lewm import MLP, ARPredictor, Embedder, ViTBlock
from go2nav.models.norm import LayerNorm
from go2nav.perception.range import BEAMS, MAX_RANGE_M, MIN_RANGE_M

COMMAND_SCALE = np.maximum(np.abs(BLOCKS['continuous'][0]), np.abs(BLOCKS['continuous'][1])).astype(np.float32)
LOG_SPAN = math.log(MAX_RANGE_M/MIN_RANGE_M)


@dataclass(frozen=True)
class RingConfig:
    sectors: int = 24
    stem: int = 64
    width: int = 128
    depth: int = 4
    heads: int = 4
    mlp: int = 512
    latent: int = 128
    projector_hidden: int = 512
    history: int = 4
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_head_dim: int = 32
    predictor_mlp: int = 512
    predictor_dropout: float = .1
    action_smoothed_dim: int = 16
    decoder_depth: int = 2
    decoder_grad: bool = True
    single_vector: bool = False

    @property
    def tokens(self):
        return 1 if self.single_vector else self.sectors

    def to_dict(self):
        return asdict(self)


def scan_input(ranges, valid=None):
    """Scans (..., 720) in metres (0 where a beam did not return, if `valid` is not given) to the model's input
    (..., 720, 2) float32: the log range in [0, 1] and the validity flag. Torch tensors or arrays."""
    torch_in = torch.is_tensor(ranges)
    r = ranges.float() if torch_in else torch.as_tensor(np.asarray(ranges, np.float32))
    v = (r > 0) if valid is None else (valid if torch.is_tensor(valid) else torch.as_tensor(np.asarray(valid, bool)))
    v = v.to(r.device)
    x = torch.log(r.clamp(MIN_RANGE_M, MAX_RANGE_M)/MIN_RANGE_M)/LOG_SPAN
    return torch.stack((torch.where(v, x, torch.zeros_like(x)), v.float()), -1)


def ranges_of(decoded):
    """Decoded log ranges (..., 720) back to metres."""
    return MIN_RANGE_M*torch.exp(decoded.clamp(0., 1.)*LOG_SPAN)


def command_input(requested):
    """Requested commands (..., 3) scaled by the continuous block's bounds."""
    requested = requested if torch.is_tensor(requested) else torch.as_tensor(np.asarray(requested, np.float32))
    return requested.float()/torch.as_tensor(COMMAND_SCALE, device=requested.device)


class CircularStem(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.a = nn.Conv1d(2, out//2, 7, padding=3, padding_mode='circular')
        self.b = nn.Conv1d(out//2, out, 7, padding=3, padding_mode='circular')

    def forward(self, x):                       # (N, 720, 2) to (N, out, 720)
        return F.gelu(self.b(F.gelu(self.a(x.transpose(1, 2)))))


class RingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.per = BEAMS//config.sectors
        self.stem = CircularStem(config.stem)
        self.patch = nn.Linear(self.per*config.stem, config.width)
        self.position = nn.Parameter(torch.randn(1, config.sectors, config.width)*.02)
        self.blocks = nn.ModuleList(ViTBlock(config.width, config.heads, config.mlp) for _ in range(config.depth))
        self.norm = LayerNorm(config.width)

    def forward(self, x):                       # (N, 720, 2) to (N, sectors, width)
        n = x.shape[0]
        h = self.stem(x).transpose(1, 2).reshape(n, self.config.sectors, self.per*self.config.stem)
        h = self.patch(h)+self.position
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


class RingDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lift = nn.Linear(config.latent, config.width)
        self.position = nn.Parameter(torch.randn(1, config.sectors, config.width)*.02)
        self.blocks = nn.ModuleList(ViTBlock(config.width, config.heads, config.mlp) for _ in range(config.decoder_depth))
        self.norm = LayerNorm(config.width)
        self.out = nn.Linear(config.width, BEAMS//config.sectors)

    def forward(self, z):                       # (N, tokens, latent) to (N, 720) log ranges
        h = self.lift(z)
        if h.shape[1] == 1:
            h = h.expand(-1, self.config.sectors, -1)
        h = h+self.position
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h)).flatten(1)


class RingJEPA(nn.Module):
    def __init__(self, config=RingConfig()):
        super().__init__()
        self.config = config
        self.encoder = RingEncoder(config)
        self.projector = MLP(config.width, config.projector_hidden, config.latent)
        self.action_encoder = Embedder(3, config.action_smoothed_dim, config.latent)
        self.predictor = ARPredictor(num_frames=config.history, depth=config.predictor_depth, heads=config.predictor_heads,
                                     dim_head=config.predictor_head_dim, mlp_dim=config.predictor_mlp, dim=config.latent,
                                     dropout=config.predictor_dropout, tokens=config.tokens)
        self.pred_proj = MLP(config.latent, config.projector_hidden, config.latent)
        self.decoder = RingDecoder(config)

    def encode(self, x):
        """Inputs (N, 720, 2) to latents (N, tokens, latent)."""
        h = self.encoder(x)
        if self.config.single_vector:
            h = h.mean(1, keepdim=True)
        n, s, w = h.shape
        return self.projector(h.reshape(n*s, w)).reshape(n, s, -1)

    def predict(self, z, actions):
        """One predictor pass: latents (B, T, tokens, D) and commands (B, T, 3) to the next frames' latents (B, T,
        tokens, D)."""
        b, t, s, d = z.shape
        c = self.action_encoder(actions)
        if s == 1:                                 # one token a frame: ARPredictor's plain path
            out = self.predictor(z[:, :, 0], c)[:, :, None]
        else:
            out = self.predictor(z, c)
        return self.pred_proj(out.reshape(b*t*s, d)).reshape(b, t, s, d)

    def rollout(self, history, actions):
        """From latents of `history` frames (B, H, tokens, D), the commands of those frames and the K frames after
        them (B, H+K-1, 3): the predicted latents of the K frames after the history (B, K, tokens, D)."""
        h = self.config.history
        window, out = history, []
        steps = actions.shape[1]-h+1
        for k in range(steps):
            nxt = self.predict(window, actions[:, k:k+h])[:, -1:]
            out.append(nxt)
            window = torch.cat((window[:, 1:], nxt), 1)
        return torch.cat(out, 1)

    def decode(self, z):
        """Latents (..., tokens, D) to log ranges (..., 720)."""
        lead = z.shape[:-2]
        if not self.config.decoder_grad:
            z = z.detach()
        return self.decoder(z.reshape(-1, *z.shape[-2:])).reshape(*lead, BEAMS)
