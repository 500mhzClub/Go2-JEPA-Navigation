"""LeWorldModel (LeWM; Maes, Le Lidec, Scieur, LeCun and Balestriero, arXiv:2603.19312) for the Go2: an image encoder and
an action-conditioned latent predictor trained jointly from pixels, with a next-embedding loss and the SIGReg
anti-collapse regulariser, and no stop-gradient, EMA or pretrained weights. The C5/C6 world model.

Ported from the official implementation (github.com/lucas-maes/le-wm at 8edfeb3; `module.py`, `jepa.py`; MIT,
Copyright (c) 2026 Lucas Maes, notice in THIRD_PARTY_NOTICES.md), without its training frameworks: the predictor,
AdaLN-zero blocks, action embedder, projectors and SIGReg follow it line for line; the encoder is a ViT in the Hugging
Face layout that the official code instantiates (`vit_hf`), written out here.

**Go2 configuration** (see docs/lewm/PLAN.md):
- one model step is 100 ms: one camera frame and the five 50 Hz policy steps after it (the paper's frameskip 5);
- the action of a step is its block of five executed (forward, yaw) commands, normalised by the control statistics
  (data/training/control_norm_stats.json): 10 values; with `command_dim=3` (corpus v2, `go2nav.data.lewm_corpus_v2`)
  the block holds (forward, lateral, yaw), normalised by corpus v2's statistics (`CONTROL_STATS_3`): 15 values;
- frames are the primary camera's 640 x 480 RGB resized to 224 x 168 (bicubic; `resize_frame`), the full field of
  view; with patch 14 the encoder sees 16 x 12 tokens;
- the predictor attends over `history` = 4 frames (t-300 .. t ms) and rolls out autoregressively.

**Encoder.** ViT (pre-norm blocks, learned [CLS] and position embeddings, final LayerNorm); the [CLS] token goes
through a BatchNorm MLP projector (the final LayerNorm would otherwise keep SIGReg from acting) to z. ViT-Tiny: width
192, 12 blocks, 3 heads, MLP 768, z 192 (about 5.5M parameters); ViT-Small: 384, 12, 6, 1536, z 384.

**Predictor.** A causal transformer over the history's z (learned temporal embedding), each position conditioned on
its own action by AdaLN-zero (scale, shift, gate; zero-initialised), then a BatchNorm MLP projector: 6 blocks,
16 heads of 64, MLP 2048, dropout 0.1 (about 10.8M parameters at z 192). Position i predicts z of frame i+1.

**Loss** (`losses`): the paper's teacher-forced next-embedding MSE over the history, plus (optionally, the Go2
configuration) the MSE of a K-step autoregressive rollout, plus lambda x SIGReg of every frame's z, step by step.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

FRAME_SIZE = (224, 168)  # (width, height) of the encoder input
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REPO = Path(__file__).resolve().parents[2]
CONTROL_STATS = REPO/'data/training/control_norm_stats.json'
CONTROL_STATS_3 = REPO/'data/training/control_norm_stats_corpus_v2.json'
COMMAND_AXES = {2: [0, 2], 3: [0, 1, 2]}   # the command components (forward, lateral, yaw) an action block holds


@dataclass(frozen=True)
class LeWMConfig:
    image_hw: tuple = (FRAME_SIZE[1], FRAME_SIZE[0])
    patch: int = 14
    width: int = 192
    depth: int = 12
    heads: int = 3
    mlp: int = 768
    latent: int = 192
    projector_hidden: int = 2048
    history: int = 4
    block_steps: int = 5
    command_dim: int = 2
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    predictor_mlp: int = 2048
    predictor_dropout: float = .1
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4
    inverse_dynamics: bool = False  # the latent-difference action decoder (a Delta-JEPA-style variant, not the paper's)
    inverse_spans: tuple = (1,)     # frame spans the inverse dynamics decodes over (1: consecutive frames)
    inverse_span_weights: tuple = ()  # relative weights of those spans in the inverse term (empty: equal)
    motion_dynamics: bool = False   # also decode the body-frame motion over each span (privileged poses; a variant)

    @property
    def action_dim(self):
        return self.block_steps*self.command_dim

    @classmethod
    def named(cls, size, **overrides):
        """`tiny` (the paper's encoder) or `small`."""
        sizes = dict(tiny=dict(width=192, depth=12, heads=3, mlp=768, latent=192),
                     small=dict(width=384, depth=12, heads=6, mlp=1536, latent=384))
        return cls(**(sizes[size] | overrides))


# -- frames and actions -----------------------------------------------------------------------------------------------

def resize_frame(rgb):
    """A primary-camera RGB frame (HxWx3 uint8) to the encoder's 168 x 224 x 3 uint8 (bicubic). Training data and the
    forecaster both use this, so they see identical pixels."""
    image = Image.fromarray(np.asarray(rgb))
    return np.array(image.convert('RGB').resize(FRAME_SIZE, Image.Resampling.BICUBIC), dtype=np.uint8)


def normalise_frames(frames):
    """uint8 (..., H, W, 3) frames (a tensor, on any device) to float (..., 3, H, W), ImageNet-normalised."""
    x = frames.float().div(255.).movedim(-1, -3)
    mean = x.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(IMAGENET_STD)[:, None, None]
    return (x-mean)/std


def control_stats(command_dim=2):
    """Mean and standard deviation of the command components a block holds (`COMMAND_AXES`)."""
    stats = json.loads((CONTROL_STATS if command_dim == 2 else CONTROL_STATS_3).read_text())
    return np.asarray(stats['control_mean'], np.float32), np.asarray(stats['control_std'], np.float32)


def action_blocks(commands, mean, std):
    """Executed commands (..., 5, 3) [vx, vy, wz] (or (..., 5, 2) [vx, wz]) to normalised action blocks (..., 10), or
    (..., 15) with three-component statistics."""
    commands = torch.as_tensor(commands, dtype=torch.float32)
    if commands.shape[-1] == 3 and len(mean) == 2:
        commands = commands[..., [0, 2]]
    mean, std = (torch.as_tensor(v, dtype=torch.float32, device=commands.device) for v in (mean, std))
    return ((commands-mean)/std).flatten(-2)


# -- encoder ----------------------------------------------------------------------------------------------------------

class ViTBlock(nn.Module):
    """Hugging Face ViT layer: pre-norm self-attention (qkv with bias) and GELU MLP, both residual."""

    def __init__(self, width, heads, mlp):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, eps=1e-12)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-12)
        self.mlp = nn.Sequential(nn.Linear(width, mlp), nn.GELU(), nn.Linear(mlp, width))

    def forward(self, x):
        b, n, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, w//self.heads).permute(2, 0, 3, 1, 4)
        x = x+self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w))
        return x+self.mlp(self.norm2(x))


class ViTEncoder(nn.Module):
    """Images (B, 3, H, W) to the final-layer [CLS] token (B, width).

    The patch embedding is the usual stride-p convolution computed as a reshape and a linear layer (weights in the
    convolution's (channel, row, column) order), which avoids the im2col copy that dominated it on ROCm.

    `checkpoint_blocks` (a runtime setting, not part of the model): in training, the first `checkpoint_blocks` blocks
    recompute their activations in the backward pass instead of storing them, trading compute for memory (H-JEPA
    encodes about 19 frames per sample, LeWM 12). 0, the default, stores everything."""

    checkpoint_blocks = 0

    def __init__(self, config):
        super().__init__()
        h, w = config.image_hw
        if h % config.patch or w % config.patch:
            raise ValueError('the patch size must divide the image')
        self.patch = config.patch
        self.patch_embed = nn.Linear(3*config.patch**2, config.width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.zeros(1, 1+(h//config.patch)*(w//config.patch), config.width))
        self.blocks = nn.ModuleList(ViTBlock(config.width, config.heads, config.mlp) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.width, eps=1e-12)
        self.apply(self._init)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.position, std=.02)

    @staticmethod
    def _init(module):  # the Hugging Face ViT initialisation
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patches(self, pixels):
        """(B, 3, H, W) to (B, H/p * W/p, 3 p p), patches in row-major order, values in (channel, row, column) order."""
        b, c, h, w = pixels.shape
        p = self.patch
        return pixels.reshape(b, c, h//p, p, w//p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h//p)*(w//p), c*p*p)

    def forward(self, pixels):
        x = self.patch_embed(self.patches(pixels))
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), dim=1)+self.position
        recompute = self.training and torch.is_grad_enabled()
        for i, block in enumerate(self.blocks):
            if recompute and i < self.checkpoint_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)[:, 0]


# -- the official modules ----------------------------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x*(1+scale)+shift


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regulariser: the Epps-Pulley statistic of M random 1-D projections (17 trapezoid
    knots on [0, 3], Gaussian window), times the sample count, averaged over projections and time steps."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3/(knots-1)
        weights = torch.full((knots,), 2*dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square()/2.)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights*window)

    def forward(self, proj, generator=None):
        """proj (T, B, D), float32. `generator` (on proj's device) draws the projections reproducibly."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, generator=generator)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj@A).unsqueeze(-1)*self.t
        err = (x_t.cos().mean(-3)-self.phi).square()+x_t.sin().mean(-3).square()
        statistic = (err@self.weights)*proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Causal scaled dot-product attention (its own LayerNorm, as in the official module)."""

    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner = dim_head*heads
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner*3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        b, n, _ = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.to_qkv(self.norm(x)).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=causal)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on each position's action embedding."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x+gate_msa*self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        return x+gate_mlp*self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class ARPredictor(nn.Module):
    """Autoregressive next-embedding predictor: z (B, T<=history, D) and action embeddings (B, T, D) to (B, T, D)."""

    def __init__(self, *, num_frames, depth, heads, dim_head, mlp_dim, dim, dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.layers = nn.ModuleList(ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        x = x+self.pos_embedding[:, :x.size(1)]
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class Embedder(nn.Module):
    """Action blocks (B, T, A) to embeddings (B, T, D): a per-step 1x1 convolution, then an MLP."""

    def __init__(self, input_dim, smoothed_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale*emb_dim), nn.SiLU(), nn.Linear(mlp_scale*emb_dim, emb_dim))

    def forward(self, x):
        return self.embed(self.patch_embed(x.float().transpose(1, 2)).transpose(1, 2))


class MLP(nn.Module):
    """Linear, BatchNorm, GELU, Linear (the projectors)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


# -- the world model --------------------------------------------------------------------------------------------------

class LeWorldModel(nn.Module):
    def __init__(self, config=LeWMConfig()):
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(config)
        self.projector = MLP(config.width, config.projector_hidden, config.latent)
        self.action_encoder = Embedder(config.action_dim, config.action_smoothed_dim, config.latent, config.action_mlp_scale)
        self.predictor = ARPredictor(num_frames=config.history, depth=config.predictor_depth,
                                     heads=config.predictor_heads, dim_head=config.predictor_head_dim,
                                     mlp_dim=config.predictor_mlp, dim=config.latent, dropout=config.predictor_dropout)
        self.pred_proj = MLP(config.latent, config.projector_hidden, config.latent)
        self.inverse = MLP(2*config.latent, config.projector_hidden, config.action_dim) if config.inverse_dynamics else None
        longer = [int(k) for k in config.inverse_spans if int(k) > 1] if config.inverse_dynamics else []
        if longer:  # one head per longer span (absent from single-span models, so their checkpoints load unchanged)
            self.inverse_long = nn.ModuleDict({str(k): MLP(2*config.latent, config.projector_hidden, config.action_dim)
                                               for k in longer})
        if config.motion_dynamics:
            self.motion = nn.ModuleDict({str(int(k)): MLP(2*config.latent, config.projector_hidden, 3)
                                         for k in config.inverse_spans})

    @classmethod
    def seeded(cls, seed, config=LeWMConfig()):
        torch.manual_seed(int(seed))
        return cls(config)

    def encode(self, pixels):
        """Normalised frames (B, T, 3, H, W) to z (B, T, D)."""
        b = pixels.size(0)
        return self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, (b, -1))

    def predict(self, z, actions):
        """z (B, T, D) and action blocks (B, T, A) to the predicted next z at every position (B, T, D)."""
        out = self.predictor(z, self.action_encoder(actions))
        return self.pred_proj(out.flatten(0, 1)).unflatten(0, (z.size(0), -1))

    def rollout(self, z, actions, steps):
        """Autoregressive prediction from the history: z (B, N, D) of the N history frames and action blocks
        (B, N-1+steps, A), the action after each history frame and then after each predicted frame. Step k attends over
        the last N frames (k..k+N-1, true then predicted) with their actions `actions[:, k:k+N]`. Returns the `steps`
        predicted z (B, steps, D)."""
        n = self.config.history
        if z.size(1) != n or actions.size(1) < n-1+steps:
            raise ValueError(f'{n} history frames and {n-1+steps} actions required')
        sequence, predicted = z, []
        for k in range(steps):
            nxt = self.predict(sequence[:, -n:], actions[:, k:k+n])[:, -1:]
            predicted.append(nxt)
            sequence = torch.cat((sequence, nxt), dim=1)
        return torch.cat(predicted, dim=1)

    def infer_actions(self, z, span=1):
        """The inverse-dynamics variant: the action block between consecutive frames, from [z_t, z_(t+1) - z_t];
        z (B, T, D) to (B, T-1, A). With `span` k > 1, the mean action block over frames t..t+k, from
        [z_t, z_(t+k) - z_t] (B, T-k, A)."""
        head = self.inverse if span == 1 else self.inverse_long[str(span)]
        pairs = torch.cat((z[:, :-span], z[:, span:]-z[:, :-span]), dim=-1)
        return head(pairs.flatten(0, 1)).unflatten(0, (z.size(0), -1))


MOTION_SCALE_PER_STEP = (.008, .008, .034)  # m, m, rad: the corpus's 100-ms body-frame motion spread (forward, yaw)


def body_motion(poses, span):
    """Planar poses (B, T, 3) (x, y, yaw) to the body-frame motion (dx, dy, dyaw) from frame t to t+span (B, T-span, 3)."""
    d = poses[:, span:, :2]-poses[:, :-span, :2]
    c, s = poses[:, :-span, 2].cos(), poses[:, :-span, 2].sin()
    dyaw = torch.remainder(poses[:, span:, 2]-poses[:, :-span, 2]+math.pi, 2*math.pi)-math.pi
    return torch.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1], dyaw), -1)


def losses(model, pixels, actions, *, rollout_steps, sigreg, lam, inverse_weight=0., generator=None, poses=None,
           motion_weight=0., z=None):
    """The training objective on windows of `history + rollout_steps` frames (rollout_steps >= 1).

    pixels (B, T, 3, H, W) normalised, actions (B, T-1, A); or, with `pixels` None, the frames' z (B, T, D) already
    encoded (H-JEPA encodes every level's frames in one pass). Returns a dict with `loss` and its terms:
    - `pred`: the paper's teacher-forced MSE, positions 0..N-1 of the history predicting frames 1..N;
    - `rollout` (rollout_steps > 1, the Go2 configuration): the MSE of the autoregressive rollout from the N history
      frames to frames N..N+K-1, as the forecaster uses the model. With K = 1 the objective is the paper's;
    - `sigreg`: SIGReg over every frame's z, per time step (float32);
    - `inverse` (inverse_weight > 0, the inverse-dynamics variant): the MSE of the action blocks decoded from
      consecutive z (`LeWorldModel.infer_actions`), which makes latent differences carry the camera's motion; with
      `inverse_spans` beyond 1, the mean over the spans (weighted by `inverse_span_weights`) of each span's MSE
      against the mean action block across it;
    - `motion` (motion_weight > 0, with `poses` (B, T, 3)): the same weighted mean of the MSE of the body-frame motion
      over each span, decoded from [z_t, z_(t+k) - z_t] and scaled by k x `MOTION_SCALE_PER_STEP`.
    """
    n = model.config.history
    z = model.encode(pixels) if z is None else z
    pred = model.predict(z[:, :n], actions[:, :n])
    terms = dict(pred=(pred-z[:, 1:n+1]).pow(2).mean())
    if rollout_steps > 1:
        rolled = model.rollout(z[:, :n], actions, rollout_steps)
        terms['rollout'] = (rolled-z[:, n:n+rollout_steps]).pow(2).mean()
    terms['sigreg'] = sigreg(z.float().transpose(0, 1), generator)
    terms['loss'] = terms['pred']+terms.get('rollout', 0.)+lam*terms['sigreg']
    if inverse_weight:
        sums = torch.cat((torch.zeros_like(actions[:, :1]), actions.cumsum(1)), dim=1)
        spans = [int(k) for k in model.config.inverse_spans]
        weights = [float(w) for w in model.config.inverse_span_weights] or [1.]*len(spans)
        terms['inverse'] = sum(w*(model.infer_actions(z, k)-(sums[:, k:]-sums[:, :-k])/k).pow(2).mean()
                               for k, w in zip(spans, weights))/sum(weights)
        terms['loss'] = terms['loss']+inverse_weight*terms['inverse']
    if motion_weight:
        spans = [int(k) for k in model.config.inverse_spans]
        weights = [float(w) for w in model.config.inverse_span_weights] or [1.]*len(spans)
        scale = torch.tensor(MOTION_SCALE_PER_STEP, device=z.device)
        def term(k):
            pairs = torch.cat((z[:, :-k], z[:, k:]-z[:, :-k]), dim=-1)
            decoded = model.motion[str(k)](pairs.flatten(0, 1)).unflatten(0, (z.size(0), -1))
            return (decoded-body_motion(poses.float(), k)/(k*scale)).pow(2).mean()
        terms['motion'] = sum(w*term(k) for k, w in zip(spans, weights))/sum(weights)
        terms['loss'] = terms['loss']+motion_weight*terms['motion']
    return terms


def save_world_model(path, model, *, extra=None):
    torch.save(dict(config=asdict(model.config), model_state_dict=model.state_dict(), **(extra or {})), path)


def load_world_model(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(state['config'])
    config['image_hw'] = tuple(config['image_hw'])
    model = LeWorldModel(LeWMConfig(**config))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)


# -- motion heads (C5, C6) --------------------------------------------------------------------------------------------

class LatentMotionDecoder(nn.Module):
    """C5's decoder: the body-frame motion (x, y, yaw) between the current frame and a predicted future one, read
    from LeWM latents only (it never sees the commands, as C3's decoder does not): per horizon, an MLP on
    [z_t, z_hat(t+h) - z_t, z_t - z_(t-1), z_t - z_(t-3)], through a per-channel target normalisation."""

    def __init__(self, latent, mean, scale, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*latent, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
        self.register_buffer('target_mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('target_scale', torch.as_tensor(scale, dtype=torch.float32).clone())

    def normalized(self, z, future):
        """z (B, 4, D) of frames t-3..t; future (B, D) the predicted z at t+h."""
        current = z[:, -1]
        return self.net(torch.cat((current, future-current, current-z[:, -2], current-z[:, 0]), dim=-1))

    def forward(self, z, future):
        return self.normalized(z, future)*self.target_scale+self.target_mean


class LatentDirectPredictor(nn.Module):
    """C6, C4's counterpart on LeWM's own encoder: motion at a horizon from the four frames' z, the normalised 1.5-s
    applied-command history (15 x (forward, yaw)) and the candidate's commands up to the horizon (later ones zeroed),
    with no predicted future."""

    def __init__(self, latent, mean, scale, hidden=896):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*latent+30+16+1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
        self.register_buffer('target_mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('target_scale', torch.as_tensor(scale, dtype=torch.float32).clone())

    def normalized(self, z, past, tape, horizon):
        """z (B, 4, D); past (B, 15, 2) normalised; tape (B, 8, 2) normalised commands; horizon (B,) in 1-8."""
        steps = torch.arange(8, device=z.device)[None]
        future = tape*(steps < horizon[:, None])[:, :, None]
        return self.net(torch.cat((z.flatten(1), past.flatten(1), future.flatten(1), horizon[:, None].float()/8), dim=1))

    def forward(self, z, past, tape, horizon):
        return self.normalized(z, past, tape, horizon)*self.target_scale+self.target_mean


def load_heads(path, device='cpu'):
    """The C5 decoder and C6 of a fit file ({'c5', 'c6', 'latent', 'target_mean', 'target_scale'})."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    c5 = LatentMotionDecoder(state['latent'], state['target_mean'], state['target_scale'])
    c5.load_state_dict(state['c5'])
    c6 = LatentDirectPredictor(state['latent'], state['target_mean'], state['target_scale'])
    c6.load_state_dict(state['c6'])
    return c5.to(device).eval().requires_grad_(False), c6.to(device).eval().requires_grad_(False)
