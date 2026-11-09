"""H-JEPA (Zhang, Terver, Rabbat, LeCun and Balestriero, arXiv:2610.06805) for the Go2: a stack of action-conditioned
JEPAs trained jointly, end to end, with the LeWorldModel (`models.lewm`) as level 1. The design, its settings and the
timescales are argued in docs/hjepa/PLAN.md and docs/hjepa/TIMESCALES.md.

**Levels.** Level 1 is a `LeWorldModel` on frames 100 ms apart. Each upper level l (2, 3, ...) has a stride `s_l`: its
consecutive states are `s_l` states of the level below apart (window 1, as in every experiment of the paper), so with
strides 16 and 8 the levels step 100 ms, 1.6 s and 12.8 s. An upper level holds:
- a state encoder, a two-layer MLP with BatchNorm applied to each state of the level below on its own, so a level-l
  state is the composition of the encoders up to l applied to one level-1 z;
- a macro-action encoder (DROID's form): the `s_l` lower actions of one transition concatenated (raw normalised
  command blocks at level 2, lower macro-actions above), an MLP and a LayerNorm to `macro` dimensions;
- a causal AdaLN-zero predictor over `context` states (level 1's block, `models.lewm.ARPredictor`) with a
  BatchNorm projector, conditioned on each transition's macro-action embedding;
- an inverse-dynamics head: the (detached) macro-action from [S_t, S_(t+1) - S_t].

**Objective** (`losses`): one sum over levels. Level 1 has its own LeWM loss (teacher-forced and rollout terms,
SIGReg, inverse dynamics). Each upper level has a teacher-forced one-step prediction loss over its states, plus
`sigreg` x SIGReg of its states, `inverse` x the inverse-dynamics MSE and `action_sigreg` x SIGReg of its macro-action
embeddings. There is no stop-gradient and no EMA target: upper levels' gradients flow through every encoder below,
level 1's ViT included. Level 1 and the upper levels train on their own samples of one batch (`data.hjepa_dataset`:
level 1 on short windows of every unit, the upper levels on long clips of mission and explore units).
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from torch.nn import functional as F

from go2nav.models.lewm import (MLP, MOTION_SCALE_PER_STEP, ARPredictor, LeWMConfig, LeWorldModel, body_motion,
                                losses as level1_losses)
from go2nav.models.norm import LayerNorm


@dataclass(frozen=True)
class UpperConfig:
    stride: int
    context: int                 # states the predictor attends over
    latent: int = 192
    encoder_hidden: int = 768
    projector_hidden: int = 2048
    macro: int = 8
    macro_hidden: int = 64
    predictor_depth: int = 4
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    predictor_mlp: int = 2048
    predictor_dropout: float = .1
    rollout_steps: int = 0       # P15: also an autoregressive rollout of this many steps from the context (0: none)

    @property
    def crop(self):
        """States per training sample: the context and the one it predicts, or the context and the rollout's."""
        return self.context+max(1, self.rollout_steps)


@dataclass(frozen=True)
class HJEPAConfig:
    level1: LeWMConfig = LeWMConfig()
    upper: tuple = (UpperConfig(stride=16, context=3), UpperConfig(stride=8, context=2))
    heads: bool = False          # P15: the stop-gradient heads (`Heads`)

    @property
    def periods(self):
        """Level-1 steps between consecutive states of each level (1 for level 1)."""
        out = [1]
        for u in self.upper:
            out.append(out[-1]*u.stride)
        return tuple(out)

    @property
    def clip_length(self):
        """Level-1 frames of a training clip: the top level's crop."""
        return (self.upper[-1].crop-1)*self.periods[-1]+1 if self.upper else self.level1.history

    def to_dict(self):
        return dict(level1=asdict(self.level1), upper=[asdict(u) for u in self.upper], heads=self.heads)

    @classmethod
    def from_dict(cls, d):
        level1 = dict(d['level1'])
        for key in ('image_hw', 'inverse_spans', 'inverse_span_weights', 'grid'):
            if key in level1:
                level1[key] = tuple(level1[key])
        return cls(level1=LeWMConfig(**level1), upper=tuple(UpperConfig(**u) for u in d['upper']),
                   heads=bool(d.get('heads', False)))


CONTACT_M = .10                                  # `data.head_labels.CONTACT_M`
MOTION_SCALE_LEVEL2 = (.5, .5, .8)               # m, m, rad: the spread of 1.6-s body-frame motion


class Heads(nn.Module):
    """P15's heads, read from the latent with stop-gradient (they never shape it; docs/dynamic/PLAN.md P15):
    - `contact`: whether the body outline is within `CONTACT_M` of a wall or walker at this frame (a logit), from a
      level-1 z; the gate reads it on the model's 0.8-s rollout;
    - `presence`: a walker within 2 m, in the camera's view and in the ring's line of sight (two logits), from a z;
    - `motion`: the body-frame motion (dx, dy, dyaw) over one step, from [z_t, z_(t+1) - z_t], over
      `MOTION_SCALE_PER_STEP`;
    - `motion2`: the same over one level-2 step (1.6 s), from level-2 states, over `MOTION_SCALE_LEVEL2`."""

    def __init__(self, latent1, latent2, hidden=512):
        super().__init__()
        self.contact = MLP(latent1, hidden, 1)
        self.presence = MLP(latent1, hidden, 2)
        self.motion = MLP(2*latent1, hidden, 3)
        self.motion2 = MLP(2*latent2, hidden, 3)

    @staticmethod
    def pairs(states):
        return torch.cat((states[:, :-1], states[:, 1:]-states[:, :-1]), -1)

    def run(self, head, x):
        return head(x.flatten(0, -2)).unflatten(0, x.shape[:-1])


class UpperLevel(nn.Module):
    """One upper level (see the module docstring)."""

    def __init__(self, config, lower_latent, lower_action):
        super().__init__()
        self.config = config
        c = config
        self.encoder = MLP(lower_latent, c.encoder_hidden, c.latent)
        self.action_encoder = nn.Sequential(nn.Linear(c.stride*lower_action, c.macro_hidden), nn.GELU(),
                                            nn.Linear(c.macro_hidden, c.macro), LayerNorm(c.macro))
        self.action_embed = nn.Sequential(nn.Linear(c.macro, 4*c.latent), nn.SiLU(), nn.Linear(4*c.latent, c.latent))
        self.predictor = ARPredictor(num_frames=c.context, depth=c.predictor_depth, heads=c.predictor_heads,
                                     dim_head=c.predictor_head_dim, mlp_dim=c.predictor_mlp, dim=c.latent,
                                     dropout=c.predictor_dropout)
        self.pred_proj = MLP(c.latent, c.projector_hidden, c.latent)
        self.inverse = MLP(2*c.latent, c.projector_hidden, c.macro)

    def encode(self, lower):
        """Lower-level states (B, T, D_lower) to this level's (B, T, D), each on its own."""
        return self.encoder(lower.flatten(0, 1)).unflatten(0, (lower.size(0), -1))

    def macro(self, lower_actions):
        """Lower actions (B, T x stride, A_lower) to macro-actions (B, T, macro): one per `stride` consecutive."""
        b, n, a = lower_actions.shape
        if n % self.config.stride:
            raise ValueError(f'{n} lower actions do not group into strides of {self.config.stride}')
        return self.action_encoder(lower_actions.float().reshape(b, n//self.config.stride, self.config.stride*a))

    def predict(self, states, macros):
        """States (B, T<=context, D) and their transitions' macro-actions (B, T, macro) to the next state at every
        position (B, T, D)."""
        out = self.predictor(states, self.action_embed(macros))
        return self.pred_proj(out.flatten(0, 1)).unflatten(0, (states.size(0), -1))

    def infer(self, states):
        """The macro-action of each consecutive pair, from [S_t, S_(t+1) - S_t]: (B, T-1, macro)."""
        pairs = torch.cat((states[:, :-1], states[:, 1:]-states[:, :-1]), dim=-1)
        return self.inverse(pairs.flatten(0, 1)).unflatten(0, (states.size(0), -1))


class HJEPA(nn.Module):
    def __init__(self, config=HJEPAConfig()):
        super().__init__()
        self.config = config
        self.level1 = LeWorldModel(config.level1)
        latent, action = config.level1.latent, config.level1.action_dim
        levels = []
        for u in config.upper:
            levels.append(UpperLevel(u, latent, action))
            latent, action = u.latent, u.macro
        self.upper = nn.ModuleList(levels)
        self.heads = Heads(config.level1.latent, config.upper[0].latent) if config.heads else None

    @classmethod
    def seeded(cls, seed, config=HJEPAConfig()):
        torch.manual_seed(int(seed))
        return cls(config)

    def states(self, z1, level):
        """Level-1 z (B, T, D1) to level-`level` states (B, T, D): the encoders up to that level, composed."""
        x = z1
        for k in range(level-1):
            x = self.upper[k].encode(x)
        return x

    def macros(self, actions, level):
        """Level-1 action blocks (B, N, A1) to level-`level` macro-actions (B, N / period, macro)."""
        x = actions
        for k in range(level-1):
            x = self.upper[k].macro(x)
        return x

    def parameter_groups(self):
        """Each level's parameters (for per-level gradient clipping)."""
        return [list(self.level1.parameters())]+[list(u.parameters()) for u in self.upper]


def upper_rollout(level, states, macros, steps):
    """The autoregressive rollout from a crop's first `context` states: (B, steps, D), the states after them."""
    n = level.config.context
    sequence = states[:, :n]
    for k in range(steps):
        sequence = torch.cat((sequence, level.predict(sequence[:, -n:], macros[:, k:k+n])[:, -1:]), 1)
    return sequence[:, n:]


def upper_terms(level, states, macros, sigreg, *, generator=None):
    """One upper level's unweighted terms on a crop: states (B, crop, D), macros (B, crop-1, macro). The
    teacher-forced prediction covers the first context + 1 states; with `rollout_steps` (P15), `rollout` is the MSE
    of the autoregressive rollout over the rest."""
    n, steps = level.config.context, level.config.rollout_steps
    pred = level.predict(states[:, :n], macros[:, :n])
    terms = dict(pred=(pred-states[:, 1:n+1]).pow(2).mean(),
                 sigreg=sigreg(states.float().transpose(0, 1), generator),
                 inverse=(level.infer(states)-macros.detach()).pow(2).mean(),
                 action_sigreg=sigreg(macros.float().transpose(0, 1), generator))
    if steps:
        terms['rollout'] = (upper_rollout(level, states, macros, steps)-states[:, n:n+steps]).pow(2).mean()
    return terms


def head_terms(model, z1, poses1, labels, states2, poses2):
    """The heads' losses (P15), all on detached inputs: z1 (B, T, D1) level-1 z of the windows with their planar poses
    (B, T, 3) and `labels` (clearance (B, T), presence (B, T, 2)); states2 (B, c, D2) level-2 states of the clips with
    their poses (B, c, 3)."""
    h, z1, states2 = model.heads, z1.detach(), states2.detach()
    contact = (labels['clearance'] < CONTACT_M).float()
    scale1 = torch.tensor(MOTION_SCALE_PER_STEP, device=z1.device)
    scale2 = torch.tensor(MOTION_SCALE_LEVEL2, device=z1.device)
    return dict(
        contact=F.binary_cross_entropy_with_logits(h.run(h.contact, z1)[..., 0].float(), contact),
        presence=F.binary_cross_entropy_with_logits(h.run(h.presence, z1).float(), labels['presence'].float()),
        motion=(h.run(h.motion, h.pairs(z1)).float()-body_motion(poses1.float(), 1)/scale1).pow(2).mean(),
        motion2=(h.run(h.motion2, h.pairs(states2)).float()-body_motion(poses2.float(), 1)/scale2).pow(2).mean())


def losses(model, batch, *, level1, upper, sigreg, generator=None):
    """The joint objective on a batch (`data.hjepa_dataset`), all frames normalised:
    - batch['frames']: per level, level 1's windows (B1, history + rollout_steps, 3, H, W), then each upper level's
      crops of states (Bu, crop, 3, H, W);
    - batch['actions']: per level, level 1's action blocks (B1, T - 1, A1), then for each upper level the level-1 blocks
      spanning its crops' transitions (Bu, (crop - 1) x period, A1).
    - batch['obs'] (P15, a model with `ObservationTokens`): per level, the frames' other sensors (..., obs_dim);
    - batch['poses'] and batch['labels'] (P15, a model with heads): level 1's and level 2's planar poses (B, T, 3), and
      level 1's head labels (`data.head_labels`: clearance (B1, T), presence (B1, T, 2)).
    `level1`: keyword arguments of `models.lewm.losses` (rollout_steps, lam, inverse_weight); `upper`: per upper level a
    dict of weights (sigreg, inverse, action_sigreg, and rollout, default 1). Returns the terms per level and `loss`,
    their weighted sum."""
    frames, actions, obs = batch['frames'], batch['actions'], batch.get('obs')
    if frames[0].size(1) != model.config.level1.history+level1['rollout_steps']:
        raise ValueError('level 1 frames must be history + rollout_steps')
    # One encoder pass over every level's frames (with their other sensors, P15).
    flat = torch.cat([f.flatten(0, 1) for f in frames])
    z_all = model.level1.embed(flat, None if obs is None else torch.cat([o.flatten(0, 1) for o in obs]))
    parts, start = [], 0
    for f in frames:
        n = f.size(0)*f.size(1)
        parts.append(z_all[start:start+n].unflatten(0, (f.size(0), f.size(1))))
        start += n
    terms = {f'l1/{k}': v for k, v in level1_losses(model.level1, None, actions[0], sigreg=sigreg, generator=generator,
                                                      z=parts[0], **level1).items()}
    total = terms['l1/loss']
    for k, weights in enumerate(upper):
        level = k+2
        states = model.states(parts[k+1], level)
        macros = model.macros(actions[k+1], level)
        t = upper_terms(model.upper[k], states, macros, sigreg, generator=generator)
        t['loss'] = t['pred']+weights['sigreg']*t['sigreg']+weights['inverse']*t['inverse'] \
            + weights['action_sigreg']*t['action_sigreg']+weights.get('rollout', 1.)*t.get('rollout', 0.)
        terms |= {f'l{level}/{key}': v for key, v in t.items()}
        total = total+t['loss']
        if k == 0:
            states2 = states
    if model.heads is not None:
        h = head_terms(model, parts[0], batch['poses'][0], batch['labels'], states2, batch['poses'][1])
        terms |= {f'heads/{key}': v for key, v in h.items()}
        total = total+sum(h.values())          # stop-gradient: these move the heads only
    terms['loss'] = total
    return terms


def save_hjepa(path, model, *, extra=None):
    torch.save(dict(config=model.config.to_dict(), model_state_dict=model.state_dict(), **(extra or {})), path)


def load_hjepa(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = HJEPA(HJEPAConfig.from_dict(state['config']))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)


def export_level1(model, path, *, extra=None):
    """Level 1 alone in the LeWM checkpoint format (`models.lewm.load_world_model`): what C7/C8 and every LeWM tool
    (readouts, decoders, the turn gate, the stuck replay) load."""
    from go2nav.models.lewm import save_world_model
    save_world_model(path, model.level1, extra=dict(extra or {}, source='hjepa_level1'))


def period_seconds(config, step_s=.1):
    """Each level's step in seconds."""
    return tuple(round(p*step_s, 6) for p in config.periods)

