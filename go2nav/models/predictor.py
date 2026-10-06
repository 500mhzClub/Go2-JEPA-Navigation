"""The JEPA predictor: from three past frames and a planned action, predict the V-JEPA tokens of a future frame.

**Inputs.**
- `context` (B, 3, 768, 1024): layer-normalised V-JEPA tokens of the frames 1 s, 0.5 s and 0 s before the decision.
- `action`: the planned commands (see below).
- `control` (B, 3, 5, 2): the robot's own applied commands (forward, yaw) over the last 1.5 s, five per context frame,
  normalised by the training statistics (an efference copy, available at run time).
- `mask` (B, 768): which target tokens to predict (all of them in this project).

**Architecture.** A 6-block transformer of width 384 (6 heads) over:
- 3 x 768 context tokens (linear projection plus learned spatial and per-frame temporal embeddings);
- 3 control tokens, one per context frame;
- 768 target queries (a learned mask token plus spatial and target-time embeddings).
The action conditions every block through adaptive layer norm (AdaLN-Zero: scale, shift and gate, initialised to zero).
The output is the predicted target token grid (768 x 1024).

**Action inputs over the training lineage.**
- `DensePredictor` (stages T1-T3): 10 values, the five applied commands (forward, yaw) of the 500-ms action, predicting
  the frame 500 ms ahead.
- `HorizonConditionedPredictor` (stage T4, the deployed predictor): 17 values, the eight applied commands of an
  800-ms tape (commands at or after the target horizon zeroed) plus the target time, predicting the frame 100-800 ms
  ahead. The extra inputs enter through a zero-initialised extension of the first action layer, so T4 starts exactly
  as its T3 parent.

Parameter names and creation order match the original modules, so the shipped checkpoints load directly and a seeded
initialisation reproduces the original one.
"""
import torch
from torch import nn
from torch.nn import functional as F

from go2nav.models.vjepa import TOKEN_DIM, TOKENS

CONTEXT_FRAMES = 3
ACTION_DIM = 10           # five commands x (forward, yaw)
CONTROL_SLOT_DIM = 10     # five applied commands x (forward, yaw) per context frame
HORIZON_ACTION_DIM = 17   # eight commands x (forward, yaw) + target time
HORIZON_STEPS = 8         # 100-800 ms in 100-ms steps


class PredictorBlock(nn.Module):
    """Pre-norm transformer block with action AdaLN-Zero (scale, shift, gate for attention and MLP)."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, width*3)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
        self.ada = nn.Linear(width, width*6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, conditioning):
        s1, b1, g1, s2, b2, g2 = self.ada(conditioning).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x)*(1+s1)+b1
        b, n, w = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, w//self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x+g1*self.proj(a.transpose(1, 2).reshape(b, n, w))
        h = self.norm2(x)*(1+s2)+b2
        return x+g2*self.mlp(h)


class DensePredictor(nn.Module):
    def __init__(self, token_dim=TOKEN_DIM, width=384, depth=6, heads=6):
        super().__init__()
        self.width = width
        self.input = nn.Linear(token_dim, width)
        self.output = nn.Linear(width, token_dim)
        self.spatial = nn.Parameter(torch.zeros(1, TOKENS, width))
        self.temporal = nn.Parameter(torch.zeros(CONTEXT_FRAMES+1, 1, width))  # three context frames + the target
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.spatial, std=0.02)
        nn.init.trunc_normal_(self.temporal, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.action = nn.Sequential(nn.Linear(ACTION_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.control_in = nn.Linear(CONTROL_SLOT_DIM, width)
        self.control_modality = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.control_modality, std=0.02)
        self.blocks = nn.ModuleList([PredictorBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    @classmethod
    def seeded(cls, seed, **kwargs):
        """A fresh predictor initialised from `seed` (as the T1 training did)."""
        torch.manual_seed(int(seed))
        return cls(**kwargs)

    def forward(self, context, action, mask, *, control):
        b, t, n, _ = context.shape
        if control.shape != (b, t, 5, 2):
            raise ValueError(f'control must be (B, {t}, 5, 2); got {tuple(control.shape)}')
        x = self.input(context.reshape(b*t, n, -1)).reshape(b, t, n, self.width)
        x = (x+self.spatial.unsqueeze(1)+self.temporal[:CONTEXT_FRAMES].unsqueeze(0)).reshape(b, t*n, self.width)
        control_tokens = self.control_in(control.reshape(b, t, CONTROL_SLOT_DIM))+self.control_modality \
            + self.temporal[:CONTEXT_FRAMES].squeeze(1)
        query = self.mask_token.expand(b, n, -1)+self.spatial+self.temporal[CONTEXT_FRAMES]
        sequence = torch.cat([x, control_tokens, query], dim=1)
        conditioning = self.action(action)
        for block in self.blocks:
            sequence = block(sequence, conditioning)
        return self.output(self.norm(sequence[:, -n:]))


class ExtendedActionInput(nn.Module):
    """First action layer of T4: the T3 layer on the first 10 inputs plus a zero-initialised layer on the 7 new ones."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.extension = nn.Linear(HORIZON_ACTION_DIM-ACTION_DIM, original.out_features, bias=False,
                                   device=original.weight.device, dtype=original.weight.dtype)
        nn.init.zeros_(self.extension.weight)

    def forward(self, action):
        return self.original(action[:, :ACTION_DIM].contiguous())+self.extension(action[:, ACTION_DIM:].contiguous())


class HorizonConditionedPredictor(nn.Module):
    """T4: predict the frame `horizon` x 100 ms ahead (1-8) given the 800-ms command tape.

    With `action_blind`, the future commands are zeroed (the no-future-action control arm); the target time stays.
    """

    def __init__(self, parent, *, action_blind=False):
        super().__init__()
        self.backbone = parent
        self.backbone.action[0] = ExtendedActionInput(self.backbone.action[0])
        self.action_blind = action_blind

    @staticmethod
    def condition(future_commands, horizon, *, action_blind=False):
        """(B, 8, 2) commands and (B,) integer horizons in 1-8 to the 17-value action input."""
        if future_commands.shape[1:] != (HORIZON_STEPS, 2):
            raise ValueError('(B, 8, 2) forward/yaw commands required')
        h = torch.as_tensor(horizon, device=future_commands.device)
        if not bool(((h >= 1) & (h <= HORIZON_STEPS)).all()):
            raise ValueError('horizon must be 1-8 (100-800 ms)')
        valid = torch.arange(HORIZON_STEPS, device=h.device)[None] < h[:, None]
        commands = torch.where(valid[:, :, None], future_commands, torch.zeros_like(future_commands))
        if action_blind:
            commands = torch.zeros_like(commands)
        time = (h.to(dtype=commands.dtype)-5)/5
        return torch.cat((commands.flatten(1), time[:, None]), dim=1)

    def forward(self, context, future_commands, horizon, mask, *, control):
        condition = self.condition(future_commands, horizon, action_blind=self.action_blind)
        return self.backbone(context, condition, mask, control=control)


def load_horizon_predictor(path, *, action_blind=False):
    """The deployed T4 predictor from a checkpoint ({'model_state_dict': ...})."""
    model = HorizonConditionedPredictor(DensePredictor(), action_blind=action_blind)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    return model.eval().requires_grad_(False)
