"""C001: RGB SAVi-style adaptation and anchored object-masked latent predictor.

Implemented from C-JEPA v2 Sections 4.1/4.2 and Appendices D/E. This is not
the stochastic SAVi implementation: RGB MSE, no KL term. See research record.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.models.lewm import ViTBlock
from go2nav.models.norm import LayerNorm
from go2nav.models.slots import SlotAttention


def coordinates(size):
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    return torch.stack((x, y, 1-x, 1-y), -1)


@dataclass(frozen=True)
class RGBSlotConfig:
    slots: int = 5
    dim: int = 128
    iterations: int = 2
    channels: int = 64


class RGBSlots(nn.Module):
    def __init__(self, config=RGBSlotConfig()):
        super().__init__()
        self.config = config
        d, c = config.dim, config.channels
        layers, previous = [], 3
        for _ in range(4):
            layers.extend((nn.Conv2d(previous, c, 5, padding=2), nn.ReLU()))
            previous = c
        self.conv = nn.Sequential(*layers)
        self.position = nn.Linear(4, c)
        self.features = nn.Sequential(LayerNorm(c), nn.Linear(c, d), nn.ReLU(), nn.Linear(d, d))
        self.attention = SlotAttention(d, config.iterations, 256)
        self.carry = ViTBlock(d, 4, 256)
        self.mu = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, d))
        self.decoder_position = nn.Linear(4, d)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(d, c, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(c, c, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(c, c, 4, 2, 1), nn.ReLU(), nn.Conv2d(c, 4, 3, padding=1))
        self.register_buffer('grid64', coordinates(64), persistent=False)
        self.register_buffer('grid8', coordinates(8), persistent=False)
        noise = torch.randn(1, config.slots, d, generator=torch.Generator().manual_seed(2026111506))
        self.register_buffer('eval_noise', noise)

    def initial(self, batch, generator=None):
        noise = torch.randn(batch, self.config.slots, self.config.dim, generator=generator, device=self.mu.device) \
            if self.training else self.eval_noise.expand(batch, -1, -1)
        return self.mu+self.log_sigma.exp()*noise

    def forward(self, pixels, slots=None, generator=None):
        b, t = pixels.shape[:2]
        x = self.conv(pixels.flatten(0, 1)).movedim(1, -1)+self.position(self.grid64)
        x = self.features(x).reshape(b, t, 64*64, self.config.dim)
        states = []
        for i in range(t):
            if slots is None:
                start = self.initial(b, generator)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                    start = self.carry(slots)
            slots, _ = self.attention(x[:, i], start)
            states.append(slots)
        return torch.stack(states, 1)

    def decode(self, slots):
        b, t, k, d = slots.shape
        x = slots.reshape(-1, 1, 1, d)+self.decoder_position(self.grid8)
        values = self.decoder(x.movedim(-1, 1)).reshape(b, t, k, 4, 64, 64)
        masks = values[:, :, :, 3:4].softmax(2)
        return (values[:, :, :, :3]*masks).sum(2), masks.squeeze(3)


class MaskedBlock(nn.Module):
    def __init__(self, dim=128, heads=16, head_dim=64, mlp=2048):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm1, self.norm2 = LayerNorm(dim), LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*heads*head_dim)
        self.proj = nn.Linear(heads*head_dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        b, length = x.shape[:2]
        q, k, v = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        with sdpa_kernel(SDPBackend.MATH):
            x = x+self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, -1))
        return x+self.ff(self.norm2(x))


class ObjectMaskedPredictor(nn.Module):
    def __init__(self, dim=128, depth=6, heads=16, head_dim=64, mlp=2048):
        super().__init__()
        self.config = dict(dim=dim, depth=depth, heads=heads, head_dim=head_dim, mlp=mlp)
        self.anchor = nn.Linear(dim, dim)
        self.time = nn.Parameter(torch.randn(1, 4, 1, dim)*.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, dim)*.02)
        self.action = nn.Conv1d(2, dim, 1)
        self.action_type = nn.Parameter(torch.randn(1, 1, dim)*.02)
        self.blocks = nn.Sequential(*(MaskedBlock(dim, heads, head_dim, mlp) for _ in range(depth)))
        self.out = nn.Sequential(LayerNorm(dim), nn.Linear(dim, dim))

    def tokens(self, history, actions, masked=None):
        b, t, k, d = history.shape
        if t != 3 or actions.shape != (b, 3, 2):
            raise ValueError('Three history frames and their three following actions are required')
        if masked is None:
            masked = torch.zeros(b, k, dtype=torch.bool, device=history.device)
        anchor = self.anchor(history[:, :1])
        selection = torch.cat((torch.zeros_like(masked[:, None]), masked[:, None].expand(-1, 2, -1),
                               torch.ones_like(masked[:, None])), 1)
        available = torch.cat((history, torch.zeros_like(history[:, :1])), 1)
        tokens = torch.where(selection[..., None], anchor+self.mask_token, available)+self.time
        action = self.action(actions.transpose(1, 2)).transpose(1, 2)+self.time[:, :3, 0]+self.action_type
        return torch.cat((tokens.flatten(1, 2), action), 1), selection

    def forward(self, history, actions, masked=None):
        tokens, selection = self.tokens(history, actions, masked)
        b, _, k, d = history.shape
        predicted = self.out(self.blocks(tokens))[:, :4*k].reshape(b, 4, k, d)
        return predicted, selection

    def rollout(self, history, actions, steps):
        sequence, predictions = history, []
        for i in range(steps):
            pred, _ = self(sequence[:, -3:], actions[:, i:i+3])
            next_ = pred[:, -1:]
            sequence = torch.cat((sequence, next_), 1)
            predictions.append(next_)
        return torch.cat(predictions, 1)


def object_mask(batch, slots, generator, device):
    count = torch.randint(3, (batch,), generator=generator, device=device)
    order = torch.rand(batch, slots, generator=generator, device=device).argsort(-1).argsort(-1)
    return order < count[:, None]


def predictor_loss(model, states, actions, masked):
    predicted, selection = model(states[:, :3], actions, masked)
    error = (predicted-states.detach()).square().mean(-1)
    history_mask = selection[:, :3]
    past = (error[:, :3]*history_mask).sum()/history_mask.sum().clamp_min(1)
    future = error[:, 3].mean()
    return future+past, dict(future=future, masked_history=past)


class SlotWorld(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder, self.predictor = encoder, predictor

    def encode(self, pixels):
        return self.encoder(pixels)

    def rollout(self, history, actions, steps):
        return self.predictor.rollout(history, actions, steps)
