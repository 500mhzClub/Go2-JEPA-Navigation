"""Self-supervised object slots on frozen DINOv2 patch features (docs/dynamic/PLAN.md P23 item 3): VideoSAUR's recipe
(Zadaianchuk, Seitzer and Martius, NeurIPS 2023, arXiv 2306.04829) as C-JEPA (arXiv 2602.11389) trained its encoder.

- **Inputs:** a frame's 16 x 12 DINOv2 ViT-S/14 patch tokens (384), through a LayerNorm and an MLP to `dim`, plus a
  learned position embedding per patch.
- **Slot Attention** (Locatello et al. 2020): `slots` slots compete for the patches (softmax over slots) for
  `iterations` rounds, each a GRU update and a residual MLP.
- **Over time:** the first frame's slots are drawn from a learned Gaussian; each later frame's start from the last
  frame's through one transformer block over the slots (`carry`), so a slot keeps its object. The block's attention
  runs on the plain (math) kernel: on this ROCm build the fused kernels, over 6 tokens, made illegal memory accesses
  within 50-200 training steps (P23; with the math kernel 400 steps ran clean).
- **Decoder** (DINOSAUR's MLP broadcast decoder): each slot, with a learned embedding per patch, through an MLP to
  every patch's features, a mask logit, and a row of logits over the next frame's patches; the masks are a softmax
  over slots, and the reconstruction and the similarity logits are the mask-weighted sums.
- **Loss** (`losses`): the reconstructed features' squared error; plus `similarity_weight` x the cross-entropy of
  the predicted next-frame affinity against VideoSAUR's target, softmax over the next frame's patches of the cosine
  similarity of the frozen features over `temperature` (the motion bias); plus `sigreg_weight` x SIGReg of the slot
  latents (each slot of each frame a sample), for a later hierarchy (docs/REVIEW_2026-11-11.md section 5).
`masks` (frames, slots, 192) is what P23's measures read: the image columns a slot covers.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.models.lewm import SIGReg, ViTBlock
from go2nav.models.norm import LayerNorm

PATCHES, FEATURES = 192, 384


@dataclass(frozen=True)
class SlotConfig:
    slots: int = 6
    dim: int = 128
    iterations: int = 2
    mlp: int = 256
    decoder_hidden: int = 512
    decoder_layers: int = 3
    carry_heads: int = 4
    similarity: bool = True

    def to_dict(self):
        return asdict(self)


class SlotAttention(nn.Module):
    def __init__(self, dim, iterations, hidden, eps=1e-8):
        super().__init__()
        self.iterations, self.eps, self.scale = iterations, eps, dim**-.5
        self.norm_inputs, self.norm_slots, self.norm_mlp = LayerNorm(dim), LayerNorm(dim), LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, inputs, slots):
        """inputs (B, N, D), slots (B, K, D) to slots (B, K, D) and attention (B, K, N) (summing to 1 over slots)."""
        b, k, d = slots.shape
        x = self.norm_inputs(inputs)
        keys, values = self.k(x), self.v(x)
        for _ in range(self.iterations):
            previous = slots
            q = self.q(self.norm_slots(slots))
            attention = torch.einsum('bkd,bnd->bkn', q, keys).mul(self.scale).softmax(1)+self.eps
            weights = attention/attention.sum(-1, keepdim=True)
            updates = torch.einsum('bkn,bnd->bkd', weights, values)
            slots = self.gru(updates.reshape(b*k, d), previous.reshape(b*k, d)).reshape(b, k, d)
            slots = slots+self.mlp(self.norm_mlp(slots))
        return slots, attention


class SlotModel(nn.Module):
    def __init__(self, config=SlotConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.encoder = nn.Sequential(LayerNorm(FEATURES), nn.Linear(FEATURES, FEATURES), nn.ReLU(), nn.Linear(FEATURES, d))
        self.position = nn.Parameter(torch.randn(1, PATCHES, d)*.02)
        self.mu = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, d))
        self.attention = SlotAttention(d, config.iterations, config.mlp)
        self.carry = ViTBlock(d, config.carry_heads, config.mlp)
        self.decoder_position = nn.Parameter(torch.randn(1, 1, PATCHES, d)*.02)
        layers, width = [], d
        for _ in range(config.decoder_layers):
            layers += [nn.Linear(width, config.decoder_hidden), nn.ReLU()]
            width = config.decoder_hidden
        out = FEATURES+1+(PATCHES if config.similarity else 0)
        self.decoder = nn.Sequential(*layers, nn.Linear(width, out))

    def initial(self, batch, generator=None):
        noise = torch.randn(batch, self.config.slots, self.config.dim, device=self.mu.device, generator=generator)
        return self.mu+self.log_sigma.exp()*noise

    def forward(self, features, slots=None, generator=None):
        """Features (B, T, 192, 384) to slots (B, T, K, D) and attention masks (B, T, K, 192); `slots` (B, K, D)
        continues from earlier frames."""
        b, t = features.shape[:2]
        x = self.encoder(features)+self.position
        out, masks = [], []
        for i in range(t):
            if slots is None:
                start = self.initial(b, generator)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                    start = self.carry(slots)
            slots, attention = self.attention(x[:, i], start)
            out.append(slots), masks.append(attention)
        return torch.stack(out, 1), torch.stack(masks, 1)

    def decode(self, slots):
        """Slots (B, T, K, D) to reconstructed features (B, T, 192, 384), similarity logits (B, T, 192, 192) or None,
        and the decoder's masks (B, T, K, 192)."""
        h = self.decoder(slots[..., None, :]+self.decoder_position)                     # (B, T, K, N, out)
        masks = h[..., FEATURES].softmax(2)
        features = (masks[..., None]*h[..., :FEATURES]).sum(2)
        similarity = (masks[..., None]*h[..., FEATURES+1:]).sum(2) if self.config.similarity else None
        return features, similarity, masks


def similarity_target(features, temperature):
    """VideoSAUR's target: for each patch at t, a softmax over the patches at t+1 of the cosine similarity (B, T-1,
    192, 192)."""
    f = F.normalize(features.float(), dim=-1)
    return (torch.einsum('btnd,btmd->btnm', f[:, :-1], f[:, 1:])/temperature).softmax(-1)


def losses(model, features, *, sigreg, similarity_weight, sigreg_weight, temperature, generator=None):
    slots, _ = model(features, generator=generator)
    recon, similarity, masks = model.decode(slots)
    feature = (recon.float()-features.float()).pow(2).mean()
    terms = dict(feature=feature)
    total = feature
    if similarity is not None and similarity_weight:
        target = similarity_target(features, temperature)
        sim = -(target*similarity[:, :-1].float().log_softmax(-1)).sum(-1).mean()
        terms['similarity'] = sim
        total = total+similarity_weight*sim
    if sigreg_weight:
        with torch.autocast('cuda', enabled=False):
            sig = sigreg(slots.float().transpose(0, 1).flatten(1, 2), generator)          # (T, B*K, D)
        terms['sigreg'] = sig
        total = total+sigreg_weight*sig
    return total, terms
