"""Self-supervised object slots on frozen DINOv2 patch features (docs/dynamic/PLAN.md P23 item 3): VideoSAUR's recipe
(Zadaianchuk, Seitzer and Martius, NeurIPS 2023, arXiv 2306.04829) as C-JEPA (arXiv 2602.11389) trained its encoder.

- **Inputs:** a frame's DINOv2 ViT-S/14 patch tokens (384; `patches` of them, 16 x 12 = 192 on the corpus's 224 x 168
  frames, 32 x 24 on frames upsampled twice), through a LayerNorm and an MLP to `dim`, plus a learned position embedding
  per patch.
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
`masks` (frames, slots, patches) is what P23's measures read: the image columns a slot covers.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.models.lewm import SIGReg, ViTBlock
from go2nav.models.norm import LayerNorm

PATCHES, FEATURES = 192, 384
DECODE_ELEMENTS = 2**26      # the broadcast decoder runs on blocks of frames of at most this many elements a tensor:
                             # on this ROCm build, gradients through its tensors of 3e8 elements (32 clips of 4
                             # frames at 768 patches) came back wrong, at 1e8 right (P23's third run)


def load_model(path, device='cpu'):
    """Read existing s0 without exposing a new ungated slot-training entrypoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = SlotModel(SlotConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()


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
    patches: int = PATCHES             # patch tokens a frame (`data.dino_features` index's grid)
    # VideoSAUR's own choices (martius-lab/videosaur, configs/videosaur/ytvis_dinov2.yml; P23's second run); the
    # defaults keep the first model's
    decoder: str = 'mlp'               # 'mlp': DINOSAUR's broadcast decoder; 'mixer': the SlotMixer decoder
    mixer_blocks: int = 3              # the mixer's allocator: blocks of attention from the patches to the slots
    mixer_heads: int = 4
    renderer: int = 1024               # the mixer's renderer: three layers this wide
    slot_mlp: bool = True              # the residual MLP after each slot-attention round (VideoSAUR: off)
    first_iterations: int = 0          # slot-attention rounds on a clip's first frame (0: `iterations`; VideoSAUR: 3)
    input_position: bool = True        # a learned position embedding added to the patches (VideoSAUR: none)
    encoder_hidden: int = FEATURES     # the input MLP's width (VideoSAUR: twice the features)
    videosaur_init: bool = False       # first-frame slots from mean 0 and a learned scale starting at dim ** -0.5
    gru_from_normed: bool = False      # the GRU updates the normalised slots (VideoSAUR's slot attention)

    def to_dict(self):
        return asdict(self)


class SlotAttention(nn.Module):
    def __init__(self, dim, iterations, hidden, eps=1e-8, use_mlp=True, gru_from_normed=False):
        super().__init__()
        self.iterations, self.eps, self.scale, self.gru_from_normed = iterations, eps, dim**-.5, gru_from_normed
        self.norm_inputs, self.norm_slots, self.norm_mlp = LayerNorm(dim), LayerNorm(dim), LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)) if use_mlp else None

    def forward(self, inputs, slots, iterations=None):
        """inputs (B, N, D), slots (B, K, D) to slots (B, K, D) and attention (B, K, N) (summing to 1 over slots)."""
        b, k, d = slots.shape
        x = self.norm_inputs(inputs)
        keys, values = self.k(x), self.v(x)
        for _ in range(iterations or self.iterations):
            normed = self.norm_slots(slots)
            previous = normed if self.gru_from_normed else slots
            q = self.q(normed)
            attention = torch.einsum('bkd,bnd->bkn', q, keys).mul(self.scale).softmax(1)
            weights = attention+self.eps
            weights = weights/weights.sum(-1, keepdim=True)
            updates = torch.einsum('bkn,bnd->bkd', weights, values)
            slots = self.gru(updates.reshape(b*k, d), previous.reshape(b*k, d)).reshape(b, k, d)
            if self.mlp is not None:
                slots = slots+self.mlp(self.norm_mlp(slots))
        return slots, attention


class Allocator(nn.Module):
    """VideoSAUR's allocator for the SlotMixer decoder (`networks.TransformerEncoder` with a memory): pre-norm blocks
    in which each patch's query attends to the slots (not to the other patches), then a feed-forward layer."""

    def __init__(self, dim, blocks, heads):
        super().__init__()
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            norm1=LayerNorm(dim), attention=nn.MultiheadAttention(dim, heads, batch_first=True), norm2=LayerNorm(dim),
            ff=nn.Sequential(nn.Linear(dim, 4*dim), nn.ReLU(), nn.Linear(4*dim, dim)))) for _ in range(blocks))

    def forward(self, queries, memory):
        x = queries
        for block in self.blocks:
            with sdpa_kernel(SDPBackend.MATH):
                x = x+block['attention'](block['norm1'](x), memory, memory, need_weights=False)[0]
            x = x+block['ff'](block['norm2'](x))
        return x


class SlotMixer(nn.Module):
    """The SlotMixer decoder (Sajjadi et al. 2022, as VideoSAUR's `decoders.SlotMixerDecoder` with layer norms and the
    position embedding added): per patch, the allocator's query picks a mixture of the slots (softmax over slots: the
    masks), and the renderer turns the mixed slot and the patch's embedding into the features and similarity logits."""

    def __init__(self, dim, out, blocks, heads, renderer, patches=PATCHES):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, patches, dim)*dim**-.5)
        self.allocator = Allocator(dim, blocks, heads)
        self.norm_memory, self.norm_q, self.norm_k = LayerNorm(dim), LayerNorm(dim), LayerNorm(dim)
        self.to_q, self.to_k = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.scale = dim**-.5
        self.renderer = nn.Sequential(nn.Linear(dim, renderer), nn.ReLU(), nn.Linear(renderer, renderer), nn.ReLU(),
                                      nn.Linear(renderer, renderer), nn.ReLU())
        self.out = nn.Linear(renderer, out)
        for layer in (self.to_q, self.to_k, self.out):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, slots):
        """Slots (M, K, D) to outputs (M, N, out) and masks (M, K, N), N patches."""
        position = self.position.expand(len(slots), -1, -1)
        queries = self.allocator(position, self.norm_memory(slots))
        q, k = self.to_q(self.norm_q(queries)), self.to_k(self.norm_k(slots))
        attention = torch.einsum('mpd,mkd->mpk', q, k).mul(self.scale).softmax(-1)
        mixed = torch.einsum('mpk,mkd->mpd', attention, slots)+position
        return self.out(self.renderer(mixed)), attention.transpose(1, 2)


class SlotModel(nn.Module):
    def __init__(self, config=SlotConfig()):
        super().__init__()
        self.config = config
        d, patches = config.dim, config.patches
        self.encoder = nn.Sequential(LayerNorm(FEATURES), nn.Linear(FEATURES, config.encoder_hidden), nn.ReLU(),
                                     nn.Linear(config.encoder_hidden, d))
        if config.input_position:
            self.position = nn.Parameter(torch.randn(1, patches, d)*.02)
        if config.videosaur_init:
            self.mu = nn.Parameter(torch.zeros(1, 1, d))
            self.log_sigma = nn.Parameter(torch.full((1, 1, d), math.log(d**-.5)))
        else:
            self.mu = nn.Parameter(torch.randn(1, 1, d)*.02)
            self.log_sigma = nn.Parameter(torch.zeros(1, 1, d))
        self.attention = SlotAttention(d, config.iterations, config.mlp, use_mlp=config.slot_mlp,
                                       gru_from_normed=config.gru_from_normed)
        self.carry = ViTBlock(d, config.carry_heads, config.mlp)
        if config.decoder == 'mixer':
            self.mixer = SlotMixer(d, FEATURES+(patches if config.similarity else 0), config.mixer_blocks,
                                   config.mixer_heads, config.renderer, patches)
        elif config.decoder == 'mlp':
            self.decoder_position = nn.Parameter(torch.randn(1, 1, patches, d)*.02)
            layers, width = [], d
            for _ in range(config.decoder_layers):
                layers += [nn.Linear(width, config.decoder_hidden), nn.ReLU()]
                width = config.decoder_hidden
            out = FEATURES+1+(patches if config.similarity else 0)
            self.decoder = nn.Sequential(*layers, nn.Linear(width, out))
        else:
            raise ValueError("decoder is 'mlp' or 'mixer'")

    def initial(self, batch, generator=None):
        noise = torch.randn(batch, self.config.slots, self.config.dim, device=self.mu.device, generator=generator)
        return self.mu+self.log_sigma.exp()*noise

    def forward(self, features, slots=None, generator=None):
        """Features (B, T, N, 384) to slots (B, T, K, D) and attention masks (B, T, K, N); `slots` (B, K, D)
        continues from earlier frames."""
        b, t = features.shape[:2]
        x = self.encoder(features)
        if self.config.input_position:
            x = x+self.position
        out, masks = [], []
        for i in range(t):
            first = slots is None
            if first:
                start = self.initial(b, generator)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                    start = self.carry(slots)
            slots, attention = self.attention(x[:, i], start, self.config.first_iterations if first else None)
            out.append(slots), masks.append(attention)
        return torch.stack(out, 1), torch.stack(masks, 1)

    def decode(self, slots):
        """Slots (B, T, K, D) to reconstructed features (B, T, N, 384), similarity logits (B, T, N, N) or None,
        and the decoder's masks (B, T, K, N)."""
        if self.config.decoder == 'mixer':
            b, t = slots.shape[:2]
            h, masks = self.mixer(slots.flatten(0, 1))
            h, masks = h.unflatten(0, (b, t)), masks.unflatten(0, (b, t))
            return h[..., :FEATURES], (h[..., FEATURES:] if self.config.similarity else None), masks
        b, t = slots.shape[:2]
        flat = slots.flatten(0, 1)                                                     # (M, K, D)
        c = self.config
        per = max(1, DECODE_ELEMENTS//(c.slots*c.patches*max(c.decoder_hidden, FEATURES, c.patches)))
        parts = [self.broadcast(flat[i:i+per]) for i in range(0, len(flat), per)]
        return tuple(None if p[0] is None else torch.cat(p, 0).unflatten(0, (b, t)) for p in zip(*parts))

    def broadcast(self, slots):
        """The broadcast decoder on frames' slots (M, K, D): features (M, N, 384), similarity logits (M, N, N) or
        None, masks (M, K, N)."""
        hidden = self.decoder[:-1](slots[..., None, :]+self.decoder_position[0])          # (M, K, N, hidden)
        # the last layer in its three parts (slices of its weight: the same layer): as one layer of 1 + 384 + N
        # outputs its weight gradient ran on a slow kernel on this ROCm build (0.9 s a call at 768 patches)
        last = self.decoder[-1]
        masks = F.linear(hidden, last.weight[FEATURES:FEATURES+1], last.bias[FEATURES:FEATURES+1])[..., 0].softmax(1)
        features = torch.einsum('mkn,mknd->mnd', masks, F.linear(hidden, last.weight[:FEATURES], last.bias[:FEATURES]))
        similarity = torch.einsum('mkn,mknj->mnj', masks, F.linear(hidden, last.weight[FEATURES+1:],
                                                                   last.bias[FEATURES+1:])) \
            if self.config.similarity else None
        return features, similarity, masks


def similarity_target(features, temperature, threshold=None):
    """VideoSAUR's target: for each patch at t, a softmax over the patches at t+1 of the cosine similarity (B, T-1,
    N, N); with `threshold`, similarities under it count for nothing (VideoSAUR's DINOv2 setting: 0)."""
    f = F.normalize(features.float(), dim=-1)
    similarity = torch.einsum('btnd,btmd->btnm', f[:, :-1], f[:, 1:])
    if threshold is not None:
        similarity = similarity.masked_fill(similarity < threshold, -torch.inf)
        empty = torch.isinf(similarity).all(-1, keepdim=True)              # a row with nothing over it: uniform
        similarity = similarity.masked_fill(empty, 0.)
    return (similarity/temperature).softmax(-1)


def losses(model, features, *, sigreg, similarity_weight, sigreg_weight, temperature, generator=None,
           threshold=None):
    slots, _ = model(features, generator=generator)
    recon, similarity, masks = model.decode(slots)
    feature = (recon.float()-features.float()).pow(2).mean()
    terms = dict(feature=feature)
    total = feature
    if similarity is not None and similarity_weight:
        target = similarity_target(features, temperature, threshold)
        sim = -(target*similarity[:, :-1].float().log_softmax(-1)).sum(-1).mean()
        terms['similarity'] = sim
        total = total+similarity_weight*sim
    if sigreg_weight:
        with torch.autocast('cuda', enabled=False):
            sig = sigreg(slots.float().transpose(0, 1).flatten(1, 2), generator)          # (T, B*K, D)
        terms['sigreg'] = sig
        total = total+sigreg_weight*sig
    return total, terms
