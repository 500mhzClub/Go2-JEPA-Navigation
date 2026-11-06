"""The walker memory (docs/dynamic/PLAN.md, P9): a body-frame map of where walkers are, seen or not, carried across
frames by the world model's states and the robot's own motion.

Every step (5 Hz, every second frame) the memory, a `BEV_CELLS` x `BEV_CELLS` grid of `channels` features at
`memory_labels.CELL_M` round the body origin (±3.2 m), is moved into the current body frame by the pose change since
the last step (`warp`; the visual odometry's pose at run time), then updated by a convolutional GRU from what this
frame's world-model state shows (`lift`: each cell attends over the state's [CLS] and grid tokens). `logits` reads
each cell's walker occupancy from the memory, and `occupancy` the labelled grid of `memory_labels` (its central
`memory_labels.CELLS` cells). The geometry (where a remembered cell moves to when the robot moves) is given; what
is there, how long it stays and where a walker goes are learned.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from go2nav.data import memory_labels as ml

BEV_CELLS = 2*ml.CELLS
STEP_FRAMES = 2
INITIAL_LOGIT = -4.6               # about 1% occupied, the cells' base rate


@dataclass(frozen=True)
class MemoryConfig:
    tokens: int = 13                # [CLS] and the 3 x 4 grid of go2_v2_walkers' state
    latent: int = 192
    channels: int = 32
    query: int = 64
    heads: int = 4

    def to_dict(self):
        return asdict(self)


class ConvGRU(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.gates = nn.Conv2d(inputs+hidden, 2*hidden, 3, padding=1)
        self.candidate = nn.Conv2d(inputs+hidden, hidden, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat((x, h), 1))).chunk(2, 1)
        n = torch.tanh(self.candidate(torch.cat((x, r*h), 1)))
        return (1-z)*h+z*n


class WalkerMemory(nn.Module):
    def __init__(self, config=MemoryConfig()):
        super().__init__()
        c = self.config = config
        self.queries = nn.Parameter(torch.randn(BEV_CELLS*BEV_CELLS, c.query)*.02)
        self.token_position = nn.Parameter(torch.randn(c.tokens, c.latent)*.02)
        self.keys = nn.Linear(c.latent, c.query)
        self.values = nn.Linear(c.latent, c.query)
        self.attention = nn.MultiheadAttention(c.query, c.heads, batch_first=True)
        self.observed = nn.Linear(c.query, c.channels)
        self.cell = ConvGRU(c.channels, c.channels)
        self.refine = nn.Sequential(nn.Conv2d(c.channels, c.channels, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(c.channels, c.channels, 3, padding=1))
        self.head = nn.Conv2d(c.channels, 1, 1)
        nn.init.constant_(self.head.bias, INITIAL_LOGIT)

    def initial(self, batch, device):
        return torch.zeros(batch, self.config.channels, BEV_CELLS, BEV_CELLS, device=device)

    def lift(self, state):
        """World-model states (B, tokens, latent) to observed features on the grid (B, channels, BEV, BEV)."""
        x = state+self.token_position
        q = self.queries.expand(len(x), -1, -1)
        out, _ = self.attention(q, self.keys(x), self.values(x), need_weights=False)
        return self.observed(out).transpose(1, 2).unflatten(2, (BEV_CELLS, BEV_CELLS))

    @staticmethod
    def warp(memory, motion):
        """The memory (B, C, BEV, BEV) in the body frame of the next step, given that step's pose in this step's body
        frame (B, 3: x, y, yaw). Row i of the grid is x (forward), column j is y (left), as `memory_labels`."""
        half = BEV_CELLS*ml.CELL_M/2
        c, s = torch.cos(motion[:, 2]), torch.sin(motion[:, 2])
        # a body point p of the new frame was at R p + t in the old one: x_old = c x - s y + t_x, y_old = s x + c y + t_y.
        # grid_sample's first coordinate u indexes columns (y), the second v rows (x), both y/half and x/half
        theta = torch.stack([torch.stack([c, s, motion[:, 1]/half], -1),
                             torch.stack([-s, c, motion[:, 0]/half], -1)], 1).to(memory.dtype)
        grid = F.affine_grid(theta, memory.shape, align_corners=False)
        return F.grid_sample(memory, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def step(self, memory, state, motion):
        """One step: warp by `motion`, then update from `state`."""
        memory = self.warp(memory, motion)
        return self.cell(self.lift(state), memory)

    def logits(self, memory):
        """Walker-occupancy logits over the whole memory grid (B, BEV, BEV)."""
        return self.head(memory+self.refine(memory))[:, 0]

    def occupancy(self, memory):
        """Logits on `memory_labels`' grid (B, CELLS, CELLS), its central cells."""
        lo = (BEV_CELLS-ml.CELLS)//2
        return self.logits(memory)[:, lo:lo+ml.CELLS, lo:lo+ml.CELLS]


def save_memory(path, model, *, extra=None):
    torch.save(dict(config=model.config.to_dict(), model_state_dict=model.state_dict(), **(extra or {})), path)


def load_memory(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = WalkerMemory(MemoryConfig(**state['config']))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval()
