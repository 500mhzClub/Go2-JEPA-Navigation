"""P22b's value of the state (docs/dynamic/PLAN.md): the cost to go on a leg of the round trip, from a window round the
robot, as a correction to the frozen rule's distance term.

**The window** (`draw`): `WINDOW_M` square round the robot at `PIXEL_M`, aligned with it (rows ahead, columns to its
left), in `CHANNELS` layers: the walls' distance (to `WALL_CLIP_M`); the walking distance to the target through the
walls, less the robot's own, over `DISTANCE_M` (closed ground as far); each walker as a disc of its radius carrying its
velocity in the robot's frame (over `SPEED_M_S`) and its memory (`perception.ring_walkers.Memory`). The robot is the
window's centre and heading, so its outline would be a constant layer and is left out. The layout's two grids (walls,
the target's distance) are sampled bilinearly from `planning.fields.Fields`'s nodes.

**The network** (`ValueNet`): four strided convolutions, then the window's features with two numbers (the frozen rule's
distance term over `BASE_S`, the robot's walking distance over `DISTANCE_M`) through an MLP to one number, the
correction in units of `BASE_S`; the last layer starts at zero, so an untrained value is the frozen rule.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

WINDOW_M, PIXEL_M = 8., .125
SIZE = int(round(WINDOW_M/PIXEL_M))
WALL_CLIP_M, DISTANCE_M, FAR_M, SPEED_M_S, BASE_S = 1., 10., 50., .35, 100.
RADIUS_M = .175                       # a walker's (`sim.obstacles.RADIUS_M`)
MEMORY = 4                            # `perception.ring_walkers.Memory.FEATURES`
CHANNELS = 3+2+MEMORY                 # walls, distance, presence; velocity; memory
PIXELS = (np.arange(SIZE)+.5)*PIXEL_M-WINDOW_M/2


@dataclass
class ValueConfig:
    width: int = 32
    hidden: int = 256


class ValueNet(nn.Module):
    def __init__(self, config=ValueConfig()):
        super().__init__()
        self.config, w = config, config.width
        self.convs = nn.Sequential(nn.Conv2d(CHANNELS, w, 5, 2, 2), nn.GELU(), nn.Conv2d(w, 2*w, 3, 2, 1), nn.GELU(),
                                   nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.GELU(), nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.GELU())
        cells = (SIZE//16)**2*4*w
        self.head = nn.Sequential(nn.Linear(cells+2, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, window, base_s, distance_m):
        """The correction (seconds, (B,)) to the distance term `base_s` (B,) of windows (B, CHANNELS, SIZE, SIZE)."""
        features = self.convs(window).flatten(1)
        scalars = torch.stack((base_s/BASE_S, distance_m/DISTANCE_M), 1)
        return self.head(torch.cat((features, scalars), 1))[:, 0]*BASE_S


def wall_grid(fields):
    """A layout's walls' distance (nx, ny) float32 on `planning.fields.Fields`'s nodes (origin `fields.low`)."""
    return np.minimum(fields.geometry.wall_distance(fields.nodes), WALL_CLIP_M).astype(np.float32)


def distance_grid(fields, target):
    """The walking distance to `target` (nx, ny) float32 on the same nodes, closed ground at `FAR_M`."""
    f = fields.static(np.asarray(target, float))
    return np.where(np.isfinite(f), np.minimum(f, FAR_M), FAR_M).astype(np.float32)


def draw(walls, distance, low, xy, yaw, own_m, walkers_xy, walkers_v, memory, present, resolution=.05):
    """Windows (B, CHANNELS, SIZE, SIZE), torch on the grids' device.
    walls (B or 1, nx, ny), distance (B or 1, nx, ny): each sample's grids (`wall_grid`, `distance_grid` for its
    target); low (2,) or (B, 2): their origin; xy (B, 2), yaw (B,): the robot; own_m (B,): its walking distance to the
    target; walkers_xy, walkers_v (B, S, 2), memory (B, S, MEMORY), present (B, S) bool."""
    device = walls.device
    count = xy.shape[0]
    u = torch.as_tensor(PIXELS, dtype=torch.float32, device=device)
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    ahead, left = u[None, :, None], u[None, None, :]
    world_x = xy[:, 0, None, None]+c*ahead-s*left                                    # (B, SIZE, SIZE)
    world_y = xy[:, 1, None, None]+s*ahead+c*left
    nx, ny = walls.shape[-2:]
    low = low.reshape(-1, 2)
    grid = torch.stack((2*(world_y-low[:, 1, None, None])/resolution/(ny-1)-1,
                        2*(world_x-low[:, 0, None, None])/resolution/(nx-1)-1), -1)
    layers = torch.stack((walls.expand(count, -1, -1), distance.expand(count, -1, -1)), 1)
    sampled = F.grid_sample(layers, grid, mode='bilinear', padding_mode='border', align_corners=True)
    wall = sampled[:, 0]/WALL_CLIP_M
    far = torch.clamp((sampled[:, 1]-own_m[:, None, None])/DISTANCE_M, -1., 1.)
    d = walkers_xy-xy[:, None]                                                       # (B, S, 2) in the robot's frame
    rel = torch.stack((c[:, :, 0]*d[..., 0]+s[:, :, 0]*d[..., 1], -s[:, :, 0]*d[..., 0]+c[:, :, 0]*d[..., 1]), -1)
    vel = torch.stack((c[:, :, 0]*walkers_v[..., 0]+s[:, :, 0]*walkers_v[..., 1],
                       -s[:, :, 0]*walkers_v[..., 0]+c[:, :, 0]*walkers_v[..., 1]), -1)/SPEED_M_S
    rel = torch.where(present[..., None], rel, torch.full_like(rel, 1e3))
    gap = torch.sqrt((ahead[:, None]-rel[..., 0, None, None])**2+(left[:, None]-rel[..., 1, None, None])**2)
    disc = (gap < RADIUS_M+PIXEL_M/2).float()                                       # (B, S, SIZE, SIZE)
    presence = disc.amax(1)
    values = torch.where(present[..., None], torch.cat((vel, memory), -1), 0.)          # an empty slot's may be NaN
    carried = torch.einsum('bsij,bsk->bkij', disc, values)
    return torch.cat((wall[:, None], far[:, None], presence[:, None], carried), 1)


def save(path, model, extra=None):
    torch.save(dict(config=asdict(model.config), model=model.state_dict(), **(extra or {})), path)


def load(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    model = ValueNet(ValueConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()
