"""P37 (docs/dynamic/PLAN.md P37): one shared object detector for identical obstacles.

P33 and P35 gave each learned keypoint its own heatmap. With identical obstacles nothing tells one keypoint which
obstacle is its own, and the keypoints collapsed, drifted or died (four of six sat on nothing, with no gradient). Here
one map serves every object, so identical obstacles are treated alike and none has to be told apart.

- **The mask network** (`MaskNet`): P30's keypoint network (`models.keypoints.KeypointNet`'s five 3 x 3 convolutions)
  with one output channel; a sigmoid gives a mask M in [0, 1] at full resolution. Always float32.
- **Rebuilding** (`MaskTransporter`): P30's frame-pair rebuilding with the mask in place of the keypoints' Gaussians:
  F <- (1 - M_s)(1 - M_t) F_s + M_t F_t, and a decoder rebuilds frame t from F. The background comes from frame s, so
  only what moved needs the mask. Since the mask is itself the rebuilding mask, every pixel of it gets a gradient: a
  missed obstacle raises the loss where the mask could cover it, so nothing can sit dead. A cost on the mask's mean
  (`training.shared_detector`) keeps it to what moves; without it a mask of ones would copy frame t whole.
- **Detections** (`detect`, no gradient): the mask smoothed by a Gaussian of `SMOOTH` px; its local maxima over 5 x 5
  above `PEAK` are the peaks, at most `SLOTS` by height. Each is refined to the mask's centre of mass over the pixels
  within `WINDOW` px (Chebyshev) of it that are nearer it than any other peak: a local soft-argmax, split between
  neighbours so touching discs do not pull each other. Detections within `AGENT_PX` of the agent (found by its colour,
  as P33: the robot knows itself) are dropped. NaN past each frame's count.
- No labels and no colours of obstacles are used; labels enter only P33's measures (`eval.hard_bench`).
"""
import torch
from torch import nn
from torch.nn import functional as F

from go2nav.models.keypoints import KeypointConfig, KeypointNet, conv

SLOTS, SMOOTH, PEAK, WINDOW, AGENT_PX = 6, 1.5, .5, 4, 2.


class MaskNet(nn.Module):
    def __init__(self, config=KeypointConfig(keypoints=1)):
        super().__init__()
        self.config = config
        self.net = KeypointNet(KeypointConfig(keypoints=1, width=config.width, size=config.size,
                                              channels=config.channels)).net

    def forward(self, pixels):
        """Frames (N, 3, H, W) in [-1, 1] to the mask (N, 1, H, W) in [0, 1]."""
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            return torch.sigmoid(self.net(pixels.float()))


class MaskTransporter(nn.Module):
    def __init__(self, config=KeypointConfig(keypoints=1)):
        super().__init__()
        self.config = c = config
        self.mask = MaskNet(config)
        self.features = nn.Sequential(conv(c.channels, c.width), nn.ReLU(), conv(c.width, c.width), nn.ReLU(),
                                      conv(c.width, c.features))
        self.decoder = nn.Sequential(conv(c.features, 2*c.width), nn.ReLU(), conv(2*c.width, 2*c.width), nn.ReLU(),
                                     conv(2*c.width, c.channels))

    def forward(self, source, target):
        """Frames s and t (N, 3, H, W) in [-1, 1] to the rebuilt frame t (N, 3, H, W) and both masks (N, 1, H, W)."""
        both = self.mask(torch.cat((source, target)))
        ms, mt = both.chunk(2)
        fs, ft = self.features(torch.cat((source, target))).chunk(2)
        return self.decoder((1-ms)*(1-mt)*fs+mt*ft), ms, mt


def gaussian_kernel(sigma, device):
    r = int(2*sigma+.5)
    x = torch.arange(-r, r+1, dtype=torch.float32, device=device)
    g = torch.exp(-x.square()/(2*sigma**2))
    g = g/g.sum()
    return (g[:, None]*g[None, :])[None, None], r


@torch.no_grad()
def smoothed_peak(mask, smooth=SMOOTH):
    """Each frame's highest value of the smoothed mask (N,), for P37 round 2's threshold (no labels)."""
    kernel, r = gaussian_kernel(smooth, mask.device)
    return F.conv2d(F.pad(mask.float(), (r, r, r, r)), kernel).flatten(1).max(1).values


@torch.no_grad()
def detect(mask, agent=None, slots=SLOTS, smooth=SMOOTH, peak=PEAK, window=WINDOW, agent_px=AGENT_PX):
    """Masks (N, 1, H, W) and the agent's position (N, 2) px (or None) to detections (N, slots, 2) px (x, y), NaN past
    each frame's count, highest peak first; see the module's description."""
    mask = mask.float()
    n, _, h, w = mask.shape
    kernel, r = gaussian_kernel(smooth, mask.device)
    s = F.conv2d(F.pad(mask, (r, r, r, r)), kernel)[:, 0]                                          # (N, H, W)
    s = s+1e-5*torch.arange(h*w, device=mask.device).view(h, w)/(h*w)    # breaks exact ties on a saturated plateau
    top =(s == F.max_pool2d(s[:, None], 5, 1, 2)[:, 0]) & (s > peak)
    height, index = torch.where(top, s, torch.full_like(s, -1.)).flatten(1).topk(slots+1, -1)
    live = height > 0                                                     # one spare slot, for a peak on the agent
    py, px = (index//w).float()+.5, (index % w).float()+.5                                         # pixel centres
    cx = torch.arange(w, dtype=torch.float32, device=mask.device)+.5
    cy = torch.arange(h, dtype=torch.float32, device=mask.device)+.5
    dx, dy = cx[None, None, None, :]-px[..., None, None], cy[None, None, :, None]-py[..., None, None]  # (N, P, H, W)
    d2 = torch.where(live[..., None, None], dx.square()+dy.square(), torch.full_like(dx, float('inf')))
    nearest = d2.argmin(1, keepdim=True)                                                           # (N, 1, H, W)
    own = (nearest == torch.arange(slots+1, device=mask.device)[None, :, None, None]) & \
          (dx.abs() <= window) & (dy.abs() <= window)
    weight = mask*own                                                                              # (N, P, H, W)
    total = weight.sum((2, 3)).clamp(min=1e-9)
    xy = torch.stack(((weight*cx).sum((2, 3))/total, (weight*cy[:, None]).sum((2, 3))/total), -1)  # (N, P, 2)
    if agent is not None:
        live = live & ((xy-agent[:, None].float()).norm(dim=-1) >= agent_px)
    rank = torch.where(live, height, torch.full_like(height, -1.)).argsort(-1, descending=True)[:, :slots]
    xy, live = xy.gather(1, rank[..., None].expand(-1, -1, 2)), live.gather(1, rank)
    return torch.where(live[..., None], xy, torch.full_like(xy, float('nan')))
