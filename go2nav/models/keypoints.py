"""P30 (docs/dynamic/PLAN.md P30 item 1): learned keypoints after Transporter (Kulkarni, Gupta, Ionescu, Borgeaud,
Reynolds, Zisserman and Mnih, "Unsupervised learning of object keypoints for perception and control", NeurIPS 2019),
written from the paper for 64 x 64 frames.

- **The keypoint network** (`KeypointNet`) maps a frame to K heatmaps at full resolution (five 3 x 3 convolutions, a
  receptive field of 11 px, against a disc 7 px across; dilated convolutions were 30-50 times slower in MIOpen on the
  R9700); a spatial softmax over each gives a position, the heatmap's expectation, in pixels
  (pixel k's centre at k + 0.5, as `sim.balls`). It always runs in float32: under bf16 autocast a position near 64 px
  would carry a quarter of a pixel of rounding.
- **Transport** (`Transporter`): to rebuild frame t from frame s, the feature map of s is taken, and for each keypoint
  in turn the features under s's and t's Gaussian masks round it are removed and t's features under t's mask pasted:
  F <- (1 - H_s^k)(1 - H_t^k) F + H_t^k F_t. A decoder maps F to frame t. The loss is the pixel error, so the
  keypoints must sit where things moved; the background comes from frame s.
- **Coarse to fine:** keypoints whose heatmaps start flat sit on nothing and get no gradient from narrow masks (the
  second smoke run: three of four dead). Training starts the masks wide and narrows them (`training.keypoints`).
- **Separation** (`separation`): with the keypoint network at full resolution, nothing in the loss stops several
  keypoints settling on one object, and the first smoke run put all four on the same disc. A penalty on pairs of
  keypoints in a frame, exp(-d^2 / 2s^2) with s = 4 px, keeps them apart. It uses no labels.
- No labels and no colours are used. The assignment of keypoints to objects for scoring (`assign`) is the only
  place labels enter, and in `models.object_tokens` it is a fixed reordering of the tokens.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class KeypointConfig:
    keypoints: int = 4
    width: int = 32
    features: int = 32
    sigma: float = 2.            # px: the Gaussian masks' deviation
    size: int = 64


def conv(a, b, kernel=3, dilation=1):
    return nn.Conv2d(a, b, kernel, padding=dilation*(kernel//2), dilation=dilation)


class KeypointNet(nn.Module):
    def __init__(self, config=KeypointConfig()):
        super().__init__()
        w, self.config = config.width, config
        self.net = nn.Sequential(conv(3, w), nn.ReLU(), conv(w, w), nn.ReLU(), conv(w, 2*w), nn.ReLU(),
                                 conv(2*w, 2*w), nn.ReLU(), conv(2*w, 2*w), nn.ReLU(), conv(2*w, config.keypoints, 1))
        self.register_buffer('centres', torch.arange(config.size, dtype=torch.float32)+.5, persistent=False)

    def forward(self, pixels):
        """Frames (N, 3, H, W) in [-1, 1] to keypoints (N, K, 2), (x, y) in pixels."""
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            logits = self.net(pixels.float())
            prob = logits.flatten(2).softmax(-1).unflatten(2, logits.shape[2:])
            x, y = (prob.sum(2)*self.centres).sum(-1), (prob.sum(3)*self.centres).sum(-1)
        return torch.stack((x, y), -1)


class Transporter(nn.Module):
    def __init__(self, config=KeypointConfig()):
        super().__init__()
        self.config = c = config
        self.keypoints = KeypointNet(config)
        self.features = nn.Sequential(conv(3, c.width), nn.ReLU(), conv(c.width, c.width), nn.ReLU(),
                                      conv(c.width, c.features))
        self.decoder = nn.Sequential(conv(c.features, 2*c.width), nn.ReLU(), conv(2*c.width, 2*c.width), nn.ReLU(),
                                     conv(2*c.width, 3))
        self.register_buffer('centres', torch.arange(c.size, dtype=torch.float32)+.5, persistent=False)

    def masks(self, xy, sigma=None):
        """Keypoints (N, K, 2) px to Gaussian masks (N, K, H, W) of deviation `sigma` (the config's by default)."""
        sigma = self.config.sigma if sigma is None else sigma
        dx = self.centres[None, None, None, :]-xy[..., 0, None, None]
        dy = self.centres[None, None, :, None]-xy[..., 1, None, None]
        return torch.exp(-(dx.square()+dy.square())/(2*sigma**2))

    def forward(self, source, target, sigma=None):
        """Frames s and t (N, 3, H, W) in [-1, 1] to the rebuilt frame t (N, 3, H, W), and t's keypoints. `sigma`: the
        masks' deviation (training anneals it from wide to the config's)."""
        both = self.keypoints(torch.cat((source, target)))
        hs, ht = self.masks(both[:len(source)], sigma), self.masks(both[len(source):], sigma)
        fs, ft = self.features(torch.cat((source, target))).chunk(2)
        out = fs
        for k in range(self.config.keypoints):
            out = (1-hs[:, k:k+1])*(1-ht[:, k:k+1])*out+ht[:, k:k+1]*ft
        return self.decoder(out), both[len(source):]


def separation(xy, scale=4.):
    """The mean over pairs of keypoints in a frame of exp(-d^2 / 2 scale^2): keypoints (N, K, 2) px to a scalar."""
    d2 = (xy[:, :, None]-xy[:, None]).square().sum(-1)
    k = xy.shape[1]
    pairs = ~torch.eye(k, dtype=torch.bool, device=xy.device)
    return torch.exp(-d2/(2*scale**2))[:, pairs].mean()


# -- scoring (labels for scoring only) --------------------------------------------------------------------------------

def assign(keypoints, truth):
    """Keypoints (N, K, 2) and true positions (N, J, 2), J <= K, to the keypoint of each object (J,) by the least
    mean distance (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(keypoints[:, :, None]-truth[:, None], axis=-1).mean(0)          # (K, J)
    rows, cols = linear_sum_assignment(cost)
    order = np.zeros(truth.shape[1], np.int64)
    order[cols] = rows
    return order


def measures(keypoints, truth, order, offsets):
    """P30's stage-1 measures on frames: keypoints (N, K, 2), truth (N, J, 2), the keypoint of each object (J,) and
    each object's constant offset (J, 2) (truth less keypoint, fitted on training frames). Per object: binding (share
    within 1 px), precision (mean error) and swaps (share of frames where the keypoint is nearer another object),
    after removing the offset and raw."""
    out = {}
    for name, shift in (('corrected', offsets), ('raw', np.zeros_like(offsets))):
        placed = keypoints[:, order]+shift                                                     # (N, J, 2)
        distance = np.linalg.norm(placed[:, :, None]-truth[:, None], axis=-1)                   # (N, J, J)
        own = np.diagonal(distance, axis1=1, axis2=2)
        other = np.where(np.eye(truth.shape[1], dtype=bool), np.inf, distance).min(-1)
        out[name] = dict(binding=(own < 1.).mean(0).tolist(), precision=own.mean(0).tolist(),
                         median=np.median(own, 0).tolist(), swaps=(other < own).mean(0).tolist())
    return out
