"""C1's motion model: body-frame motion of a command tape from commands alone (no images).

Forecast = nominal integration of the commands + a learned correction. The correction is a ridge regression per horizon
h = 100-800 ms (penalty 1, on standardised features) of the residual true motion minus the nominal integration, on 447
features:
- the 8 x 3 prospective commands known up to h (later steps zero);
- the nominal integration at h (3);
- the applied-command history (420): for each of the four frames t-300 ms .. t, the last 15 applied commands
  (scaled by 0.3, 1, 0.5), their 15 x 3 valid flags and their 15 ages (seconds; 1.5 if missing).

Commands are [vx, vy, yaw rate] at 100-ms steps; outputs are [x, y, yaw] (metres, radians) at each 100-ms horizon.
"""
from pathlib import Path

import numpy as np

COMMAND_SCALE = np.array([.3, 1., .5])
HORIZONS = 8
FEATURES = 447


def nominal(commands):
    """Integrate an (8, 3) command tape over 100-ms steps (midpoint heading); returns (8, 3) [x, y, yaw]."""
    values, state = np.zeros((8, 3)), np.zeros(3)
    for h, (vx, vy, w) in enumerate(commands):
        angle = state[2]+.05*w
        state[:2] += .1*np.array([vx*np.cos(angle)-vy*np.sin(angle), vx*np.sin(angle)+vy*np.cos(angle)])
        state[2] += .1*w
        values[h] = state
    return values


def history_features(values, valid, measured_ns, frame_ns):
    """The 420 history features from four frames' applied-command records.

    values (4, 15, 3), valid (4, 15, 3), measured_ns (4, 15) and frame_ns (4,) for the frames t-300, -200, -100, 0 ms.
    """
    rows = []
    for v, ok, measured, now in zip(values, valid, measured_ns, frame_ns, strict=True):
        if np.any(np.asarray(measured) > now):
            raise ValueError('future command observations are not allowed')
        age = np.where(np.asarray(measured) >= 0, (now-np.asarray(measured))/1e9, 1.5)[:, None]
        rows.append(np.concatenate((np.asarray(v, np.float64)/COMMAND_SCALE, np.asarray(ok).astype(float), age), axis=1).ravel())
    out = np.concatenate(rows)
    if out.shape != (420,) or not np.isfinite(out).all():
        raise ValueError('finite 420-value command history required')
    return out


def features(commands, history):
    """Per-horizon features (8, 447) and the nominal integration (8, 3) of one (8, 3) command tape."""
    commands = np.asarray(commands, float)
    base = nominal(commands)
    x = np.zeros((HORIZONS, FEATURES))
    for h in range(HORIZONS):
        known = np.zeros((HORIZONS, 3))
        known[:h+1] = commands[:h+1]
        x[h] = np.concatenate((known.ravel(), base[h], history))
    return x, base


class CommandHistoryModel:
    """Per-horizon ridge correction: arrays mean (8, 447), scale (8, 447), bias (8, 3), coefficient (8, 447, 3)."""

    def __init__(self, mean, scale, bias, coefficient):
        self.mean, self.scale, self.bias, self.coefficient = (np.asarray(a) for a in (mean, scale, bias, coefficient))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')))

    def save(self, path):
        np.savez_compressed(Path(path), mean=self.mean, scale=self.scale, bias=self.bias, coefficient=self.coefficient)

    def predict(self, commands, history):
        """Forecasts (K, 8, 3) for K command tapes (K, 8, 3) sharing one 420-value history."""
        commands = np.asarray(commands, float)
        result = np.asarray([nominal(c) for c in commands])
        for h in range(HORIZONS):
            known = np.zeros(commands.shape)
            known[:, :h+1] = commands[:, :h+1]
            x = np.concatenate((known.reshape(len(commands), 24), result[:, h], np.broadcast_to(history, (len(commands), 420))), axis=1)
            result[:, h] += ((x-self.mean[h])/self.scale[h])@self.coefficient[h]+self.bias[h]
        return result

    @classmethod
    def fit(cls, x, residual, valid, weights, penalty=1.):
        """Weighted ridge per horizon. x (N, 8, 447); residual = true - nominal (N, 8, 3); valid (N, 8); weights (N,)."""
        values = dict(mean=[], scale=[], bias=[], coefficient=[])
        for h in range(HORIZONS):
            ok = valid[:, h]
            xh, y, w = x[ok, h], residual[ok, h], weights[ok]
            mean = np.average(xh, axis=0, weights=w)
            scale = np.sqrt(np.average((xh-mean)**2, axis=0, weights=w))
            scale[scale < 1e-8] = 1.
            bias = np.average(y, axis=0, weights=w)
            z = (xh-mean)/scale
            coefficient = np.linalg.solve(z.T@(z*w[:, None])+penalty*np.eye(xh.shape[1]), z.T@((y-bias)*w[:, None]))
            for k, v in dict(mean=mean, scale=scale, bias=bias, coefficient=coefficient).items():
                values[k].append(v)
        return cls(*(np.asarray(values[k]) for k in ('mean', 'scale', 'bias', 'coefficient')))
