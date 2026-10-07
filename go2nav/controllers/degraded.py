"""Deliberately degraded forecasts, for the forecast-sensitivity experiment (a controller's own forecast, altered).

`spec` is one of:
- `scale:S`: every candidate's predicted displacement and heading change times S;
- `fwdscale:S`: the translating candidates' (forward and the arcs) displacement times S, heading unchanged;
- `turnscale:S`: the in-place turns' displacement and heading change times S;
- `noise:E`: per plan and candidate, a 2-D Gaussian displacement error whose median magnitude at 700 ms is E mm and a
  heading error whose median magnitude at 700 ms is E/10 degrees, both growing linearly with the horizon; seeded by
  the level and the frame, so missions reproduce.
The degraded forecast is float64 (as in the original).
"""
import math

import numpy as np

from go2nav.planning.actions import ACTIONS

NOISE_PROFILE = np.arange(1, 9)/7.  # error grows linearly with the horizon; 1.0 at the scored 700 ms
TRANSLATING = [ACTIONS.index(a) for a in ('forward', 'left_arc', 'right_arc')]
TURNING = [ACTIONS.index(a) for a in ('left_turn', 'right_turn')]


class DegradedForecaster:
    def __init__(self, base, spec):
        kind, value = spec.split(':')
        if kind not in ('scale', 'noise', 'fwdscale', 'turnscale'):
            raise ValueError('degradation must be scale:S, noise:E_mm, fwdscale:S or turnscale:S')
        self.base, self.kind, self.value = base, kind, float(value)

    def forecast(self, context, prefix, *, pulse):
        p = np.array(self.base.forecast(context, prefix, pulse=pulse), float, copy=True)
        yaw = np.arctan2(p[..., 2], p[..., 3])
        value = self.value
        if self.kind == 'scale':
            p[..., :2] *= value
            yaw = yaw*value
        elif self.kind == 'fwdscale':
            p[TRANSLATING, :, :2] *= value
        elif self.kind == 'turnscale':
            p[TURNING, :, :2] *= value
            yaw[TURNING] = yaw[TURNING]*value
        else:
            rng = np.random.default_rng([20261002, int(round(value*1000)), int(context.frame)])
            sigma_xy = value/1000/math.sqrt(2*math.log(2))          # median |N(0, s^2 I2)| = s sqrt(2 ln 2)
            sigma_yaw = math.radians(value/10)/0.6744897501960817   # median |N(0, s^2)| = 0.6745 s
            p[..., :2] += rng.normal(0, sigma_xy, size=(p.shape[0], 1, 2))*NOISE_PROFILE[None, :, None]
            yaw = yaw+rng.normal(0, sigma_yaw, size=(p.shape[0], 1))*NOISE_PROFILE[None, :]
        p[..., 2], p[..., 3] = np.sin(yaw), np.cos(yaw)
        return p
