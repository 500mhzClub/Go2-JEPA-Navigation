"""C1A: the command-history forecaster with its travel adapted online, per movement type.

C1 predicts motion from commands alone, so on a slippery floor it over-predicts travel. C1A rescales C1's forecasts
by how far the robot actually moved, measured during the mission. At each plan (time t), against the plan made at
t0 = t - h*100 ms (h from 8 down to 4, the longest available):
- C1 re-forecasts [t0, t) from the commands planned over that window (each plan's committed prefix, then its command
  for its duration, then zero) and the command history it saw at t0;
- the registered pose (visual odometry corrected against the floor) gives the XY travelled (in the body frame at
  t0) and the heading change.
A window whose planned commands are all hold gives no sample. A translation sample needs a forecast of at least 2 cm,
a rotation sample at least 2 degrees; the ratio is tracked / forecast. Samples are filed under the window's movement
type (`planning.actions.movement`); hold and rest_start windows are never adapted.

Each type's translation and rotation ratios are the median of its samples from the last 3 s, clipped to [0.3, 1.5],
once it has two; otherwise 1.0. Every candidate's XY is scaled by its type's translation ratio and its yaw by its
rotation ratio; a candidate's type is its committed prefix then its command held, against the planned commands of
the 1.5 s before t. Inputs: the registered pose and the controller's own plans only.
"""
from collections import defaultdict, deque

import numpy as np

from go2nav.controllers.forecasters import command_history
from go2nav.planning.actions import PERIOD_NS, PREFIX_TICKS, command_sequences, movement

ADAPTED = ('cruise', 'arc_steady', 'turn', 'switch')
HORIZONS = range(8, 3, -1)
WINDOW_NS = 3_000_000_000
MIN_TRANSLATION_M, MIN_ROTATION_RAD = .02, float(np.radians(2.))
CLIP, MIN_SAMPLES = (.3, 1.5), 2
PAST_TICKS = 15


def planned_tape(commitments, start_ns, steps):
    """(steps, 3) commands planned for start_ns + k*100 ms by the latest plan at each tick, or None before any plan."""
    out = []
    for k in range(steps):
        tick = start_ns+k*PERIOD_NS
        prior = [c for c in commitments if c.plan.observed_ns <= tick]
        if not prior:
            return None
        latest = prior[-1]
        j = (tick-latest.plan.observed_ns)//PERIOD_NS
        duration = (latest.plan.expires_ns-latest.plan.dispatch_ns)//PERIOD_NS
        if j < len(latest.prefix):
            out.append(latest.prefix[j])
        elif j < len(latest.prefix)+duration:
            out.append(latest.plan.command)
        else:
            out.append((0., 0., 0.))
    return np.asarray(out, float)


def planned_past(commitments, end_ns):
    """(15, 3) commands planned for the 1.5 s before end_ns (zero before the first plan)."""
    out = np.zeros((PAST_TICKS, 3))
    start = end_ns-PAST_TICKS*PERIOD_NS
    for k in range(PAST_TICKS):
        tick = planned_tape(commitments, start+k*PERIOD_NS, 1)
        if tick is not None:
            out[k] = tick[0]
    return out


def window_movement(ticks, past):
    tape = np.zeros((8, 3))
    tape[:len(ticks)] = ticks
    tape[len(ticks):] = ticks[-1]
    return movement(tape, past)


def candidate_movements(prefix, pulse, past):
    sequences = command_sequences(prefix, pulse=pulse)
    return [movement(np.concatenate((s[:PREFIX_TICKS], np.repeat(s[PREFIX_TICKS:PREFIX_TICKS+1], 5, axis=0))), past)
            for s in sequences]


def tracked_motion(poses, frame0, frame1):
    """XY displacement (body frame at frame0) and heading change from frame0 to frame1 of the registered poses."""
    p0, R0 = np.asarray(poses[frame0].position, float), np.asarray(poses[frame0].rotation, float)
    p1, R1 = np.asarray(poses[frame1].position, float), np.asarray(poses[frame1].rotation, float)
    relative = R0.T@R1
    return (R0.T@(p1-p0))[:2], float(np.arctan2(relative[1, 0], relative[0, 0]))


def current_ratio(samples, now_ns):
    """(ratio, samples used): the clipped median of the last 3 s, or 1.0 with fewer than two samples."""
    recent = [s['ratio'] for s in samples if now_ns-s['ns'] <= WINDOW_NS]
    if len(recent) < MIN_SAMPLES:
        return 1., len(recent)
    return float(np.clip(np.median(recent), *CLIP)), len(recent)


def adapt(forecast, movements, translation, rotation):
    """The (6, 8, 5) forecast with each candidate's XY and yaw scaled by its movement type's ratios."""
    adapted = forecast.copy()
    yaw = np.arctan2(forecast[:, :, 2], forecast[:, :, 3])
    for i, kind in enumerate(movements):
        adapted[i, :, :2] *= translation.get(kind, 1.)
        yaw[i] *= rotation.get(kind, 1.)
    adapted[:, :, 2], adapted[:, :, 3] = np.sin(yaw), np.cos(yaw)
    return adapted


class AdaptiveTravelForecaster:
    """C1A: see the module docstring. `base` is the C1 `CommandHistoryForecaster`; `log` keeps each plan's ratios."""

    def __init__(self, base):
        self.base = base
        self.decisions = deque(maxlen=64)
        self.translation = defaultdict(lambda: deque(maxlen=64))
        self.rotation = defaultdict(lambda: deque(maxlen=64))
        self.log = []

    def forecast(self, context, prefix, *, pulse):
        forecast = self.base.forecast(context, prefix, pulse=pulse)
        now = context.measured_ns
        kind, translation, rotation = self._sample(context)
        if kind in ADAPTED:
            if translation is not None:
                self.translation[kind].append(translation)
            if rotation is not None:
                self.rotation[kind].append(rotation)
        self.decisions.append(dict(ns=now, frame=context.frame, history=command_history(context.history)))
        translation_ratio = {k: current_ratio(self.translation[k], now)[0] for k in ADAPTED}
        rotation_ratio = {k: current_ratio(self.rotation[k], now)[0] for k in ADAPTED}
        movements = candidate_movements(prefix, pulse, planned_past(context.commitments, now))
        adapted = adapt(forecast, movements, translation_ratio, rotation_ratio)
        if not np.isfinite(adapted).all():
            raise ValueError('finite adaptive forecast required')
        self.log.append(dict(ns=now, candidate_movement=movements, translation_ratio=translation_ratio,
                             rotation_ratio=rotation_ratio, new_sample_movement=kind,
                             new_translation_sample=translation, new_rotation_sample=rotation))
        return adapted

    def _sample(self, context):
        """(movement, translation sample, rotation sample) against the longest usable earlier plan."""
        now, frame, poses = context.measured_ns, context.frame, context.poses
        by_ns = {d['ns']: d for d in self.decisions}
        for h in HORIZONS:
            earlier = by_ns.get(now-h*PERIOD_NS)
            if earlier is None or earlier['frame'] not in poses or frame not in poses:
                continue
            ticks = planned_tape(context.commitments, earlier['ns'], h)
            if ticks is None:
                continue
            if not np.any(np.abs(ticks) > 1e-9):
                return None, None, None
            kind = window_movement(ticks, planned_past(context.commitments, earlier['ns']))
            sequence = np.zeros((6, 8, 3))
            sequence[:, :h], sequence[:, h:] = ticks, ticks[-1]
            forecast = self.base.model.predict(sequence, earlier['history'])[0, h-1]
            travelled, turned = tracked_motion(poses, earlier['frame'], frame)
            sample = dict(ns=int(now), horizon_steps=h, movement=kind)
            translation = rotation = None
            if np.linalg.norm(forecast[:2]) >= MIN_TRANSLATION_M:
                translation = dict(sample, ratio=float(np.linalg.norm(travelled)/np.linalg.norm(forecast[:2])))
            if abs(forecast[2]) >= MIN_ROTATION_RAD:
                rotation = dict(sample, ratio=float(abs(turned)/abs(forecast[2])))
            return kind, translation, rotation
        return None, None, None
