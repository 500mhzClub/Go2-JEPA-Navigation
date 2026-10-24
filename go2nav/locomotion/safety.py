"""Command safety limits of the Go2 (the platform manifest's locomotion safety block).

Every requested command is clipped to forward ±0.3 m/s, lateral 0, yaw ±0.5 rad/s, then rate-limited per control tick
(forward 0.25 m/s, yaw 0.35 rad/s) from the previous executed command. Computed in float32, as on the robot.

The continuous contract's block (`BLOCKS['continuous']`; docs/dynamic/PLAN.md decision 1) opens the limits to the
range its policy was trained on (configs/locomotion/continuous_ppo.yaml): forward -0.25 to 0.45 m/s, lateral ±0.2 m/s,
yaw ±0.8 rad/s, with lateral rate-limited like forward; the mission guard rises from 0.3 to 0.5 m/s with it.
"""
import numpy as np

LOWER = np.array([-.3, 0., -.5], dtype=np.float32)
UPPER = np.array([.3, 0., .5], dtype=np.float32)
DELTA = np.array([.25, 0., .35], dtype=np.float32)
BLOCKS = dict(primitive=(LOWER, UPPER, DELTA),
              continuous=(np.array([-.25, -.2, -.8], dtype=np.float32), np.array([.45, .2, .8], dtype=np.float32),
                          np.array([.25, .25, .35], dtype=np.float32)))
GUARD_SPEED_M_S = dict(primitive=.3, continuous=.5)


def apply_limits(requested, previous, block='primitive'):
    """Executed commands (n, T, 3) float32 for requested (n, T, 3) after previous (n, 3) executed commands."""
    lower, upper, delta = BLOCKS[block]
    requested = np.asarray(requested)
    executed = np.empty_like(requested, dtype=np.float32)
    prev = np.asarray(previous).astype(np.float32, copy=True)
    for t in range(requested.shape[1]):
        bounded = np.clip(requested[:, t, :], lower, upper)
        bounded = np.clip(bounded, prev-delta, prev+delta)
        executed[:, t, :] = bounded
        prev = bounded
    return executed
