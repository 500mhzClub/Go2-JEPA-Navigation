"""Command safety limits of the Go2 (the platform manifest's locomotion safety block).

Every requested command is clipped to forward ±0.3 m/s, lateral 0, yaw ±0.5 rad/s, then rate-limited per control tick
(forward 0.25 m/s, yaw 0.35 rad/s) from the previous executed command. Computed in float32, as on the robot.
"""
import numpy as np

LOWER = np.array([-.3, 0., -.5], dtype=np.float32)
UPPER = np.array([.3, 0., .5], dtype=np.float32)
DELTA = np.array([.25, 0., .35], dtype=np.float32)


def apply_limits(requested, previous):
    """Executed commands (n, T, 3) float32 for requested (n, T, 3) after previous (n, 3) executed commands."""
    requested = np.asarray(requested)
    executed = np.empty_like(requested, dtype=np.float32)
    prev = np.asarray(previous).astype(np.float32, copy=True)
    for t in range(requested.shape[1]):
        bounded = np.clip(requested[:, t, :], LOWER, UPPER)
        bounded = np.clip(bounded, prev-DELTA, prev+DELTA)
        executed[:, t, :] = bounded
        prev = bounded
    return executed
