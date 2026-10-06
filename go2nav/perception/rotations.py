"""Rotation helpers and the gyro attitude integrator."""
import math

import numpy as np


def angle(R):
    """Rotation angle of a rotation matrix (radians)."""
    R = np.asarray(R, float)
    return float(np.arctan2(np.linalg.norm([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/2,
                            np.clip((np.trace(R)-1)/2, -1, 1)))


def rotation_increment(rotation_vector):
    """Rodrigues' formula: the rotation matrix of a rotation vector (series form below 1e-6 rad)."""
    vector = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(vector))
    x, y, z = vector
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if theta < 1e-6:
        a, b = 1-theta*theta/6, .5-theta*theta/24
    else:
        a, b = math.sin(theta)/theta, (1-math.cos(theta))/(theta*theta)
    return np.eye(3)+a*cross+b*(cross@cross)


def orthonormalize(rotation):
    """Nearest rotation matrix (SVD)."""
    u, _, vt = np.linalg.svd(rotation)
    return u@vt


class GyroAttitude:
    """Body rotation since the first frame, from the 500 Hz body gyro.

    Each 100-ms frame brings 51 rate samples (the first repeats the previous frame's last); consecutive samples are
    integrated with the midpoint rule (50 increments of 2 ms). The visual odometry's integrator re-orthonormalises the
    result once per frame; the obstacle observer's does not (`orthonormalize=False`), as in the original.
    """

    def __init__(self, orthonormalize=True):
        self.rotation = np.eye(3)
        self.started = False
        self.orthonormalize = orthonormalize

    def update(self, rates):
        """Integrate one frame's (51, 3) rate block (rad/s); the first call only starts the integration."""
        if not self.started:
            self.started = True
            return self.rotation
        rates = np.asarray(rates)
        updated = self.rotation.copy()
        for previous, current in zip(rates[:-1], rates[1:], strict=True):
            updated = updated@rotation_increment((previous+current)*.001)
        self.rotation = orthonormalize(updated) if self.orthonormalize else updated
        return self.rotation
