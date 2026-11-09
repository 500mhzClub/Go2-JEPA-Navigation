"""The range sensor as the robot knows it (docs/dynamic/PLAN.md, P13 item 1): the beam geometry, and its returns as points.

The sensor is modelled after the Go2's L1 LiDAR (360° x 90°, 0.05 m near blind spot), idealised to one ring of `BEAMS`
beams in the body's horizontal plane, from `MOUNT_BODY_M`. Beam k points at azimuth `AZIMUTHS[k]` (0 straight ahead,
counter-clockwise, from -pi). A scan is (ranges_m, valid): a beam is valid when it returned between `MIN_RANGE_M` and
`MAX_RANGE_M`. The simulator casts the beams (`go2nav.sim.range_sensor`); perception reads them only through here.
"""
import numpy as np

BEAMS = 720
MOUNT_BODY_M = np.array([.30, 0., .12])
MIN_RANGE_M, MAX_RANGE_M = .05, 20.
AZIMUTHS = -np.pi+np.arange(BEAMS)*(2*np.pi/BEAMS)
DIRECTIONS_BODY = np.stack((np.cos(AZIMUTHS), np.sin(AZIMUTHS), np.zeros(BEAMS)), axis=1)


def body_points(ranges_m, valid):
    """The valid returns as (n, 3) points in the body frame."""
    ranges_m, valid = np.asarray(ranges_m, float), np.asarray(valid, bool)
    return MOUNT_BODY_M+ranges_m[valid, None]*DIRECTIONS_BODY[valid]
