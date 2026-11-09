"""The range sensor (perception.range, sim.range_sensor; docs/dynamic/PLAN.md P13 item 1)."""
import math

import numpy as np

from go2nav.perception import range as beams
from go2nav.sim import range_sensor
from go2nav.sim.obstacles import RADIUS_M

LEVEL = (1., 0., 0., 0.)          # wxyz: the body level, facing +x
BASE = np.array([0., 0., .30])    # the base origin 0.30 m above the floor
FORWARD = int(np.argmin(np.abs(beams.AZIMUTHS)))
LEFT = int(np.argmin(np.abs(beams.AZIMUTHS-math.pi/2)))
BEHIND = 0                        # azimuth -pi


def spec(boxes=()):
    return dict(layout_index=3, geometry=dict(wall_boxes=[dict(centre_xyz=list(c), size_xyz=list(s), yaw_rad=y)
                                                          for c, s, y in boxes]))


def test_a_wall_ahead_and_a_walker_beside_and_behind():
    wall = ((3., 0., .7), (.08, 4., 1.4), 0.)                     # its face at x = 2.96
    sensor = range_sensor.RangeSensor(spec([wall]), noise_sigma_m=0.)
    walkers = [(.3, 1.5), (-2., 0.)]
    t = sensor.true_distances(BASE, LEVEL, walkers, [.9, .9])
    assert np.isclose(t[FORWARD], 2.96-beams.MOUNT_BODY_M[0])
    assert np.isclose(t[LEFT], 1.5-RADIUS_M)                       # the mount is at y = 0, above x = 0.30
    assert np.isclose(t[BEHIND], 2.+beams.MOUNT_BODY_M[0]-RADIUS_M)
    assert not np.isfinite(t[LEFT+60])                             # nothing to the front-left: open space, level beams


def test_a_turned_box_is_hit_where_its_face_is():
    wall = ((2., 0., .7), (.1, 4., 1.4), math.pi/4)                # turned 45°: the forward beam meets a slanted face
    sensor = range_sensor.RangeSensor(spec([wall]), noise_sigma_m=0.)
    t = sensor.true_distances(BASE, LEVEL)
    assert np.isclose(t[FORWARD], 2.-.05*math.sqrt(2)-beams.MOUNT_BODY_M[0], atol=1e-9)


def test_the_ring_tilts_with_the_body_and_meets_the_floor():
    pitch = .2                                                     # nose down
    wxyz = (math.cos(pitch/2), 0., math.sin(pitch/2), 0.)
    sensor = range_sensor.RangeSensor(spec(), noise_sigma_m=0.)
    t = sensor.true_distances(BASE, wxyz)
    R = range_sensor.rotation_wxyz(wxyz)
    height = (BASE+R@beams.MOUNT_BODY_M)[2]
    assert np.isclose(t[FORWARD], height/math.sin(pitch))
    assert not np.isfinite(t[BEHIND])                              # the rear of the ring rises


def test_near_blind_spot_range_limit_and_exact_replay():
    sensor = range_sensor.RangeSensor(spec(), noise_sigma_m=0.)
    near = [(beams.MOUNT_BODY_M[0]+RADIUS_M+.03, 0.)]              # the walker's face 3 cm from the mount
    ranges, valid = sensor.scan(BASE, LEVEL, near, [.9], frame=0)
    assert not valid[FORWARD] and ranges[FORWARD] == 0.
    noisy = range_sensor.RangeSensor(spec([((3., 0., .7), (.08, 4., 1.4), 0.)]))
    a, b = noisy.scan(BASE, LEVEL, frame=7), noisy.scan(BASE, LEVEL, frame=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], noisy.scan(BASE, LEVEL, frame=8)[0])
    assert abs(a[0][FORWARD]-(2.96-beams.MOUNT_BODY_M[0])) < .05


def test_returns_become_body_points():
    ranges = np.zeros(beams.BEAMS, np.float32)
    valid = np.zeros(beams.BEAMS, bool)
    ranges[FORWARD], valid[FORWARD] = 2., True
    ranges[LEFT], valid[LEFT] = 1., True
    points = beams.body_points(ranges, valid)
    assert np.allclose(sorted(map(tuple, points.round(6))), sorted([(2.3, 0., .12), (.3, 1., .12)]))
