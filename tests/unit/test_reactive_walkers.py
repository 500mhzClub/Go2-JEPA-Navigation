"""Reactive walkers (sim.obstacles; docs/dynamic/PLAN.md P13 item 3)."""
import numpy as np
import pytest

from go2nav.sim import obstacles
from go2nav.sim.obstacles import PITCH_M, RADIUS_M, ReactiveWalker, robot_gap

DT = .01


def corridor(cells=5):
    """A straight passage of `cells` cells along x."""
    layout = dict(cells=[[x, 0] for x in range(cells)], edges=[[[x, 0], [x+1, 0]] for x in range(cells-1)])
    return dict(evaluation_layout=layout)


def walker(start=(0, 0), seed=1, **kwargs):
    return ReactiveWalker(obstacles.passages(corridor()), start, .3, np.random.default_rng(seed), **kwargs)


def run(w, robot_xy, robot_yaw, seconds):
    xs, gaps = [], []
    for _ in range(int(seconds/DT)):
        w.step(DT, robot_xy, robot_yaw)
        xs.append(float(w.xy[0]))
        gaps.append(float(robot_gap(w.xy, robot_xy, robot_yaw)[0]-RADIUS_M))
    return np.array(xs), np.array(gaps)


def test_robot_gap_is_the_oriented_rectangle():
    assert np.isclose(robot_gap([(1., 0.)], (0., 0.), 0.)[0], 1.-obstacles.ROBOT_HALF_LENGTH_M)
    assert np.isclose(robot_gap([(0., 1.)], (0., 0.), 0.)[0], 1.-obstacles.ROBOT_HALF_WIDTH_M)
    assert np.isclose(robot_gap([(0., 1.)], (0., 0.), np.pi/2)[0], 1.-obstacles.ROBOT_HALF_LENGTH_M)
    assert robot_gap([(0., 0.)], (0., 0.), 0.)[0] < 0                       # inside
    assert np.isclose(robot_gap([(1., 0.)], (0., 0.), None)[0], 1.-obstacles.ROBOT_REACH_M)


def test_it_squeezes_past_a_robot_standing_to_one_side():
    robot, yaw = (2*PITCH_M, -.38), 0.                                       # against the right wall, along the passage
    xs, gaps = run(walker(patience=30.), robot, yaw, 90.)
    assert gaps.min() >= obstacles.PASS_GAP_M-1e-9
    assert xs.max() > robot[0]+.6                                            # it went past


def test_blocked_it_waits_at_its_gap_then_turns_back():
    robot, yaw = (2*PITCH_M, 0.), 0.                                         # in the middle: no lane is wide enough
    w = walker(patience=3.)
    xs, gaps = run(w, robot, yaw, 60.)
    assert gaps.min() >= obstacles.PASS_GAP_M-1e-9
    assert xs.max() < robot[0]                                               # never past
    assert w.turned >= 1


def test_a_yielding_walker_keeps_its_distance():
    robot, yaw = (2*PITCH_M, 0.), 0.
    _, gaps = run(walker(patience=30., yields=True), robot, yaw, 40.)
    assert gaps.min() >= obstacles.YIELD_GAP_M-.02


def test_two_walkers_pass_each_other_without_touching():
    placement = [dict(start_cell=[0, 0], speed_m_s=.3, walk_seed=3, height_m=.9, behaviour='reactive', patience_s=20.,
                      yields=False),
                 dict(start_cell=[4, 0], speed_m_s=.3, walk_seed=4, height_m=.9, behaviour='reactive', patience_s=20.,
                      yields=False)]

    class Entity:
        def set_pos(self, *args, **kwargs):
            pass
    field = obstacles.ObstacleField(corridor(), placement, [Entity(), Entity()])
    gaps = []
    for k in range(int(60./DT)):
        a, b = field.update(DT, None, k*DT)
        gaps.append(np.hypot(a[0]-b[0], a[1]-b[1])-2*RADIUS_M)
    assert min(gaps) >= obstacles.PASS_GAP_M-1e-9
    xy = field.arrays()['xy']
    crossed = np.sign(xy[:, 0, 0]-xy[:, 1, 0])
    assert (crossed != crossed[0]).any()                                     # they passed one another


def line(cells=8):
    spec = corridor(cells)
    episode = dict(home_se2_world=[0., 0., 0.], beacon_xy_world=[(cells-1)*PITCH_M, 0.])
    return spec, episode


def test_reactive_placement_keeps_the_other_draws_and_replays():
    spec, episode = line()
    plain = obstacles.place_obstacles(spec, episode, 2, 7)
    reactive = obstacles.place_obstacles(spec, episode, 2, 7, reactive=True)
    for p, r in zip(plain, reactive):
        assert {k: r[k] for k in p} == p
        assert r['behaviour'] == 'reactive' and 2. <= r['patience_s'] <= 20. and isinstance(r['yields'], bool)
    assert reactive == obstacles.place_obstacles(spec, episode, 2, 7, reactive=True)
    with pytest.raises(ValueError):
        obstacles.place_obstacles(spec, episode, 2, 7, polite=False, reactive=True)
