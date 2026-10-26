"""Corpus v2 (go2nav.data.lewm_corpus_v2) and non-polite obstacles (go2nav.sim.obstacles)."""
import math

import numpy as np

from go2nav.data import lewm_corpus_v2 as v2
from go2nav.sim import obstacles as moving
from go2nav.sim.obstacles import GIVE_UP_S, PITCH_M, Walker

CORRIDOR = {(i, 0): [n for n in ((i-1, 0), (i+1, 0)) if 0 <= n[0] < 6] for i in range(6)}
LAYOUT = dict(evaluation_layout=dict(cells=[[i, 0] for i in range(6)], edges=[[[i, 0], [i+1, 0]] for i in range(5)]))


def walker(polite):
    w = Walker(CORRIDOR, (2, 0), .3, np.random.default_rng(0), polite=polite)
    w.cell, w.target, w.pause_left = (2, 0), (1, 0), 0.   # walking towards x = 0
    return w


def test_a_non_polite_obstacle_walks_up_to_the_robot_and_never_turns_back():
    robot = np.array([2*PITCH_M-1., 0.])   # 1 m ahead of it
    polite, rude = walker(True), walker(False)
    for _ in range(int(2*GIVE_UP_S/.02)):
        polite.step(.02, robot)
        rude.step(.02, robot)
    assert polite.target == (3, 0)                                        # it gave up and turned back
    assert rude.target == (1, 0)                                          # still waiting, facing the robot
    gap = np.linalg.norm(rude.xy-robot)-moving.RADIUS_M-moving.ROBOT_REACH_M
    assert 0. < gap <= moving.ASSERTIVE_GAP_M+.01


def test_a_non_polite_obstacle_stops_short_of_the_robot_from_the_side_too():
    w = walker(False)
    robot = w.xy+np.array([-.9, .35])      # ahead and to the side of its path
    for _ in range(200):
        w.step(.02, robot)
    assert np.linalg.norm(w.xy-robot) >= moving.RADIUS_M+moving.ROBOT_REACH_M


def test_placements_record_politeness_only_when_off_and_draw_the_same():
    episode = dict(home_se2_world=[0., 0., 0.], beacon_xy_world=[0., 0.])
    polite = moving.place_obstacles(LAYOUT, episode, 2, 7)
    rude = moving.place_obstacles(LAYOUT, episode, 2, 7, polite=False)
    assert all('polite' not in o for o in polite) and all(o.pop('polite') is False for o in rude)
    assert polite == rude


def test_random_commands_lie_on_the_corpus_lattice_with_the_lateral_share():
    rng = np.random.default_rng(0)
    draws = np.array([v2.random_command(rng) for _ in range(4000)])
    for k, axis in enumerate(('forward', 'lateral', 'yaw')):
        assert set(np.round(draws[:, k], 3)) <= set(v2.AXES[axis])
    assert abs((draws[:, 1] != 0).mean()-v2.LATERAL_SHARE) < .03


def test_unit_obstacles_are_keyed_by_unit_and_fit_the_layout():
    first = v2.unit_obstacles(LAYOUT, [0., 0., 0.], [0., 0.], 3, 'free')
    assert first == v2.unit_obstacles(LAYOUT, [0., 0., 0.], [0., 0.], 3, 'free')
    for maze in range(40):
        placement, record = v2.unit_obstacles(LAYOUT, [0., 0., 0.], [0., 0.], maze, 'mission')
        assert record['count'] <= 3 and (placement is None) == (record['count'] == 0)   # three cells 3+ hops out


def driver(movers=(), half_width=.65):
    """A free driver in a passage along x (walls at y = +-`half_width`; 1.22 m wide by default), with obstacles at
    `movers`."""
    walls = [dict(centre_xyz=[2., y, .7], size_xyz=[6., .08, 1.4], yaw_rad=0.) for y in (-half_width, half_width)]
    return v2.FreeDriverV2(dict(geometry=dict(wall_boxes=walls)), np.random.default_rng(0), movers=lambda: list(movers))


def pose(x, y, yaw):
    return [x, y, .3, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]


def test_a_sideways_step_into_a_wall_becomes_a_turn_and_into_open_space_passes():
    d = driver()
    assert d._safe((0., .15, 0.), pose(2., .15, 0.))[:2] == (0., 0.)      # 0.46 m to the wall on the left
    assert d._safe((0., .15, 0.), pose(2., 0., 0.)) == (0., .15, 0.)      # 0.61 m: clear
    assert d._safe((0., .15, 0.), pose(2., .15, -math.pi/2)) == (0., .15, 0.)   # facing -y, left is +x: open
    assert d.admissible((0., -.15, 0.), pose(2., .15, 0.))                 # the other side: 0.76 m


def test_obstacles_count_in_the_stand_off_and_walls_hide_them():
    assert driver()._safe((.3, 0., 0.), pose(2., 0., 0.)) == (.3, 0., 0.)
    assert driver([(2.8, 0.)])._safe((.3, 0., 0.), pose(2., 0., 0.))[:2] == (0., 0.)    # its surface 0.63 m ahead
    assert driver([(2.5, 0.)])._safe((.3, 0., 0.), pose(2., 0., 0.)) == (-.12, 0., 0.)  # 0.33 m: back out
    beside = driver([(2., -.6)], half_width=3.)                         # in the open, 0.42 m from its surface
    assert beside._safe((0., 0., 0.), pose(2., 0., 0.)) == (0., .12, 0.)  # step away sideways
    assert driver([(3., 0.)]).in_sight([2., 0.]) is not None
    assert driver([(3., 1.5)]).in_sight([2., 0.]) is None                # beyond the wall at y = 0.65
