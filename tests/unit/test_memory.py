"""The walker memory (P9): the occupancy labels' view test, the baselines' sightings, and the memory's ego-motion warp."""
import math

import numpy as np
import torch

from go2nav.data import memory_labels as ml
from go2nav.eval import memory_baselines as mb
from go2nav.models.memory import BEV_CELLS, WalkerMemory
from go2nav.training.memory import motions

IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])


def pose(x, y, heading):
    return np.array([x, y, 0., 0., 0., math.sin(heading/2), math.cos(heading/2)])


def hot(x, y):
    m = torch.zeros(1, 1, BEV_CELLS, BEV_CELLS)
    half = BEV_CELLS*ml.CELL_M/2
    m[0, 0, int((x+half)//ml.CELL_M), int((y+half)//ml.CELL_M)] = 1.
    return m


def where(m):
    half = BEV_CELLS*ml.CELL_M/2
    i, j = divmod(int(m[0, 0].flatten().argmax()), BEV_CELLS)
    return round((i+.5)*ml.CELL_M-half, 2), round((j+.5)*ml.CELL_M-half, 2)


def test_a_wall_across_the_view_hides_the_cells_beyond_it():
    centres, halves = np.array([[.9, 0.]]), np.array([[.05, 1.]])     # a wall 0.9 m ahead, across the view
    visible = ml.cell_visible(IDENTITY, centres, halves).reshape(ml.CELLS, ml.CELLS)
    x = ml.CENTRES
    middle = ml.CELLS//2
    assert visible[(x > .5) & (x < .8), middle].all()                 # between the camera and the wall
    assert not visible[x > 1.1, middle].any()                         # behind it
    assert not visible[x < 0, :].any()                                # behind the robot


def test_the_view_test_follows_the_robot_s_heading():
    left = ml.cell_visible(pose(0., 0., math.pi/2), np.zeros((0, 2)), np.zeros((0, 2))).reshape(ml.CELLS, ml.CELLS)
    straight = ml.cell_visible(IDENTITY, np.zeros((0, 2)), np.zeros((0, 2))).reshape(ml.CELLS, ml.CELLS)
    assert (left == straight).all()                                   # the grid is in the body frame


def test_the_warp_moves_remembered_cells_against_the_robot_s_motion():
    assert where(WalkerMemory.warp(hot(1.1, .3), torch.tensor([[.2, 0., 0.]]))) == (.9, .3)
    assert where(WalkerMemory.warp(hot(1.1, .3), torch.tensor([[0., .4, 0.]]))) == (1.1, -.1)
    # a quarter turn left puts what was ahead on the right
    assert where(WalkerMemory.warp(hot(1.1, -.1), torch.tensor([[0., 0., math.pi/2]]))) == (-.1, -1.1)


def test_step_motions_are_each_pose_in_the_body_frame_of_the_one_before():
    poses = np.stack([pose(0., 0., 0.), pose(0., 0., 0.), pose(1., 0., math.pi/2), pose(1., 0., math.pi/2),
                      pose(1., 1., math.pi/2)])
    m = motions(poses)                                                # every second frame: poses 0, 2, 4
    assert np.allclose(m[0], 0.)
    assert np.allclose(m[1], [1., 0., math.pi/2], atol=1e-6)
    assert np.allclose(m[2], [1., 0., 0.], atol=1e-6)                 # forward in the turned body frame


def test_average_precision_ranks_and_shares_ties():
    label = np.array([True, False, True, False])
    assert mb.average_precision(np.array([.9, .8, .7, .1]), label) == (1+2/3)/2
    assert mb.average_precision(np.array([1., 1., 1., 1.]), label) == .5   # one tie: the base rate
    assert math.isnan(mb.average_precision(np.ones(3), np.zeros(3, bool)))


def test_a_remembered_group_is_one_walker_at_its_weighted_centre():
    from go2nav.controllers.memory import groups
    p = np.zeros((ml.CELLS, ml.CELLS))
    p[2, 7], p[2, 8], p[3, 8] = .9, .6, .3                           # behind the robot, one group
    p[12, 12] = .8                                                    # ahead, another
    found = sorted(groups(p, .5))
    assert len(found) == 2                                            # the 0.3 cell is below the threshold
    x, y, peak = found[0]
    assert peak == .9 and math.isclose(x, ml.CENTRES[2]) and ml.CENTRES[7] < y < ml.CENTRES[8]


def test_only_walkers_out_of_the_camera_s_view_are_remembered():
    from go2nav.controllers.memory import out_of_view
    assert out_of_view(-.8, 0.) and out_of_view(.2, .9)               # behind, beside
    assert not out_of_view(1.5, 0.)                                   # ahead: the depth map sees it


def test_the_planner_s_selection_reads_remembered_cells_without_changing_the_map():
    from go2nav.perception.mapping import MapSnapshot
    from go2nav.planning.planner import Planner
    empty = frozenset()
    snapshot = MapSnapshot(0, 0, empty, empty, (0., 0., 0.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), 0., 0, 0,
                           frozenset({(1, 1)}), empty, empty, empty)
    planner = Planner(arrival_radius_m=.02)
    assert planner._remembering(snapshot) is snapshot                  # nothing remembered: the map as it is
    planner.remembered = frozenset({(5, 5)})
    assert planner._remembering(snapshot).fine_occupied == {(1, 1), (5, 5)}
    assert snapshot.fine_occupied == {(1, 1)}                          # the map itself is untouched


def _selection(actions):
    from go2nav.planning.actions import command
    rows = [dict(action=a, utility_m=u) for a, u in actions]
    return dict(action=max(actions, key=lambda r: r[1])[0], candidates=rows, requested_command=None,
                memory_forecast_candidates=[dict(action=a, nominal_predicted_path_clear=True, reserve_recovery_path_clear=False)
                                            for a, _ in actions])


def _forecast(moves):
    """Per candidate (dx per tick, dyaw per tick): 8 ticks of [x, y, sin yaw, cos yaw]."""
    out = []
    for dx, dyaw in moves:
        t = np.arange(1, 9)
        yaw = dyaw*t
        out.append(np.stack([dx*t, 0*t, np.sin(yaw), np.cos(yaw)], -1))
    return np.stack(out)


def test_a_walker_remembered_behind_rules_out_backing_into_it_but_not_walking_away():
    from go2nav.planning import selection as sel
    actions = [('hold', 0.), ('cmd-0.10+0.00+0.00', .3), ('cmd+0.15+0.00+0.00', .1)]
    prediction = _forecast([(0., 0.), (-.01, 0.), (.015, 0.)])
    result = sel.remembered_veto(_selection(actions), prediction, [[-.6, 0.]])   # 0.3 m behind the rear
    assert result['remembered_veto']['ruled_out'] == ['cmd-0.10+0.00+0.00']
    assert result['action'] == 'cmd+0.15+0.00+0.00'                             # the best action left
    far = sel.remembered_veto(_selection(actions), prediction, [[-2., 0.]])     # out of reach: nothing ruled out
    assert far['action'] == 'cmd-0.10+0.00+0.00' and not far['remembered_veto']['changed']


def test_a_turn_that_swings_the_rear_toward_a_remembered_walker_is_ruled_out():
    from go2nav.planning import selection as sel
    actions = [('hold', 0.), ('cmd+0.00+0.00+0.30', .2), ('cmd+0.00+0.00-0.30', .1)]
    prediction = _forecast([(0., 0.), (0., .03), (0., -.03)])
    # a walker behind and to the right: turning left swings the rear to the right, toward it
    result = sel.remembered_veto(_selection(actions), prediction, [[-.3, -.45]])
    assert result['remembered_veto']['ruled_out'] == ['cmd+0.00+0.00+0.30']
    assert result['action'] == 'cmd+0.00+0.00-0.30'


def test_the_veto_reads_remembered_walkers_without_touching_the_map_cells():
    from go2nav.planning.planner import Planner
    planner = Planner(arrival_radius_m=.02)
    planner.remembered_walkers = ((1., 1.),)
    assert planner.remembered == frozenset()                            # the clearance checks see walls only
