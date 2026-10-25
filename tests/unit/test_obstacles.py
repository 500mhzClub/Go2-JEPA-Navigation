import numpy as np

from go2nav.sim import obstacles
from go2nav.sim.obstacles import PITCH_M, Walker, passages, place_obstacles

# A 3 x 2 maze: (0,0)-(1,0)-(2,0) along the bottom, (2,0)-(2,1)-(1,1)-(0,1) back along the top.
SPEC = dict(evaluation_layout=dict(cells=[[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
                                   edges=[[[0, 0], [1, 0]], [[1, 0], [2, 0]], [[2, 0], [2, 1]], [[2, 1], [1, 1]],
                                          [[1, 1], [0, 1]]]))
EPISODE = dict(home_se2_world=[0., 0., 0.], beacon_xy_world=[0., PITCH_M])


def test_placement_is_seeded_and_keeps_away_from_home_and_beacon():
    a = place_obstacles(SPEC, EPISODE, 2, 7)
    assert a == place_obstacles(SPEC, EPISODE, 2, 7)
    cells = {tuple(o['start_cell']) for o in a}
    assert len(cells) == 2 and cells <= {(2, 1), (1, 1)}    # three passages or more from home, not the beacon cell
    assert obstacles.placement_seed('s', 1, 0, 2) != obstacles.placement_seed('s', 1, 0, 3)


def test_walker_follows_passages_from_centre_to_centre():
    walker = Walker(passages(SPEC), (0, 0), .3, np.random.default_rng(0))
    walker.pause_left = 0.
    visited = [walker.cell]
    for _ in range(20000):
        walker.pause_left = 0.
        walker.step(.002, None)
        if walker.cell != visited[-1]:
            assert walker.cell in passages(SPEC)[visited[-1]]
            assert np.allclose(walker.xy, np.asarray(walker.cell, float)*PITCH_M)
            visited.append(walker.cell)
    assert len(visited) > 5


def test_walker_yields_then_turns_back():
    walker = Walker(passages(SPEC), (0, 0), .3, np.random.default_rng(0))
    walker.pause_left, walker.target = 0., (1, 0)
    for _ in range(500):                                   # 1 s: 0.3 m along x
        walker.step(.002, None)
    held = walker.xy.copy()
    robot = held+np.array([.5, .1])                        # ahead of it, near its line
    steps = 0
    while walker.target == (1, 0):
        assert np.allclose(walker.step(.002, robot), held)
        steps += 1
    assert abs(steps*.002-obstacles.GIVE_UP_S) < .01      # gave up after 3 s: now heading back to (0, 0)
    assert walker.target == (0, 0) and walker.cell == (1, 0)
    walker.step(.002, robot)
    assert walker.xy[0] < held[0]                          # the robot is behind it now
