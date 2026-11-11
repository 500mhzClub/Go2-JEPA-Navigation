"""The decision by an intrinsic cost (planning.cost; docs/dynamic/PLAN.md P18)."""
import math

import numpy as np

from go2nav.planning import cost
from go2nav.planning.actions import COMMIT_TICKS, HORIZONS, LATTICE, PREFIX_TICKS, command, translating


def forecast(candidates=LATTICE, pulse=False):
    """What a perfect forecaster gives: each candidate's command integrated over its commit after a still prefix."""
    out = np.zeros((len(candidates), HORIZONS, 5), np.float32)
    for i, action in enumerate(candidates):
        vx, vy, wz = command(action)
        x = y = yaw = 0.
        for t in range(HORIZONS):
            if PREFIX_TICKS <= t < PREFIX_TICKS+(1 if pulse else COMMIT_TICKS):
                x, y = x+.1*(vx*math.cos(yaw)-vy*math.sin(yaw)), y+.1*(vx*math.sin(yaw)+vy*math.cos(yaw))
                yaw += .1*wz
            out[i, t] = (x, y, math.sin(yaw), math.cos(yaw), -1000.)
    return out


def choose(goal=(1., 0.), scan=None, points=None, cells=None, pulse=False, target=True):
    return cost.select(forecast(pulse=pulse), list(LATTICE), np.asarray(goal, float), scan, points,
                       arrival_radius_m=.02, target=target, pulse=pulse, cells=cells)


def test_in_the_open_it_walks_to_the_waypoint_and_turns_to_one_beside_it():
    assert command(choose()['action']) == (.45, 0., 0.)                    # straight on, as fast as the lattice goes
    left = command(choose(goal=(0., 1.))['action'])
    assert left[2] > 0                                                      # a waypoint to the left: it turns left
    behind = command(choose(goal=(-1., .2))['action'])
    assert behind[2] > 0 and behind[0] <= 0                                 # behind: it turns, it does not walk away


def test_it_holds_on_the_waypoint_and_turns_to_a_scan_heading():
    assert choose(goal=(.005, .005))['action'] == 'hold'
    assert choose(goal=(.01, -.02), target=False)['action'] != 'hold'      # not the mission's target: it does not stand
    assert command(choose(goal=(0., 0.), scan=1.)['action']) == (0., 0., .45)
    assert command(choose(goal=(0., 0.), scan=-1.)['action']) == (0., 0., -.45)
    assert choose(goal=(0., 0.), scan=.0)['action'] == 'hold'


def test_returns_ahead_stop_the_walk_and_not_the_turn():
    wall = np.array([[.35+.14, y, 0.] for y in np.arange(-.5, .51, .02)])  # a wall 14 cm off the nose
    result = choose(points=wall)
    rows = {r['action']: r for r in result['candidates']}
    assert rows['cmd+0.45+0.00+0.00']['breached'] and not translating(result['action']) or command(result['action'])[0] < 0
    assert not rows['hold']['breached'] and result['clearance_now_m'] > cost.FLOOR_M
    far = np.array([[2., y, 0.] for y in np.arange(-.5, .51, .02)])
    assert command(choose(points=far)['action'])[0] > 0


def test_a_robot_under_the_floor_may_still_move_away():
    close = np.array([[.35+.04, 0., 0.]])                                   # a return 4 cm off the nose
    result = choose(goal=(-1., 0.), points=close)
    rows = {r['action']: r for r in result['candidates']}
    assert rows['cmd-0.10+0.00+0.00']['eligible'] and not rows['cmd+0.15+0.00+0.00']['eligible']
    assert not result['all_discarded']


def test_what_the_reflex_would_stop_is_not_chosen():
    cells = {(60, j) for j in range(-30, 31)}                               # obstacle cells 0.6 m ahead
    result = choose(cells=cells)
    rows = {r['action']: r for r in result['candidates']}
    assert rows['cmd+0.45+0.00+0.00']['reflex_would_stop'] and not rows['hold']['reflex_would_stop']
    assert not translating(result['action']) or command(result['action'])[0] < 0
    assert set(rows['hold']) >= {'utility_m', 'eligible', 'cost_s'} and result['selection_objective'] == 'intrinsic_cost_seconds'


def test_beside_the_waypoint_it_turns_to_it_and_does_not_back_up():
    result = choose(goal=(-.05, -.05), pulse=True)                          # 7 cm behind and to the right
    assert command(result['action']) == (0., 0., -.45)
