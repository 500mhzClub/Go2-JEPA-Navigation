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


def test_a_few_centimetres_from_the_target_it_closes_in_though_the_forecast_is_a_centimetre_out():
    def pulses(shift):                              # the perfect pulses, every forecast displaced by the same error
        p = forecast(pulse=True)
        p[:, PREFIX_TICKS:, :2] += shift
        return p
    go = lambda goal, shift: cost.select(pulses(shift), list(LATTICE), np.asarray(goal, float), None, None,
                                         arrival_radius_m=.02, pulse=True)
    near = go((.022, .009), (.006, .006))           # 2.4 cm off; the hold is forecast to drift into the 2-cm disk
    assert command(near['action'])[0] > 0           # it was free, and the robot stood (still maze 14/0)
    ahead = go((.039, 0.), (0., -.007))             # 3.9 cm ahead; the forecasts end 7 mm to the side
    assert command(ahead['action'])[0] > 0          # the bearing of the centre swung 15 degrees, and it stood (20/1)
    rows = {r['action']: r for r in ahead['candidates']}
    assert abs(rows['hold']['to_go_s']-math.hypot(.039, .007)/cost.V_REF) < 1e-6      # its line crosses the disk: no turn
    assert command(go((0., .3), (0., 0.))['action'])[2] > 0                  # further off and abeam: it turns to it


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


def test_recorded_decisions_replay_to_the_same_choice(tmp_path):
    from go2nav.eval import replay
    rng = np.random.default_rng(0)
    rows, made = [], []
    for k in range(12):
        row = dict(prediction=forecast(pulse=k == 3), goal_body=rng.uniform(-1, 1, 2), scan_error=1. if k == 5 else None,
                   points=None if k == 0 else np.column_stack([rng.uniform(.5, 1.5, (30, 2)), np.zeros(30)]),
                   target=bool(k % 2), pulse=k == 3, cells=None if k < 2 else {(60+k, 3), (61, -4)})
        chosen = cost.select(row['prediction'], list(LATTICE), row['goal_body'], row['scan_error'], row['points'],
                             arrival_radius_m=.02, target=row['target'], pulse=row['pulse'], cells=row['cells'])
        rows.append(dict(row, frame=4*k))
        made.append(dict(frame=4*k, action=chosen['action'], route_status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', selection={}))
    made[7]['action'] = 'hold' if made[7]['action'] != 'hold' else 'cmd+0.15+0.00+0.00'       # one the code no longer makes
    replay.save(tmp_path/'decision_inputs.npz', rows, list(LATTICE), .02)
    (tmp_path/'planning.json').write_text(__import__('json').dumps(made))
    out = replay.replay(tmp_path)
    assert out['plans'] == 12 and out['changed'] == 1 and replay.runs_of(tmp_path) == [tmp_path]


def test_paired_cohorts_and_a_bar_read_before_the_end():
    from go2nav.eval import compare
    a = {(10, 0): True, (10, 1): False, (11, 0): False, (11, 1): True, (12, 0): True}
    b = {(10, 0): True, (10, 1): True, (11, 0): True, (11, 1): False, (13, 0): True}
    out = compare.paired(a, b)
    assert out['missions'] == 4 and out['gains'] == [(10, 1), (11, 0)] and out['losses'] == [(11, 1)] and out['p'] == 1.
    assert compare.sign_test(9, 0) < .01 < compare.sign_test(7, 4)
    assert compare.bar(52, 66, 80, 62)['state'] == 'open' and compare.bar(47, 66, 80, 62)['state'] == 'out of reach'
    assert compare.bar(62, 70, 80, 62)['state'] == 'secured'


def test_a_look_around_turns_on_the_spot_though_an_arc_is_forecast_to_turn_further():
    p = forecast()
    arc = list(LATTICE).index('cmd+0.45+0.00+0.45')
    yaw = np.arctan2(p[arc, :, 2], p[arc, :, 3])+.02*(np.arange(HORIZONS) >= PREFIX_TICKS)      # the arc reads 0.02 rad on
    p[arc, :, 2], p[arc, :, 3] = np.sin(yaw), np.cos(yaw)
    chosen = cost.select(p, list(LATTICE), np.zeros(2), 1., None, arrival_radius_m=.02)
    assert command(chosen['action']) == (0., 0., .45)


def test_floor_not_yet_seen_is_not_walked_on_and_is_asked_for():
    unseen = [translating(a) and command(a)[0] > 0 for a in LATTICE]                             # all the floor ahead
    chosen = cost.select(forecast(), list(LATTICE), np.array([1., 0.]), None, None, arrival_radius_m=.02, uncovered=unseen)
    assert not translating(chosen['action']) or command(chosen['action'])[0] < 0
    assert command(chosen['coverage_wanted']) == (.45, 0., 0.)
    seen = cost.select(forecast(), list(LATTICE), np.array([1., 0.]), None, None, arrival_radius_m=.02,
                       uncovered=[False]*len(LATTICE))
    assert command(seen['action']) == (.45, 0., 0.) and seen['coverage_wanted'] is None
