"""The vacate rule (planning.planner, docs/dynamic/PLAN.md P11): out of the way of a walker that waits for the robot."""
import math
from types import SimpleNamespace

import numpy as np

from go2nav.perception.tracks import Track
from go2nav.planning import planner as planning
from go2nav.planning.actions import LATTICE, PREFIX_TICKS, command
from go2nav.planning.planner import Planner

SECOND = 1_000_000_000


def corridor(wall_behind_x=None):
    """Floor from x -3 to 3 m, 1.3 m wide, walls (1-cm cells) along y = +-0.65 m; optionally a wall across at x."""
    floor = {(i, j) for i in range(-60, 60) for j in range(-13, 13)}
    walls = {(i, s*65) for i in range(-300, 300) for s in (-1, 1)}
    if wall_behind_x is not None:
        walls |= {(int(wall_behind_x*100), j) for j in range(-65, 66)}
    return SimpleNamespace(floor=frozenset(floor), occupied=frozenset(), fine_occupied=frozenset(walls),
                           measured_ns=0)


def walker_cells(snapshot, xy):
    """Add the depth map's cells of a walker (radius 0.175 m) at xy."""
    cells = {(i, j) for i in range(-30, 31) for j in range(-30, 31) if math.hypot(i, j) <= 17.5}
    c = np.floor(np.asarray(xy)/.01).astype(int)
    snapshot.fine_occupied = snapshot.fine_occupied | frozenset((i+c[0], j+c[1]) for i, j in cells)
    return snapshot


def forecast():
    """Each lattice candidate integrated as commanded: the prefix holds, then the command for five ticks (body frame)."""
    out = np.zeros((len(LATTICE), 8, 5))
    for k, action in enumerate(LATTICE):
        vx, vy, wz = command(action)
        x = y = h = 0.
        for t in range(8):
            if t >= PREFIX_TICKS:
                x, y, h = x+.1*(vx*math.cos(h)-vy*math.sin(h)), y+.1*(vx*math.sin(h)+vy*math.cos(h)), h+.1*wz
            out[k, t] = (x, y, math.sin(h), math.cos(h), -10.)
    return out


def selection():
    return dict(action='hold', action_index=LATTICE.index('hold'), requested_command=[0., 0., 0.],
                candidates=[dict(action=a) for a in LATTICE])


def track(xy, track_id=0):
    return Track(track_id, frozenset(), 0, history=[(0, np.asarray(xy, float))], motion_updates=2)


def planner_on_a_path(path_from_x=-2.):
    p = Planner(arrival_radius_m=.02, candidates=LATTICE, vacate=True)
    p.crumbs = [np.array([x, 0.]) for x in np.arange(path_from_x, .001, .1)]
    return p


def step(p, snapshot, t_s, position=(0., 0.)):
    p.backup_plan = False
    snapshot.measured_ns = int(t_s*SECOND)
    return p._vacate(selection(), forecast(), snapshot, np.array([*position, 0.]), np.eye(3))


def test_a_walker_waiting_ahead_for_four_seconds_starts_a_retreat_straight_back_along_the_path():
    p, snapshot = planner_on_a_path(), walker_cells(corridor(), (.6, 0.))
    p.tracks = [track((.6, 0.))]
    early = [step(p, snapshot, t) for t in np.arange(0., 3.95, .4)]
    assert all('vacate' not in r and r['requested_command'] == [0., 0., 0.] for r in early)
    started = step(p, snapshot, 4.)
    assert started['vacate_started']['waited_s'] == 4. and started['vacate']['phase'] == 'retreat'
    assert started['requested_command'] == [-.10, 0., 0.] and started['action'] == 'hold' and p.backup_plan
    # the walker's own cells (its face 0.425 m away, inside the footprint) do not block the retreat
    assert started['vacate']['walker_distance_m'] == .6


def test_a_moving_walker_or_one_behind_never_triggers():
    p, snapshot = planner_on_a_path(), corridor()
    for k, t in enumerate(np.arange(0., 8., .4)):
        p.tracks = [track((.8-.02*k, 0.)), track((-.5, 0.), 1)]   # one walking up at 0.05 m/s, one standing behind
        p.tracks[0].history = [(0, np.array([.8-.2*k if k % 2 else .8, 0.]))]
        assert 'vacate_started' not in step(p, snapshot, t)
    p.tracks = [track((-.5, 0.), 1)]
    assert 'vacate_started' not in step(p, snapshot, 9.)


def test_the_retreat_follows_the_path_round_a_corner():
    p, snapshot = Planner(arrival_radius_m=.02, candidates=LATTICE, vacate=True), corridor()
    snapshot.floor = frozenset({(i, j) for i in range(-60, 60) for j in range(-60, 60)})
    snapshot.fine_occupied = frozenset()
    # the robot came up from (-.3, -2) then turned to +x: the path behind bends to the robot's right-behind
    p.crumbs = [np.array([-.3, y]) for y in np.arange(-2., -.29, .1)]+[np.array([x, 0.]) for x in (-.2, -.1, 0.)]
    p.crumbs[-4] = np.array([-.3, -.1])
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    assert r['vacate']['phase'] == 'retreat'
    vx, _, wz = r['requested_command']
    assert vx < 0 and wz > 0   # reversing while turning left swings the rear to the right, towards the path


def test_no_admissible_step_holds_then_ends_blocked_with_a_cooldown():
    p, snapshot = planner_on_a_path(), corridor(wall_behind_x=-.42)
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    assert r['vacate']['phase'] == 'blocked' and r['requested_command'] == [0., 0., 0.] and not p.backup_plan
    for k in range(4):
        r = step(p, snapshot, 4.4+.4*k)
    assert r['vacate_ended']['reason'] == 'blocked' and p.vacate_cooldown == planning.VACATE_COOLDOWN_PLANS
    assert 'vacate_started' not in step(p, snapshot, 6.4)         # the cooldown, then the track must wait again


def test_the_walker_turning_off_ends_the_retreat_after_two_clear_seconds():
    p, snapshot = planner_on_a_path(), walker_cells(corridor(), (.6, 0.))
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        step(p, snapshot, t)
    p.tracks = []                                                    # the head drops it: its depth cells still stand
    assert step(p, snapshot, 4.4)['vacate']['phase'] == 'retreat'
    snapshot = corridor()
    p.tracks = [track((.6, 1.3))]                                    # gone into a side passage, 1.4 m away
    waits = [step(p, snapshot, t, position=(-.1, 0.)) for t in (4.8, 5.6, 6.4)]
    assert [w['vacate']['phase'] for w in waits] == ['clear_wait']*3
    assert all(w['requested_command'] == [0., 0., 0.] for w in waits)
    done = step(p, snapshot, 6.8, position=(-.1, 0.))
    assert done['vacate_ended']['reason'] == 'walker_gone' and p.vacate_state is None
    assert len(p.crumbs) < 21                                        # the path walked back is used up


def test_with_no_path_behind_the_retreat_goes_straight_away_from_the_walker():
    p, snapshot = planner_on_a_path(path_from_x=-.1), corridor()
    p.tracks = [track((.6, .3))]                                     # ahead and to the left
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    assert r['vacate']['phase'] == 'retreat' and not p.crumbs
    target = np.array(r['vacate']['target_xy_m'])
    assert np.allclose(target, -.5*np.array([.6, .3])/np.hypot(.6, .3), atol=1e-3)
    assert r['requested_command'][0] < 0 and r['requested_command'][2] > 0   # turning left, the reverse bends right, away


def test_a_blocked_step_records_why():
    p, snapshot = planner_on_a_path(), corridor(wall_behind_x=-.42)
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    why = r['vacate']['why']
    assert why['clearance'] == 3 and why['nearer'] == 0 and np.isclose(why['current_clearance_m'], .41, atol=.011)
    assert why['required_clearance_m'] == .38 and why['nearest_body_xy_m'][0] < -.4


def test_beside_a_wall_the_retreat_may_keep_the_clearance_it_has():
    p, snapshot = planner_on_a_path(), corridor()
    snapshot.fine_occupied = frozenset((i, -43) for i in range(-300, 300))   # a wall 0.42 m to the right, parallel
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    assert r['vacate']['phase'] == 'retreat' and r['requested_command'] == [-.10, 0., 0.]


def test_the_rear_calves_keep_off_a_wall_corner_behind():
    p, snapshot = planner_on_a_path(), corridor()
    # beside a wall 0.41 m to the left (so the centre may keep 0.38 m), with a wall end behind-left: straight back
    # or swinging the rear left would bring the rear-left calf within 0.12 m of it
    snapshot.fine_occupied = (frozenset((i, 41) for i in range(-300, 300))
                              | frozenset((-40-k, j) for j in range(24, 42) for k in range(3)))
    p.tracks = [track((.6, 0.))]
    for t in np.arange(0., 4.01, .4):
        r = step(p, snapshot, t)
    assert p.vacate_why['rear'] > 0
    assert r['vacate']['phase'] == 'blocked' or r['requested_command'][2] > 0   # if anything, swing the rear right
