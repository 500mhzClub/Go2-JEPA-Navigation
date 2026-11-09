"""The kinematic world, the distance fields and the sampling planner (docs/dynamic/PLAN.md P17)."""
import math

import numpy as np

from go2nav.eval import kinematic_missions as km
from go2nav.eval.mission import episode_spec
from go2nav.planning.fields import Fields
from go2nav.planning.sampling import FLOOR_M, Planner
from go2nav.sim import kinematic as kin
from go2nav.sim import obstacles as moving


def world(walkers=0, kind='rude', maze=12, episode=0):
    spec, record = episode_spec('validation', maze, episode)
    placement = km.placement_for('validation', maze, episode, spec, record, walkers, kind)
    return kin.World(spec, record['home_se2_world'], placement), record


def test_the_gap_to_the_body_is_measured_from_its_rectangle():
    points = np.array([[.35+.2, 0.], [0., .16+.1], [.35+.3, .16+.4], [0., 0.]])
    gap = kin.rectangle_gap(points, np.zeros(2), 0.)
    assert np.allclose(gap, [.2, .1, .5, -.16])
    assert np.isclose(kin.rectangle_gap(np.array([0., .55]), np.zeros(2), math.pi/2), .2)     # turned a quarter
    assert np.abs(np.diff(kin.OUTLINE[np.argsort(np.arctan2(*kin.OUTLINE.T))], axis=0)).max() <= .17


def test_the_robot_moves_as_commanded_through_the_rate_limit():
    w, record = world()
    start, yaw = w.xy.copy(), w.yaw
    executed = [w.step((.45, 0., 0.)).copy() for _ in range(3)]
    assert np.allclose([e[0] for e in executed], [.25, .45, .45])               # 0.25 m/s a frame, to the bound
    along = (w.xy-start)@np.array([math.cos(yaw), math.sin(yaw)])
    assert abs(along-.115) < 1e-6 and w.frame == 3 and w.contact is None
    for _ in range(100):
        w.step((.45, 0., 0.))
    assert w.contact == 'wall' and w.minimum_clearance <= 0.                    # it walked into a wall and stopped
    frame = w.frame
    w.step((.45, 0., 0.))
    assert w.frame == frame


def test_walkers_walk_as_in_a_mission_and_a_snapshot_is_a_copy():
    w, _ = world(walkers=2)
    assert len(w.walkers) == 2 and not w.walkers[0].polite
    snapshot = w.snapshot()
    before = [v.xy.copy() for v in snapshot['walkers']]
    for _ in range(100):
        w.step((0., 0., 0.))
    assert np.linalg.norm(w.walkers_xy()-np.array(before), axis=1).max() > .5   # they walked on for 10 s
    assert all(np.array_equal(v.xy, b) for v, b in zip(snapshot['walkers'], before))
    assert np.allclose(w.xy, snapshot['xy'])                                     # a zero command holds the robot


def test_the_distance_field_follows_the_passages_and_a_walker_seals_one():
    w, record = world()
    fields = Fields(w.geometry)
    home, beacon = np.array(record['home_se2_world'][:2]), np.array(record['beacon_xy_world'])
    distance = float(fields.value(fields.static(beacon), home[None])[0])
    assert .8*record['shortest_outbound_m'] < distance < 1.1*record['shortest_outbound_m']
    cell = moving.cell_of(home, w.geometry.pitch)
    hops = moving.hops_from(w.geometry.neighbours, cell)
    towards = moving.hops_from(w.geometry.neighbours, moving.cell_of(beacon, w.geometry.pitch))
    step = min((n for n in w.geometry.neighbours[cell]), key=lambda n: towards[n])         # the next cell on the way
    blocked = fields.with_walkers(beacon, [(np.asarray(cell)+np.asarray(step))*w.geometry.pitch/2])
    assert not np.isfinite(fields.value(blocked, home[None])[0]) and hops[step] == 1
    assert np.isfinite(fields.value(fields.with_walkers(beacon, [beacon+50.]), home[None])[0])
    assert abs(fields.wall_clearance(home[None])[0]-w.geometry.wall_distance(home[None])[0]) < .01


def test_the_planner_makes_a_round_trip_in_a_still_maze():
    row = km.run_mission('validation', 12, 0, walkers=0, kind='none', horizon_s=3.2, terminal='static')
    assert row['exit'] == 'round_trip' and row['beacon'] and row['minimum_clearance_m'] > FLOOR_M-.03


def test_a_walker_that_does_not_yield_stops_a_short_sighted_planner_and_not_one_that_counts_it():
    stuck = km.run_mission('validation', 11, 1, walkers=1, kind='rude', horizon_s=1.6, terminal='static', frames=900)
    assert stuck['exit'] == 'timeout' and stuck['waiting_s'] > 60 and stuck['chosen'].get('hold', 0) > 100
    through = km.run_mission('validation', 11, 1, walkers=1, kind='rude', horizon_s=6.4, terminal='walkers')
    assert through['exit'] == 'round_trip' and through['waiting_s'] < 10


def test_the_plan_costs_a_tape_into_a_wall_out():
    w, record = world()
    planner = Planner(w.geometry, horizon_s=3.2)
    snapshot = w.snapshot()
    ahead = np.tile([.45, 0., 0.], (1, planner.frames, 1))
    total, terms = planner.costs(snapshot, record['beacon_xy_world'], np.concatenate((ahead, 0*ahead)))
    assert terms['breached'][0] or terms['clearance'][0] > 0 or total[0] < total[1]       # walking on is costed
    assert not terms['breached'][1] and terms['time'][1] == 3.2                           # holding costs its time
