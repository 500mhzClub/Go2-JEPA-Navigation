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


def test_the_simpler_models_do_not_react_and_stop_at_walls():
    from go2nav.planning import sampling as s
    w, _ = world(walkers=1)
    for _ in range(30):
        w.step((0., 0., 0.))
    snapshot = w.snapshot()
    xy, yaw = np.tile(snapshot['xy'], (2, 300, 1)), np.zeros((2, 300))
    frozen = s.make_model('frozen').walkers(snapshot, xy, yaw, np.full(2, 300), w.geometry)
    assert frozen.shape == (2, 300, 1, 2) and np.allclose(frozen, snapshot['walkers_xy'])
    carried = s.make_model('velocity').walkers(snapshot, xy, yaw, np.full(2, 300), w.geometry)
    speed = np.linalg.norm(snapshot['walkers_velocity'][0])
    assert speed > .1 and np.allclose(np.linalg.norm(carried[0, 9, 0]-snapshot['walkers_xy'][0]), speed)   # 1 s on
    assert w.geometry.wall_distance(carried[0, :, 0]).min() >= moving.RADIUS_M-1e-9      # 30 s on it stands at a wall
    assert np.allclose(carried[0, -1], carried[0, -2])


def test_fresh_dice_keep_the_rules_and_change_the_draws():
    from go2nav.planning import sampling as s
    w, _ = world(walkers=1)
    snapshot = w.snapshot()
    xy, yaw = np.tile(snapshot['xy'], (1, 400, 1)), np.zeros((1, 400))
    true = s.make_model('true').walkers(snapshot, xy, yaw, np.array([400]), w.geometry)
    fresh = s.make_model('fresh', seed=3).walkers(snapshot, xy, yaw, np.array([400]), w.geometry)
    for _ in range(400):
        w.step((0., 0., 0.))
    assert np.linalg.norm(true[0, -1, 0]-w.walkers_xy()[0]) < .3      # the walk replayed (100-ms steps lag a little)
    assert np.linalg.norm(fresh[0, -1, 0]-true[0, -1, 0]) > .5                # other draws, another walk
    assert np.allclose(fresh[0, 5], true[0, 5])                               # the same walk until the first draw


def test_the_mixture_keeps_the_true_future_and_weighs_the_false_ones():
    from go2nav.planning import sampling as s
    w, _ = world(walkers=2)
    for _ in range(30):
        w.step((0., 0., 0.))
    snapshot = w.snapshot()
    xy, yaw = np.tile(snapshot['xy'], (3, 64, 1)), np.zeros((3, 64))
    true = s.make_model('true').walkers(snapshot, xy, yaw, np.full(3, 64), w.geometry)
    model = s.make_model('mix:still=0.1,ignore=0.2')
    paths = model.walkers(snapshot, xy, yaw, np.full(3, 64), w.geometry)
    assert paths.shape == (3, 64, 6, 2) and np.allclose(paths[:, :, :2], true)
    assert np.allclose(paths[:, :, 2:4], snapshot['walkers_xy'])                  # standing where they are
    assert np.allclose(model.weights[0], [.7, .7, .1, .1, .2, .2]) and list(model.groups) == [0, 1]*3
    assert np.allclose(model.ends, true[:, -1])
    noisy = s.make_model('mix:noise=0.03,still=0.1')
    paths = noisy.walkers(snapshot, xy, yaw, np.full(3, 64), w.geometry)
    assert paths.shape == (3, 64, 10, 2) and np.allclose(noisy.weights[0], [.225]*8+[.1]*2)
    assert np.allclose(paths[:, 0, :8], np.tile(true[:, 0], (1, 4, 1)), atol=.01)     # near the truth at first
    assert not np.allclose(paths[:, -1, :2], true[:, -1])                             # off it later
    try:
        s.make_model('mix:still=0.6,ignore=0.4')
        assert False, 'the true future must keep some weight'
    except ValueError:
        pass


def test_the_dice_oracle_draws_new_dice_and_never_the_worlds():
    from go2nav.planning import sampling as s
    w, _ = world(walkers=2)
    for _ in range(30):
        w.step((0., 0., 0.))
    snapshot = w.snapshot()
    xy, yaw = np.tile(snapshot['xy'], (3, 400, 1)), np.zeros((3, 400))
    true = s.make_model('true').walkers(snapshot, xy, yaw, np.full(3, 400), w.geometry)
    model = s.make_model('dice:4', seed=3)
    paths = model.walkers(snapshot, xy, yaw, np.full(3, 400), w.geometry)
    assert paths.shape == (3, 400, 8, 2) and np.allclose(model.weights, .25) and list(model.groups) == [0, 1]*4
    assert np.allclose(paths[:, 5], np.tile(true[:, 5], (1, 4, 1)))               # the same walk until the first draw
    assert all(not np.allclose(paths[:, -1, 2*d:2*d+2], true[:, -1]) for d in range(4))   # then never the world's
    assert not np.allclose(paths[:, -1, 0:2], paths[:, -1, 2:4])                  # nor one another's
    assert np.allclose(paths[0], paths[1]) and np.allclose(model.ends, paths[:, -1, :2])   # the same for every tape


def test_the_worked_out_rule_is_the_true_rule_for_one_future_and_the_expectation_over_joint_futures():
    from itertools import product

    from go2nav.eval import contact_calibration as cc
    from go2nav.planning.sampling import integrate
    w, record = world(walkers=4)
    for _ in range(40):
        w.step((.2, 0., 0.))
    snapshot, target = w.snapshot(), record['beacon_xy_world']
    true = Planner(w.geometry, horizon_s=6.4, terminal='walkers', model='true')
    requested, _ = true.proposals(snapshot, target)
    a, terms_a = true.costs(snapshot, target, requested)
    alone = Planner(w.geometry, horizon_s=6.4, terminal='walkers', model='mix:')        # the true future at weight 1
    d, terms_d = alone.costs(snapshot, target, requested)
    assert np.allclose(a, d) and (terms_a['breached'] == terms_d['breached']).all()
    noisy = Planner(w.geometry, horizon_s=6.4, terminal='static', model='mix:noise=0.05')  # four futures a walker
    xy, yaw = integrate(snapshot, requested)
    walls = noisy.fields.wall_clearance(kin.body_points(xy, yaw)).min(-1)
    walkers = noisy.model.walkers(snapshot, xy, yaw, np.full(len(requested), noisy.frames), w.geometry)
    weights, groups = noisy.model.weights, noisy.model.groups
    worked = cc.split(*noisy.expected_costs(snapshot, target, xy, yaw, walls, walkers, weights, groups))
    ones, ids = np.ones((len(requested), len(np.unique(groups)))), np.arange(len(np.unique(groups)))
    mean = 0.
    for pick in product(*(np.flatnonzero(groups == g) for g in np.unique(groups))):   # each joint future alone
        one = noisy.expected_costs(snapshot, target, xy, yaw, walls, walkers[:, :, list(pick)], ones, ids)
        mean = mean+np.prod(weights[0, list(pick)])*cc.split(*one)
    assert len(walkers[0, 0]) == 16 and np.isclose(np.prod(weights[0, :4]), 1/256)
    assert np.allclose(worked[[0, 2, 3]], mean[[0, 2, 3]])  # time, progress, the breach: the expectation exactly
    assert np.corrcoef(worked[1], mean[1])[0, 1] > .9        # clearance from the expected nearness: near it
    for name in ('true', 'velocity'):                          # a model of one future goes through the worked-out rule
        usual, worked = (Planner(w.geometry, horizon_s=6.4, terminal='walkers', model=name) for _ in range(2))
        worked.exact = True
        g, terms_g = usual.costs(snapshot, target, requested)
        h, terms_h = worked.costs(snapshot, target, requested)
        assert np.allclose(g, h) and (terms_g['breached'] == terms_h['breached']).all()


def test_the_contact_probe_reads_the_planners_own_futures_and_leaves_the_mission_as_it_was():
    from go2nav.eval import contact_calibration as cc
    kwargs = dict(walkers=4, kind='rude', horizon_s=6.4, terminal='walkers', model='mix:noise=0.03', frames=150)
    plain = km.run_mission('validation', 12, 0, **kwargs)
    probe = cc.Probe(near=50., every=10, limit=2)
    probed = km.run_mission('validation', 12, 0, probe=probe, **kwargs)
    timing = ('plan_s', 'wall_s')
    assert {k: v for k, v in plain.items() if k not in timing} == {k: v for k, v in probed.items() if k not in timing}
    assert len(probe.rows) == 2
    for p in probe.rows:
        assert len(p['chance']) == len(p['truth']) == len(p['true_cost']) == len(p['kinds']) > 10
        assert all(0. <= c <= 1. for c in p['chance']) and p['chosen'] >= 0
        assert p['true_cost'][p['chosen']] >= min(p['true_cost'])
        assert np.allclose(np.sum(p['terms'], 0), p['cost'], atol=.05)                 # the terms make up the cost
        assert np.allclose(np.sum(p['true_terms'], 0), p['true_cost'], atol=.05)
    report = cc.summary([dict(probed, probes=probe.rows)])
    assert report['probes'] == 2 and sum(b['tapes'] for b in report['bands']) == report['tapes']


def test_a_ring_sees_only_what_is_in_its_line_of_sight():
    from go2nav.planning import sampling as s
    w, record = world(walkers=1)
    model, snapshot = s.make_model('frozen_seen'), w.snapshot()
    hidden = not w.geometry.visible(snapshot['xy'], snapshot['walkers_xy'][0])
    at, _ = model.believed(snapshot, w.geometry)
    assert hidden and len(at) == 0                                            # it starts three passages away, unseen
    snapshot['walkers_xy'] = (snapshot['xy']+np.array([np.cos(snapshot['yaw']), np.sin(snapshot['yaw'])])*.9)[None]
    at, _ = model.believed(snapshot, w.geometry)
    assert len(at) == 1                                                       # in sight: believed where it is
    seen = at[0].copy()
    snapshot['walkers_xy'] = w.walkers_xy()                                   # gone from sight; its place is in sight
    at, _ = model.believed(snapshot, w.geometry)
    assert len(at) == 0 and w.geometry.visible(snapshot['xy'], seen)


def test_the_slow_robot_asks_for_no_more_than_its_speed():
    w, record = world()
    planner = Planner(w.geometry, horizon_s=3.2, max_speed=.2, cells=False)
    requested, names = planner.proposals(w.snapshot(), np.asarray(record['beacon_xy_world']))
    assert requested[..., 0].max() <= .2+1e-9 and not any(n.startswith('cell') for n in names)
