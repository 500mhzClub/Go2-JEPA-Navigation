"""Module oracles (controllers.truth): frames, the obstacle pixel mask, the map oracle's cells and true futures."""
from types import SimpleNamespace

import numpy as np

from go2nav.controllers import truth
from go2nav.perception import cameras
from go2nav.perception.mapping import CELL_M, FINE_CELL_M, MapSnapshot
from go2nav.planning import grid, prediction
from go2nav.sim.obstacles import PITCH_M, RADIUS_M, Walker


def pose(x, y, yaw):
    return [x, y, .3, 0., 0., np.sin(yaw/2), np.cos(yaw/2)]


def session(start=pose(1., 2., .5), walkers=()):
    s = SimpleNamespace(samples=[dict(base_pose_world=start)], base_xy=None,
                        spec=dict(geometry=dict(wall_boxes=[dict(centre_xyz=[x, 0., .7], size_xyz=[.08, 1.38, 1.4],
                                                                 yaw_rad=0.) for x in (0., 2.6)])),
                        obstacles=SimpleNamespace(walkers=list(walkers)) if walkers else None)
    return s


def snapshot(B=np.eye(3)):
    empty = frozenset()
    return MapSnapshot(0, 0, empty, empty, (0., 0., 0.), tuple(map(tuple, B)), 0., 0, 0, empty, empty, empty, empty)


def test_the_true_pose_is_taken_in_the_settled_frame():
    s = session()
    t = truth.GroundTruth(s)
    s.samples.append(dict(base_pose_world=pose(1.+np.cos(.5), 2.+np.sin(.5), .5+.3)))  # 1 m ahead, turned 0.3 rad
    r = t.registered(4, 400)
    np.testing.assert_allclose(r.position, [1., 0., 0.], atol=1e-9)
    np.testing.assert_allclose(np.arctan2(r.rotation[1, 0], r.rotation[0, 0]), .3, atol=1e-9)


def test_truth_is_placed_through_the_estimated_pose():
    s = session(start=pose(0., 0., 0.))
    t = truth.GroundTruth(s)
    # the robot believes it is 0.1 m further along x than it is; a point 1 m ahead of it lands 1.1 m out
    A, b = t.placed(np.array([.1, 0., 0.]), np.eye(3))
    np.testing.assert_allclose(A@np.array([1., 0., .3])+b, [1.1, 0., 0.], atol=1e-9)  # the base stands 0.3 m up


def test_obstacle_pixels_are_those_on_the_cylinder():
    w = Walker({(1, 0): [(0, 0)], (0, 0): [(1, 0)]}, (1, 0), .3, np.random.default_rng(0))
    s = session(start=pose(0., 0., 0.), walkers=[w])
    t = truth.GroundTruth(s)
    # the primary camera looks along body x from 0.326 m ahead; depth to the cylinder's near face on the optical axis
    depth = np.full((cameras.HEIGHT, cameras.WIDTH), 5., np.float32)
    near = PITCH_M-RADIUS_M-cameras.PRIMARY[0, 3]
    depth[230:250, 310:330] = near
    hit = t.obstacle_pixels(depth, np.ones_like(depth, bool), cameras.PRIMARY)
    assert hit[240, 320] and not hit[0, 0] and hit.sum() == 400


def test_the_obstacles_oracle_adds_true_cells_and_stores_none():
    w = Walker({(1, 0): [(0, 0)], (0, 0): [(1, 0)]}, (1, 0), .3, np.random.default_rng(0))
    s = session(start=pose(0., 0., 0.), walkers=[w])
    oracle = truth.MapOracle(truth.GroundTruth(s), 'obstacles')
    out = oracle.apply(snapshot(), np.zeros(3), np.eye(3))
    centre = (int(np.floor(PITCH_M/CELL_M)), 0)
    assert centre in out.occupied and centre in out.current_occupied
    assert (int(np.floor(PITCH_M/FINE_CELL_M)), 0) in out.fine_occupied
    assert not out.floor


def test_the_full_oracle_gives_walls_and_floor():
    s = session(start=pose(.5, 0., 0.))
    out = truth.MapOracle(truth.GroundTruth(s), 'full').apply(snapshot(), np.zeros(3), np.eye(3))
    assert (-10, 0) in out.occupied       # the wall at x = 0 sits 0.5 m behind the robot
    assert out.floor and not out.floor & out.occupied


def test_oracle_tracks_follow_the_walk_and_hold_their_cells_out_of_the_stored_ones():
    w = Walker({(1, 0): [(2, 0)], (2, 0): [(1, 0)]}, (1, 0), .3, np.random.default_rng(0))
    w.pause_left = 0.
    s = session(start=pose(0., 0., 0.), walkers=[w])
    [track] = truth.oracle_tracks(truth.GroundTruth(s), snapshot(), np.zeros(3), np.eye(3))
    np.testing.assert_allclose(track.displacement(1.), [.3, 0.], atol=1e-6)
    assert track.cells < track.stored_cells
    assert w.xy[0] == PITCH_M  # the field itself is not advanced
    stored = prediction.without(track.stored_cells|{(0, 0)}, [track])
    assert stored == {(0, 0)}


def test_constant_velocity_tracks_are_checked_as_before():
    cells = frozenset((i, j) for i in range(100, 110) for j in range(-5, 5))
    track = SimpleNamespace(cells=cells, velocity=np.array([.5, 0.]), seen_ns=0)
    checked = prediction.PredictedClearance(grid.FineClearance(frozenset()), [track], now_ns=200_000_000)
    expected = grid.FineClearance(cells).minimum(np.array([0., 0.])-np.array([.5, 0.])*(.2+.4),
                                                 np.array([.3, 0.])-np.array([.5, 0.])*(.2+.8))
    assert checked.minimum_during(np.array([0., 0.]), np.array([.3, 0.]), .4, .8) == expected


def test_the_static_oracle_keeps_obstacles_out_of_the_route_grid():
    w = Walker({(1, 0): [(0, 0)], (0, 0): [(1, 0)]}, (1, 0), .3, np.random.default_rng(0))
    s = session(start=pose(.5, 0., 0.), walkers=[w])
    out = truth.MapOracle(truth.GroundTruth(s), 'static').apply(snapshot(), np.zeros(3), np.eye(3))
    # the obstacle stands 1.3 - 0.5 = 0.8 m ahead of the robot
    assert (int(np.floor(.8/FINE_CELL_M)), 0) in out.fine_occupied
    assert (int(np.floor(.8/CELL_M)), 0) not in out.occupied and (int(np.floor(.8/CELL_M)), 0) in out.floor
