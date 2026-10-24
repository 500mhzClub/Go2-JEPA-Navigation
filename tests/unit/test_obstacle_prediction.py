from types import SimpleNamespace

import numpy as np

from go2nav.perception import tracks
from go2nav.planning import grid, prediction


def arc(centre_m, radius_m=.175, facing=(-1., 0.)):
    """The 1-cm cells of a cylinder's visible half (the face toward `facing`)."""
    angles = np.linspace(-np.pi/2, np.pi/2, 200)+np.arctan2(facing[1], facing[0])
    xy = np.asarray(centre_m)+radius_m*np.stack((np.cos(angles), np.sin(angles)), 1)
    return frozenset(map(tuple, np.floor(xy/.01).astype(int).tolist()))


def coarse(cells):
    return frozenset((i//5, j//5) for i, j in cells)


def snapshot(t_s, cells, floor):
    return SimpleNamespace(measured_ns=int(t_s*1e9), current_fine_occupied=frozenset(cells), floor=frozenset(floor))


def test_tracks_an_obstacle_walking_over_seen_floor_at_its_speed():
    floor = {(i, j) for i in range(0, 80) for j in range(-10, 10)}   # a 4 m x 1 m passage seen as floor
    tracker = tracks.ObstacleTracker()
    for k in range(6):
        t = .4*k
        live = tracker.update(snapshot(t, arc((3.-.3*t, 0.)), floor))
    assert len(live) == 1
    assert np.allclose(live[0].velocity, (-.3, 0.), atol=.02)
    assert tracks.moving(live) == live


def test_wall_cells_never_seen_as_floor_are_not_tracked():
    tracker = tracks.ObstacleTracker()
    for k in range(6):
        live = tracker.update(snapshot(.4*k, arc((3.-.12*k, 0.)), floor=set()))
    assert live == []


def test_long_clusters_are_not_candidates():
    wall = {(i, 0) for i in range(300)}
    assert not tracks.candidate(np.asarray(sorted(wall)), coarse(wall))


def moving_track(centre, velocity, seen_s=0.):
    return tracks.Track(0, arc(centre), int(seen_s*1e9), [(0, np.asarray(centre))], np.asarray(velocity, float))


def test_an_approaching_obstacle_blocks_later_segments_only():
    track = moving_track((2., 0.), (-1., 0.))                      # 1 m/s toward the robot's path
    checked = prediction.PredictedClearance(grid.FineClearance(frozenset()), [track], now_ns=0)
    a, b = np.array([0., 0.]), np.array([.05, 0.])
    assert checked.minimum_during(a, b, 0., .1) > 1.5
    assert checked.minimum_during(a, b, 1.5, 1.6) < .45


def test_an_obstacle_walking_away_leaves_its_stored_cells():
    track = moving_track((.6, 0.), (.3, 0.))                       # face 0.425 m ahead, walking away
    stored = arc((.6, 0.))
    inner = grid.FineClearance(prediction.without(stored, [track]))
    checked = prediction.PredictedClearance(inner, [track], now_ns=0)
    a, b = np.array([0., 0.]), np.array([.03, 0.])
    assert grid.FineClearance(stored).minimum(a, b) < .45+.03     # stored: blocked
    assert checked.minimum_during(a, b, .7, .8) > .45+.03         # predicted: it will have moved on


def test_the_track_age_shifts_the_prediction():
    track = moving_track((2., 0.), (-1., 0.), seen_s=0.)
    young = prediction.PredictedClearance(grid.FineClearance(frozenset()), [track], now_ns=0)
    old = prediction.PredictedClearance(grid.FineClearance(frozenset()), [track], now_ns=int(1e9))
    a, b = np.array([0., 0.]), np.array([.05, 0.])
    assert old.minimum_during(a, b, 0., .1) < young.minimum_during(a, b, 0., .1)-.9
