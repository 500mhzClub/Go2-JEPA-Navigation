"""The range sensor's returns in the hand perception (docs/dynamic/PLAN.md P13 item 2): occupied cells, never floor."""
import numpy as np

from go2nav.perception import mapping, obstacles
from go2nav.perception import range as beams


def mapped(monkeypatch, range_points):
    """One map update at the identity pose, the floor at 0, the cameras seeing nothing."""
    routing_map = mapping.RoutingMap()
    routing_map.B, routing_map.floor_height = np.eye(3), 0.
    covered = np.zeros(len(mapping.GRID), bool)
    monkeypatch.setattr(mapping, 'floor_coverage', lambda *args, **kwargs: covered)
    monkeypatch.setattr(mapping, 'sampled_points', lambda *args, **kwargs: np.zeros((0, 3)))
    monkeypatch.setattr(mapping.floor, 'paired_floor_points', lambda *args, **kwargs: ((None, None),))
    monkeypatch.setattr(mapping.floor, 'joint_plane', lambda *args, **kwargs: dict(available=True))
    return routing_map.update(4, 0, np.zeros(3), np.eye(3), (None, None), (None, None), None, range_points=range_points)


def test_returns_mark_occupied_cells_behind_the_robot_but_no_floor(monkeypatch):
    behind, floor_return = (-1.025, .025, .42), (2.025, .025, .01)
    snapshot = mapped(monkeypatch, np.array([behind, floor_return]))
    assert (-21, 0) in snapshot.occupied and (-103, 2) in snapshot.fine_occupied
    assert (40, 0) not in snapshot.occupied                        # under the 3-cm band: a floor return
    assert not snapshot.floor and not snapshot.current_floor and (-21, 0) in snapshot.current_occupied


def test_without_returns_the_map_is_unchanged(monkeypatch):
    assert not mapped(monkeypatch, None).occupied
    assert not mapped(monkeypatch, np.zeros((0, 3))).occupied


def observed(monkeypatch, range_points):
    """One obstacle frame, the floor 0.30 m below the body origin, the cameras seeing only floor."""
    observer = obstacles.DepthObstacleObserver()
    clouds = iter([np.zeros((0, 3)), np.array([[1., 0., -.30]])])  # primary: nothing; auxiliary: one floor sample
    monkeypatch.setattr(obstacles, 'sampled_points', lambda *args, **kwargs: next(clouds))
    monkeypatch.setattr(obstacles.floor, 'paired_floor_points', lambda *args, **kwargs: ((None, None), None))
    plane = dict(available=True, normal_body=np.array([0., 0., 1.]), offset_body_m=.30)
    monkeypatch.setattr(obstacles.floor, 'joint_plane', lambda *args, **kwargs: plane)
    monkeypatch.setattr(obstacles.floor, 'gyro_height_plane', lambda *args, **kwargs: plane)
    return observer.observe((None, None), (None, None), np.zeros((51, 3)), np.tile([0., 0., 9.81], (20, 1)), 0,
                            range_points=range_points)


def test_the_dispatch_check_sees_a_walker_beside_and_behind(monkeypatch):
    ranges = np.zeros(beams.BEAMS, np.float32)
    valid = np.zeros(beams.BEAMS, bool)
    left = int(np.argmin(np.abs(beams.AZIMUTHS-np.pi/2)))
    ranges[left], valid[left] = .40, True                          # a walker's face 0.40 m to the left of the mount
    ranges[0], valid[0] = .62, True                                # and one behind, 0.32 m behind the base origin
    points = beams.body_points(ranges, valid)                      # (0.30, 0.40) and (-0.32, 0) m, at the mount's height
    found = observed(monkeypatch, points)
    assert found.cells == {tuple(np.floor(p[:2]/obstacles.CELL_M).astype(int)) for p in points}
    assert found.returns == (0, 1)                                 # the cameras' counts are unchanged
    assert not observed(monkeypatch, None).cells                   # without the sensor: only the floor, no obstacles
