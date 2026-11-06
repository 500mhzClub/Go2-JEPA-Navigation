"""The world model's walker tracker: placing detections in the map, association, confirmation and velocity."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2nav.controllers import walkers
from go2nav.controllers.controller import NavigationController
from go2nav.models.lewm import LeWMConfig, LeWorldModel, PacketEncoder
from go2nav.perception.mapping import FINE_CELL_M
from go2nav.perception.tracks import KEEP_UNSEEN_S, moving
from go2nav.sim.obstacles import RADIUS_M

STEP_NS = 100_000_000


def yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def seen(*points):
    """One frame's rows for `update`: a detection per point in the columns, the last column empty."""
    return [[.9, *p] for p in points]+[[.1, 9., 9.]]


def test_a_detection_lands_where_the_pose_and_map_orientation_put_it():
    tracker = walkers.ModelTracker()
    position, rotation = np.array([1., 2., 0.]), yaw(math.pi/2)
    tracks = tracker.update([seen((1., 0.))], [0], [(position, rotation)], yaw(0.))
    assert len(tracks) == 1
    assert np.allclose(tracks[0].history[-1][1], [1., 3.])
    centre = np.array(sorted(tracks[0].cells)).mean(0)*FINE_CELL_M
    assert np.linalg.norm(centre-[1., 3.]) < .02


def test_the_map_orientation_turns_the_placement():
    tracker = walkers.ModelTracker()
    tracks = tracker.update([seen((1., 0.))], [0], [(np.zeros(3), yaw(0.))], yaw(math.pi/2))
    assert np.allclose(tracks[0].history[-1][1], [0., 1.], atol=1e-9)


def test_a_track_is_confirmed_by_repeated_detections_and_a_low_presence_is_none():
    tracker = walkers.ModelTracker()
    pose = (np.zeros(3), yaw(0.))
    assert not tracker.update([[[.4, 1., 0.]]], [0], [pose], yaw(0.))
    for k in range(walkers.CONFIRM_DETECTIONS):
        tracks = tracker.update([seen((1., 0.))], [k*STEP_NS], [pose], yaw(0.))
        assert tracks[0].confirmed == (k == walkers.CONFIRM_DETECTIONS-1)
    assert len(tracks) == 1


def test_a_walking_detection_gets_its_velocity_and_a_second_walker_its_own_track():
    tracker = walkers.ModelTracker()
    pose = (np.zeros(3), yaw(0.))
    frames = 12
    detected = [seen((1.+.3*k*.1, 0.), (-2., 1.)) for k in range(frames)]
    tracks = tracker.update(detected, [k*STEP_NS for k in range(frames)], [pose]*frames, yaw(0.))
    assert len(tracks) == 2
    assert np.allclose(tracks[0].velocity, [.3, 0.], atol=1e-6)
    assert tracks[0].speed == moving(tracks)[0].speed and len(moving(tracks)) == 1


def test_a_frame_without_a_pose_is_skipped_and_an_unseen_track_is_dropped():
    tracker = walkers.ModelTracker()
    pose = (np.zeros(3), yaw(0.))
    assert not tracker.update([seen((1., 0.))], [0], [None], yaw(0.))
    assert tracker.update([seen((1., 0.))], [0], [pose], yaw(0.))
    later = int((KEEP_UNSEEN_S+.1)*1e9)
    assert not tracker.update([[[.1, 0., 0.]]], [later], [pose], yaw(0.))


def test_the_track_cells_fit_the_planner():
    tracker = walkers.ModelTracker()
    track = tracker.update([seen((1., 0.))], [0], [(np.zeros(3), yaw(0.))], yaw(0.))[0]
    assert track.cells < track.stored_cells
    assert abs(len(track.cells)-math.pi*(RADIUS_M/FINE_CELL_M)**2) < .1*len(track.cells)
    assert track.footprint <= track.trail
    assert set(track.summary(0)) >= {'id', 'centroid_m', 'velocity_m_s', 'motion_updates'}


def test_the_finder_needs_the_mover_head():
    try:
        walkers.WalkerFinder(LeWorldModel(LeWMConfig()), device='cpu')
    except ValueError as error:
        assert 'mover head' in str(error)
    else:
        raise AssertionError('a model without the mover head was accepted')


def test_one_walker_in_two_columns_is_one_track_at_their_mean():
    tracker = walkers.ModelTracker()
    tracks = tracker.update([seen((1., 0.), (1.2, 0.))], [0], [(np.zeros(3), yaw(0.))], yaw(0.))
    assert len(tracks) == 1
    assert np.allclose(tracks[0].history[-1][1], [1.1, 0.])


def test_a_jump_inside_the_gate_keeps_the_track_and_its_detections():
    tracker = walkers.ModelTracker()
    pose = (np.zeros(3), yaw(0.))
    tracks = tracker.update([seen((1., 0.)), seen((1.2, 0.))], [0, STEP_NS], [pose]*2, yaw(0.))
    assert len(tracks) == 1 and tracks[0].detections == 2


def test_a_held_track_keeps_its_fit_for_the_logs_and_gives_the_planner_no_velocity():
    tracker = walkers.ModelTracker(held=True)
    pose = (np.zeros(3), yaw(0.))
    frames = 12
    detected = [seen((1.+.3*k*.1, 0.)) for k in range(frames)]
    tracks = tracker.update(detected, [k*STEP_NS for k in range(frames)], [pose]*frames, yaw(0.))
    assert np.allclose(tracks[0].fitted, [.3, 0.], atol=1e-6)
    assert np.all(tracks[0].velocity == 0) and tracks[0].confirmed and not moving(tracks)


class CountingModel:
    """Stands in for a world model: counts encodings."""
    config = SimpleNamespace(mover_head=True)

    def __init__(self):
        self.encodings = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, pixels):
        self.encodings += 1
        return torch.zeros(1, 4, 13, 8)

    def movers(self, state):
        return torch.zeros(4, 4, 3)


def packets(start_ns):
    frame = np.zeros((480, 640, 3), np.uint8)
    return tuple(dict(image=dict(rgb=frame, measured_ns=start_ns+k*STEP_NS)) for k in range(4))


def test_the_walkers_and_the_forecast_read_one_encoding_per_set_of_frames():
    model = CountingModel()
    encoder = PacketEncoder(model, 'cpu')
    finder = walkers.WalkerFinder(model, device='cpu', encoder=encoder)
    finder.detect(packets(0))
    encoder(packets(0))
    assert model.encodings == 1
    finder.detect(packets(400_000_000))
    assert model.encodings == 2


def test_the_hand_tracker_stops_when_the_model_drives():
    def controller(mode):
        return NavigationController(None, goal_xy=(1., 0.), home_xy=(0., 0.), obstacle_prediction=True, mover_trails=True,
                                    walker_finder=object(), walker_mode=mode)
    assert controller('held').tracker is None
    assert controller('live').tracker is None
    assert controller('shadow').tracker is not None


def test_true_positions_replace_the_hand_tracker_and_need_the_ground_truth():
    options = dict(goal_xy=(1., 0.), home_xy=(0., 0.), obstacle_prediction=True, mover_trails=True, walker_mode='truth')
    controller = NavigationController(None, truth=object(), **options)
    assert controller.tracker is None and controller.model_tracker.held
    with pytest.raises(ValueError):
        NavigationController(None, **options)
