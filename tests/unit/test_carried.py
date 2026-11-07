"""Carried outages of visual odometry (perception.carried)."""
from types import SimpleNamespace

import numpy as np
import pytest

from go2nav.perception import carried
from go2nav.perception.rigid import PoseLost
from go2nav.perception.visual_odometry import Pose

STEP_M = .05  # the robot walks +x 5 cm a frame, turning 0.1 rad a frame (gyro)
TURN = .1


def _odometry(fails):
    """A fake visual odometry, the frame read from its time: raises on `fails`; otherwise reports the true motion since
    its own start in its own start frame (the robot walks STEP_M along its heading and turns TURN each frame)."""

    class Fake:
        def __init__(self):
            self.start = None

        def observe(self, *args):
            frame = args[-1]//100_000_000
            if frame in fails:
                raise PoseLost('neither camera measures the current frame')
            if self.start is None:
                self.start = frame
            n = frame-self.start
            p, yaw = np.zeros(3), 0.
            for _ in range(n):
                yaw += TURN
                p += STEP_M*np.array([np.cos(yaw), np.sin(yaw), 0.])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            return Pose(frame, 0, p, R, R, 0, False, None, 1, 'primary', (200, 200))
    return Fake


class Exact:
    """A carrier that knows the motion: STEP_M forward in the axes of the turned heading."""

    def step(self, previous_history, command):
        return np.array([STEP_M*np.cos(TURN), STEP_M*np.sin(TURN)])


def _frame(k):
    rates = np.tile([0., 0., TURN/.1], (51, 1))
    return SimpleNamespace(frame=k, measured_ns=k*100_000_000, rgb=None, depth=None, auxiliary_rgb=None,
                           auxiliary_depth=None, gyro_rates=rates, specific_force=None, history=(), policy=None)


def _run(monkeypatch, fails, frames, max_path_m=10.):
    monkeypatch.setattr(carried, 'VisualOdometry', _odometry(fails))
    monkeypatch.setattr(carried, 'applied_last_frame', lambda packet, now: np.zeros(3))
    odometry = carried.CarriedOdometry(Exact(), max_path_m=max_path_m)
    return odometry, [odometry.observe(_frame(k)) for k in range(frames)]


def _truth(n):
    p, yaw = np.zeros(3), 0.
    for _ in range(n):
        yaw += TURN
        p += STEP_M*np.array([np.cos(yaw), np.sin(yaw), 0.])
    return p, yaw


def test_exact_carrier_keeps_the_pose_through_an_outage_and_takeover(monkeypatch):
    odometry, poses = _run(monkeypatch, fails={6, 7, 8}, frames=20)
    assert [p.camera for p in poses[6:13]] == ['carried']*7 and poses[13].camera == 'primary'
    for k, pose in enumerate(poses):
        p, yaw = _truth(k)
        assert np.allclose(pose.position, p, atol=1e-9), k
        assert np.isclose(np.arctan2(pose.rotation[1, 0], pose.rotation[0, 0]), np.angle(np.exp(1j*yaw)), atol=1e-6), k
        assert pose.frame == k
    o, = odometry.outages
    # lost at 6; fresh starts at 6, 7 and 8 fail, the one started at 9 measures 10-13: takeover at 13
    assert (o['start_frame'], o['end_frame'], o['closed'], o['restarts']) == (6, 13, True, 3)
    assert o['takeover_jump_m'] < 1e-9 and np.isclose(o['path_m'], 8*STEP_M)
    assert odometry.open_outage() is None


def test_the_carried_path_limit_ends_the_mission(monkeypatch):
    with pytest.raises(PoseLost, match='carried'):
        _run(monkeypatch, fails=set(range(5, 40)), frames=40, max_path_m=.3)


def test_a_loss_on_the_first_frame_is_still_terminal(monkeypatch):
    with pytest.raises(PoseLost):
        _run(monkeypatch, fails={0}, frames=2)


def test_the_limit_must_be_set():
    with pytest.raises(ValueError):
        carried.CarriedOdometry(Exact(), max_path_m=None)


def _packet(now_ns, values):
    measured = [now_ns-(14-i)*100_000_000 for i in range(15)]
    return dict(sensor_state=dict(control=dict(applied_command=dict(values=values, measured_ns=measured))))


def test_applied_last_frame_reads_the_command_recorded_at_the_frame():
    values = np.zeros((15, 3))
    values[-1], values[-2] = [.2, 0., .1], [.5, 0., 0.]
    assert np.allclose(carried.applied_last_frame(_packet(10**9, values), 10**9), [.2, 0., .1])


def test_command_and_tape_carriers():
    assert np.allclose(carried.Carrier('command').step((), [.3, 0., 0.]), [.03, 0.])

    class Tapes:
        def forecast_tapes(self, context, tape):
            assert tape.shape == (1, 8, 3) and np.allclose(tape[0], [.1, 0., 0.])
            assert context.history[-1]['image']['measured_ns'] == 5
            out = np.zeros((1, 8, 5), np.float32)
            out[0, :, 0] = .01
            return out
    history = ({'image': {'measured_ns': 5}},)
    assert np.allclose(carried.Carrier('model', Tapes()).step(history, [.1, 0., 0.]), [.01, 0.])
    assert np.allclose(carried.Carrier('c1', c1=Tapes()).step(history, [.1, 0., 0.]), [.01, 0.])
    with pytest.raises(ValueError):
        carried.Carrier('model', object())
