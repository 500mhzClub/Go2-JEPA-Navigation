"""The shadow visual odometry's outage log (controllers.pose_shadow)."""
from types import SimpleNamespace

import numpy as np

from go2nav.controllers import pose_shadow
from go2nav.perception.rigid import PoseLost


class FakeTruth:
    """The robot walks +x at 0.1 m per frame; one walker 0.55 m ahead."""

    def __init__(self):
        self.x = 0.

    def registered(self, frame, measured_ns):
        return SimpleNamespace(position=np.array([self.x, 0., 0.]), rotation=np.eye(3))

    def world_pose(self):
        return np.array([self.x, 0., .3]), np.eye(3)

    def obstacles(self):
        return [np.array([self.x+.55, 0.])]


def _odometry(fails):
    """A fake visual odometry: frames counted globally; raises on `fails`, else reports motion since its own start."""
    counter = dict(frame=-1)

    class Fake:
        def __init__(self):
            self.start = None

        def observe(self, *args):
            counter['frame'] += 1
            if counter['frame'] in fails:
                raise PoseLost('neither camera measures the current frame')
            if self.start is None:
                self.start = counter['frame']
            return SimpleNamespace(position=np.array([.1*(counter['frame']-self.start), 0., 0.]))
    return Fake


def _run(monkeypatch, fails, frames):
    monkeypatch.setattr(pose_shadow, 'VisualOdometry', _odometry(fails))
    truth = FakeTruth()
    shadow = pose_shadow.VisualShadow(truth)
    for k in range(frames):
        shadow.observe(SimpleNamespace(frame=k, measured_ns=k*100_000_000, rgb=None, depth=None, auxiliary_rgb=None,
                                       auxiliary_depth=None, gyro_rates=None, specific_force=None))
        truth.x += .1
    return shadow.log()


def test_outage_closes_after_four_measured_frames(monkeypatch):
    log = _run(monkeypatch, fails={5, 6, 7}, frames=20)
    assert len(log['outages']) == 1
    o = log['outages'][0]
    # lost at 5, 6 and 7; a fresh start at 8, measured at 9-12: closed at 12
    assert (o['start_frame'], o['end_frame'], o['closed'], o['restarts']) == (5, 12, True, 2)
    assert abs(o['duration_s']-.7) < 1e-9 and abs(o['path_m']-.7) < 1e-6 and abs(o['net_m']-.7) < 1e-6
    assert o['walker']['distance_m'] == .55 and o['error_at_loss_m'] == 0.
    assert log['final_error_m'] < 1e-9


def test_failing_fresh_start_restarts_and_open_outage_is_kept(monkeypatch):
    log = _run(monkeypatch, fails={3, 5, 6}, frames=8)
    o, = log['outages']
    assert (o['start_frame'], o['closed'], o['restarts'], o['end_frame']) == (3, False, 2, 7)
