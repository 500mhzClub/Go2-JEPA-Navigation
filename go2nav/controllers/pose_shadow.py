"""Visual odometry in shadow beside the true pose (docs/dynamic/PLAN.md, P10 step 1): how long outages last.

With the pose oracle on, the controller drives on the true pose and this runs visual odometry on the same frames. When
visual odometry loses the pose (`PoseLost`, a branch that was never ported, or any other error,
logged by name), an **outage** opens: a fresh visual
odometry starts on the next frame, again on every frame it fails, and the outage closes once the current one has
measured `TAKEOVER_FRAMES` frames in a row after its first. Each outage records its frames and times, how far the robot
moved and turned meanwhile (from the truth), the nearest walker at its start, and visual odometry's horizontal error
against the truth at the loss. A fresh visual odometry is placed at the true pose of its first frame, so its error is
its own drift since then. An outage still open when the mission ends is recorded with `closed` False.
"""
import numpy as np

from go2nav.perception.rigid import PoseLost, UnportedBranch
from go2nav.perception.visual_odometry import VisualOdometry

TAKEOVER_FRAMES = 4


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


class VisualShadow:
    """See the module docstring. `observe(frame)` once per sensor frame; `log()` the record."""

    def __init__(self, truth):
        self.truth = truth
        self.odometry, self.origin, self.start_frame = VisualOdometry(), None, None
        self.outage, self.outages, self.streak = None, [], 0
        self.error_m, self.frames, self.last_frame = None, 0, None

    def _walker(self):
        p, R = self.truth.world_pose()
        walkers = self.truth.obstacles()
        if not walkers:
            return None
        body = [(R.T@(np.r_[xy, p[2]]-p))[:2] for xy in walkers]
        nearest = min(body, key=np.linalg.norm)
        return dict(body_xy_m=np.round(nearest, 3).tolist(), distance_m=round(float(np.linalg.norm(nearest)), 3))

    def observe(self, frame):
        self.frames, self.last_frame = self.frames+1, frame
        true = self.truth.registered(frame.frame, frame.measured_ns)
        try:
            pose = self.odometry.observe(frame.rgb, frame.depth, frame.auxiliary_rgb, frame.auxiliary_depth,
                                         frame.gyro_rates, frame.specific_force, frame.measured_ns)
        except Exception as error:  # a shadow never ends the mission: any failure is an outage
            self._lost(frame, true, error if isinstance(error, (PoseLost, UnportedBranch)) else
                       RuntimeError(f'{type(error).__name__}: {error}'))
            return
        if self.origin is None:
            self.origin, self.start_frame = (true.position.copy(), true.rotation.copy()), frame.frame
        p0, R0 = self.origin
        estimate = p0+R0@np.asarray(pose.position, float)
        self.error_m = float(np.linalg.norm(estimate[:2]-true.position[:2]))
        if self.outage is None:
            return
        self._track(true)
        if frame.frame > self.start_frame:
            self.streak += 1
        if self.streak >= TAKEOVER_FRAMES:
            self._close(frame, True)

    def _lost(self, frame, true, error):
        if self.outage is None:
            self.outage = dict(start_frame=frame.frame, start_ns=frame.measured_ns, reason=str(error)[:120],
                               error_at_loss_m=None if self.error_m is None else round(self.error_m, 4),
                               walker=self._walker(), path_m=0., turned_rad=0., restarts=0,
                               _last=(true.position[:2].copy(), _yaw(true.rotation)),
                               _start=(true.position[:2].copy(), _yaw(true.rotation)))
        else:
            self._track(true)
            self.outage['restarts'] += 1
        self.odometry, self.origin, self.start_frame, self.streak, self.error_m = VisualOdometry(), None, None, 0, None

    def _track(self, true):
        xy, yaw = true.position[:2], _yaw(true.rotation)
        last_xy, last_yaw = self.outage['_last']
        self.outage['path_m'] += float(np.linalg.norm(xy-last_xy))
        self.outage['turned_rad'] += abs(float(np.angle(np.exp(1j*(yaw-last_yaw)))))
        self.outage['_last'] = (xy.copy(), yaw)

    def _close(self, frame, closed):
        o = self.outage
        start_xy, start_yaw = o.pop('_start')
        last_xy, last_yaw = o.pop('_last')
        o.update(end_frame=frame.frame, end_ns=frame.measured_ns, closed=closed,
                 duration_s=round((frame.measured_ns-o['start_ns'])/1e9, 2), path_m=round(o['path_m'], 4),
                 turned_rad=round(o['turned_rad'], 4), net_m=round(float(np.linalg.norm(last_xy-start_xy)), 4),
                 net_turn_rad=round(abs(float(np.angle(np.exp(1j*(last_yaw-start_yaw))))), 4))
        self.outages.append(o)
        self.outage, self.streak = None, 0

    def log(self):
        """The record; an outage still open is closed at the last frame seen, as not closed."""
        if self.outage is not None:
            self._close(self.last_frame, False)
        return dict(takeover_frames=TAKEOVER_FRAMES, frames=self.frames, outages=self.outages,
                    final_error_m=None if self.error_m is None else round(self.error_m, 4))
