"""Visual odometry whose outages are carried by a motion model (docs/dynamic/PLAN.md, P10).

`CarriedOdometry.observe(frame)` returns each frame's visual `Pose` in the original visual-odometry frame, as
`VisualOdometry.observe` does, but a frame that visual odometry cannot measure (`PoseLost`) opens a **carried outage**
instead of ending the mission:

- **Carried pose.** Each outage frame's position is the previous one moved by the carrier's forecast of the commands
  applied over the last 100 ms (from the previous frame's four packets, in the previous frame's body axes); its
  rotation is the previous one turned by the gyro over the same 100 ms. The carriers (`Carrier`): 'model', the
  mission's own forecaster (the world model in C5; `forecast_tapes`); 'c1', the command-history model (T6); 'command',
  the commands integrated as if executed exactly.
- **Takeover.** On every outage frame a fresh visual odometry is started (placed at that frame's carried pose) unless
  one is already running; one that fails is dropped and another starts on the next frame. Once the running one has
  measured `TAKEOVER_FRAMES` frames after its first, its poses, composed with the carried pose at its start, replace
  the carrier, and the outage closes.
- **Limit.** If the carried path during one outage grows beyond `max_path_m`, `PoseLost` is raised as before.

Carried frames are returned with `camera` 'carried' and no corners (`feature_counts` (0, 0)); the controller skips the
visual support for them. `outages` records each outage (frames, carried path, takeover jump, restarts).
"""
from dataclasses import replace

import numpy as np

from go2nav.models.command import nominal
from go2nav.perception.rigid import PoseLost, UnportedBranch
from go2nav.perception.rotations import GyroAttitude
from go2nav.perception.visual_odometry import Pose, VisualOdometry

TAKEOVER_FRAMES = 4
FRAME_NS = 100_000_000
CARRIERS = ('model', 'c1', 'command')
MAX_CARRIED_PATH_M = .05  # P10 step 2: C1's 90th-percentile error reaches 5 cm beyond this carried path (STATUS, P10)


def applied_last_frame(packet, now_ns):
    """The mean applied command [vx, vy, yaw rate] over the 100 ms before `now_ns`, from a policy packet."""
    record = packet['sensor_state']['control']['applied_command']
    values, measured = np.asarray(record['values'], float), np.asarray(record['measured_ns'])
    recent = (measured > now_ns-FRAME_NS) & (measured <= now_ns)
    return values[recent].mean(0) if recent.any() else values[-1]


class Carrier:
    """The body-frame motion (dx, dy) over the last 100 ms, from the previous frame's history and the applied commands."""

    def __init__(self, kind, forecaster=None, c1=None):
        if kind not in CARRIERS:
            raise ValueError(f'carrier must be one of {CARRIERS}')
        if kind == 'model' and not hasattr(forecaster, 'forecast_tapes'):
            raise ValueError("the 'model' carrier needs a forecaster with forecast_tapes")
        if kind == 'c1' and c1 is None:
            from go2nav import artifacts
            from go2nav.controllers.forecasters import CommandHistoryForecaster
            c1 = CommandHistoryForecaster(artifacts.path('t6_c1_command_model'))
        self.kind, self.forecaster = kind, forecaster if kind == 'model' else c1

    def step(self, previous_history, command):
        tape = np.broadcast_to(np.asarray(command, np.float32), (1, 8, 3)).copy()
        if self.kind == 'command':
            return nominal(tape[0])[0, :2]
        from go2nav.controllers.controller import PlanContext
        context = PlanContext(-1, previous_history[-1]['image']['measured_ns'], tuple(previous_history), (), {}, ())
        return np.asarray(self.forecaster.forecast_tapes(context, tape), float)[0, 0, :2]


class CarriedOdometry:
    """See the module docstring."""

    def __init__(self, carrier, *, max_path_m):
        if max_path_m is None:
            raise ValueError('the carried path limit is set from P10 step 2 (MAX_CARRIED_PATH_M)')
        self.carrier, self.max_path_m = carrier, float(max_path_m)
        self.odometry, self.offset = VisualOdometry(), (np.zeros(3), np.eye(3))  # current odometry's start, placed
        self.gyro, self.last_gyro = GyroAttitude(), None
        self.last, self.previous_history = None, None
        self.outage, self.outages = None, []
        self.fresh = None   # (VisualOdometry, its start pose placed, start frame, measured frames)

    def _placed(self, pose, offset, frame):
        p0, R0 = offset
        return replace(pose, frame=frame.frame, position=p0+R0@np.asarray(pose.position, float),
                       rotation=R0@np.asarray(pose.rotation, float))

    def _carried(self, frame, gyro):
        dx, dy = self.carrier.step(self.previous_history, applied_last_frame(frame.policy, frame.measured_ns))
        R = np.asarray(self.last.rotation, float)
        position = np.asarray(self.last.position, float)+R@np.array([dx, dy, 0.])
        rotation = R@(self.last_gyro.T@gyro)
        self.outage['path_m'] += float(np.hypot(dx, dy))
        return Pose(frame.frame, frame.measured_ns, position, rotation, gyro, -1, False, None, 0, 'carried', (0, 0))

    def observe(self, frame):
        gyro = self.gyro.update(frame.gyro_rates).copy()
        args = (frame.rgb, frame.depth, frame.auxiliary_rgb, frame.auxiliary_depth, frame.gyro_rates,
                frame.specific_force, frame.measured_ns)
        try:
            if self.outage is None:
                try:
                    pose = self._placed(self.odometry.observe(*args), self.offset, frame)
                except PoseLost as error:
                    if self.last is None:
                        raise
                    self.outage = dict(start_frame=frame.frame, start_ns=frame.measured_ns, reason=str(error)[:120],
                                       path_m=0., restarts=0)
                    pose = self._outage_frame(frame, gyro, args)
            else:
                pose = self._outage_frame(frame, gyro, args)
        finally:
            self.previous_history, self.last_gyro = frame.history, gyro
        self.last = pose
        return pose

    def _outage_frame(self, frame, gyro, args):
        carried = self._carried(frame, gyro)
        if self.outage['path_m'] > self.max_path_m:
            self.outages.append(self.outage | dict(end_frame=frame.frame, closed=False))
            raise PoseLost(f"measured visual pose unavailable (tracking): carried {self.outage['path_m']:.2f} m "
                           f"beyond {self.max_path_m:.2f} m")
        if self.fresh is None:
            self.fresh = [VisualOdometry(), (carried.position.copy(), carried.rotation.copy()), frame.frame, 0]
        odometry, offset, start, measured = self.fresh
        try:
            pose = odometry.observe(*args)
        except (PoseLost, UnportedBranch):
            self.fresh = None
            self.outage['restarts'] += 1
            return carried
        if frame.frame > start:
            self.fresh[3] = measured = measured+1
        if measured < TAKEOVER_FRAMES:
            return carried
        placed = self._placed(pose, offset, frame)
        self.odometry, self.offset, self.fresh = odometry, offset, None
        self.outages.append(self.outage | dict(end_frame=frame.frame, closed=True,
                                               duration_s=round((frame.measured_ns-self.outage['start_ns'])/1e9, 2),
                                               takeover_jump_m=round(float(np.linalg.norm(
                                                   placed.position[:2]-carried.position[:2])), 4)))
        self.outage = None
        return placed

    def open_outage(self):
        """The outage still open at the mission's end, if any."""
        return None if self.outage is None else self.outage | dict(closed=False)
