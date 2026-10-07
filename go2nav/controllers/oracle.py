"""C0: the simulator as the motion forecaster (an upper bound, impossible on a real robot).

At each plan the simulator is snapshotted (`go2nav.sim.snapshot`) and each of the six candidate command tapes (the
committed prefix, the action, then zero; the same float32 command encoding the learned models receive) is executed
for 800 ms from the snapshot; the forecast is the true body pose at 100-800 ms, relative to the pose at the plan. The
simulator is then restored, so the mission continues exactly as if nothing had been tried.

After the mission, `verify_executed` checks the oracle against what happened: wherever the executed commands matched
a branch's commands from its start, the executed poses must match that branch's poses within 1 mm and 0.1° (the
restore is not bit-perfect in Genesis, so this can occasionally fail; a failure is recorded as the run's error, as in
the original).
"""
import math

import numpy as np
import torch

from go2nav.planning.actions import command_sequences
from go2nav.sim import snapshot as snapshots
from go2nav.sim.sensors import rotation_xyzw

SCALE = (.3, 1., .5)
STEPS_PER_TICK, TICKS = 5, 8


def encoded_tapes(prefix, pulse):
    """(6, 8, 3) float32 candidate commands after the models' float32 normalisation round trip."""
    scale = torch.tensor(SCALE)
    blocks = torch.as_tensor(command_sequences(prefix, pulse=pulse), dtype=torch.float32)/scale
    return (blocks*scale).numpy()


def heading(rotation):
    return math.atan2(rotation[1, 0], rotation[0, 0])


class OracleForecaster:
    """C0 (see the module docstring); `receipts` and `branches` record every plan's branches."""

    def __init__(self, session):
        self.session = session
        self.receipts, self.branches = [], []

    def forecast(self, context, prefix, *, pulse):
        session = self.session
        requested = encoded_tapes(prefix, pulse)
        snapshot = snapshots.capture(session)
        if snapshot['measured_ns'] != context.measured_ns:
            raise ValueError('the oracle snapshot is not at the plan time')
        records = {name: getattr(session, name) for name in snapshots.RECORD_LISTS}
        initial = session.samples[-1]['base_pose_world'].copy()
        Q = rotation_xyzw(initial[3:])
        motions, branches, applied = [], [], []
        try:
            for tape in requested:
                snapshots.restore(session, snapshot)
                for name in records:
                    setattr(session, name, [])
                commands = [session.step(command) for command in np.repeat(tape, STEPS_PER_TICK, axis=0)]
                poses = np.stack([row['base_pose_world'] for row in session.samples])
                horizon = poses[np.arange(49, 400, 50)]
                xy = (horizon[:, :3]-initial[:3])@Q
                yaw = [heading(Q.T@rotation_xyzw(p[3:])) for p in horizon]
                motions.append(np.c_[xy[:, :2], yaw])
                branches.append(dict(applied=np.asarray(commands), poses=poses))
                applied.append(np.asarray(commands)[::STEPS_PER_TICK].tolist())
        finally:
            snapshots.restore(session, snapshot)
            for name, value in records.items():
                setattr(session, name, value)
        motion = torch.tensor(np.asarray(motions), dtype=torch.float32)
        self.receipts.append(dict(observed_ns=context.measured_ns, requested_commands=requested.tolist(),
                                  applied_commands=applied, motion_xy_yaw=motion.tolist()))
        self.branches.append(dict(observed_ns=context.measured_ns, branches=branches))
        return torch.cat((motion[:, :, :2], motion[:, :, 2:3].sin(), motion[:, :, 2:3].cos(),
                          torch.full((6, TICKS, 1), -1000.)), dim=-1).numpy()

    def verify_executed(self, requests, samples):
        """The executed-prefix check (see the module docstring)."""
        stamps = np.rint(np.array([s['timestamp_s'] for s in samples])*1e9).astype(np.int64)
        poses = np.stack([s['base_pose_world'] for s in samples])
        by_time = {r['simulator_ns']: r for r in requests}
        rows = []
        for plan in self.branches:
            origin = plan['observed_ns']
            for index, branch in enumerate(plan['branches']):
                length = 0
                for tick in range(TICKS*STEPS_PER_TICK):
                    row = by_time.get(origin+tick*20_000_000)
                    if row is None or not np.allclose(row['applied_command'], branch['applied'][tick], rtol=0, atol=1e-7):
                        break
                    length += 1
                if not length:
                    rows.append(dict(observed_ns=origin, candidate=index, matched_prefix_ms=0, passed=False,
                                     reason='NO_MATCHING_EXECUTED_PREFIX'))
                    continue
                indices = np.searchsorted(stamps, origin+np.arange(1, length*10+1)*2_000_000)
                actual, expected = poses[indices], branch['poses'][:length*10]
                position = np.linalg.norm(actual[:, :3]-expected[:, :3], axis=1)
                yaw = [abs(math.atan2(math.sin(d), math.cos(d))) for d in
                       (heading(rotation_xyzw(a[3:]))-heading(rotation_xyzw(b[3:])) for a, b in zip(actual, expected))]
                rows.append(dict(observed_ns=origin, candidate=index, matched_prefix_ms=length*20,
                                 maximum_position_error_m=float(position.max()), maximum_yaw_error_deg=math.degrees(max(yaw)),
                                 passed=float(position.max()) <= .001 and max(yaw) <= math.radians(.1)))
        return dict(passed=bool(rows) and all(r['passed'] for r in rows), comparisons=rows, position_tolerance_m=.001,
                    yaw_tolerance_degrees=.1)
