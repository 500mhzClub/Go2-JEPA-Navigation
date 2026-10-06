"""Replay a recorded bundle's component traffic into a ported component and compare, call by call (porting tool).

    python tools/porting/replay_check.py pose outputs/porting/bundles/C1_on_prelim30 [--frames N] [--shadow]

Components:
- `pose`: `go2nav.perception.visual_odometry.VisualOdometry` against the pose worker's `current_pose` (frame,
  reference frame, position, rotation, gyro rotation, promotion and keyframe count; exact equality).
- `obstacles`: `go2nav.perception.obstacles.DepthObstacleObserver` against the obstacle worker's occupied cells.
- `registration`: `go2nav.perception.registration.FloorRegistration`, fed the recorded visual pose, against the
  registration worker's corrected pose (position, rotation, transported or registered; exact equality).

`--shadow` also runs the original component (from the snapshot) side by side, to compare internal state at the first
divergence. Unpickling a bundle needs the snapshot on sys.path (GO2NAV_SNAPSHOT, default the local snapshot).
"""
import argparse
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(os.environ.get('GO2NAV_SNAPSHOT', Path.home()/'Workspace/LeWMQuad-v3-snapshot-2026-10-06'))
sys.path[:0] = [str(REPO), str(SNAPSHOT), str(SNAPSHOT/'lewm_genesis'), str(SNAPSHOT/'lewm_worlds')]
sys.path.insert(0, str(REPO/"tools/porting"))
from replay_bundle import read_stream  # noqa: E402


def calls(bundle, pool):
    """(args, kwargs, result) of every call to a recorded pool's work function (the readiness probe skipped)."""
    for record in read_stream(Path(bundle)/f'pool_{pool}.stream'):
        if record['function'].endswith('_ready'):
            continue
        yield pickle.loads(record['args']), pickle.loads(record['kwargs']), record['result']


def pose_inputs(packet):
    return dict(rgb=packet.policy['image']['rgb'], depth=(packet.depth['depth_m'], packet.depth['valid']),
                auxiliary_rgb=packet.auxiliary_rgb['rgb'],
                auxiliary_depth=(packet.auxiliary_depth['depth_m'], packet.auxiliary_depth['valid']),
                gyro_rates=packet.fast['values'], specific_force=packet.policy['sensor_state']['sensed']['specific_force']['values'],
                now=packet.measured_ns)


def compare_pose(pose, original):
    """Field-by-field differences between a ported Pose and the original current_pose."""
    expected = dict(frame=original['frame'], measured_ns=original['measured_ns'], reference_frame=original['reference_frame'],
                    promoted_keyframe=original['promoted_keyframe'], promotion_reason=original['promotion_reason'],
                    keyframe_count=original['keyframe_count'])
    got = dict(frame=pose.frame, measured_ns=pose.measured_ns, reference_frame=pose.reference_frame,
               promoted_keyframe=pose.promoted_keyframe, promotion_reason=pose.promotion_reason,
               keyframe_count=pose.keyframe_count)
    diffs = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    for name, value in (('position_initial_body_m', pose.position), ('rotation_initial_body_from_current_body', pose.rotation),
                        ('gyro_rotation_initial_body_from_current_body', pose.gyro_rotation)):
        reference = np.asarray(original[name])
        if not np.array_equal(np.asarray(value), reference):
            diffs[name] = float(np.max(np.abs(np.asarray(value)-reference)))
    return diffs


def check_pose(bundle, frames=None, shadow=False):
    from go2nav.perception.visual_odometry import VisualOdometry
    odometry, original_tracker = VisualOdometry(), None
    if shadow:
        original_tracker = _original_tracker()
    started = time.time()
    for n, (args, kwargs, result) in enumerate(calls(bundle, 'pose')):
        if frames is not None and n >= frames:
            break
        packet = args[0]
        if original_tracker is not None:
            original_tracker.observe(packet.policy, packet.depth, packet.fast, now_ns=packet.measured_ns,
                                     auxiliary_rgb=packet.auxiliary_rgb, auxiliary_depth=packet.auxiliary_depth)
        pose = odometry.observe(**pose_inputs(packet))
        diffs = compare_pose(pose, result['current_pose'])
        if diffs:
            print(f'frame {packet.frame}: DIVERGED {diffs}')
            print('  original selection:', result['reference_selection'].get('status'), result['reference_selection'].get('selected_reference'),
                  '| camera', result['camera_selection'].get('selected_camera'), '| ported camera', pose.camera)
            return odometry, original_tracker, packet, result
        if n % 100 == 0:
            print(f'frame {packet.frame}: ok ({time.time()-started:.0f} s)', flush=True)
    print(f'all {n+1} frames identical ({time.time()-started:.0f} s)')


def visual_pose(raw):
    """The recorded visual evidence as the ported Pose."""
    from go2nav.perception.visual_odometry import Pose
    c = raw['current_pose']
    return Pose(c['frame'], c['measured_ns'], np.asarray(c['position_initial_body_m'], float),
                np.asarray(c['rotation_initial_body_from_current_body'], float),
                np.asarray(c['gyro_rotation_initial_body_from_current_body'], float), c['reference_frame'],
                c['promoted_keyframe'], c['promotion_reason'], c['keyframe_count'], None)


def compare_registration(registered, original):
    pose = original['current_pose']
    diffs = {}
    transported = pose['mode'] == 'measured_visual_floor_transport'
    if registered.transported != transported:
        diffs['transported'] = (registered.transported, transported)
    for name, value in (('position_initial_body_m', registered.position),
                        ('rotation_initial_body_from_current_body', registered.rotation)):
        reference = np.asarray(pose[name])
        if not np.array_equal(np.asarray(value), reference):
            diffs[name] = float(np.max(np.abs(np.asarray(value)-reference)))
    return diffs


def check_registration(bundle, frames=None):
    from go2nav.perception.registration import FloorRegistration
    registration = FloorRegistration()
    started = time.time()
    for n, (args, kwargs, result) in enumerate(calls(bundle, 'registration')):
        if frames is not None and n >= frames:
            break
        policy, primary, auxiliary, raw, now_ns = args
        registered = registration.observe((primary['depth_m'], primary['valid']), (auxiliary['depth_m'], auxiliary['valid']),
                                          visual_pose(raw), policy['sensor_state']['sensed']['specific_force']['values'])
        diffs = compare_registration(registered, result)
        if diffs:
            print(f'frame {registered.frame}: DIVERGED {diffs}')
            return registration, args, result
        if n % 200 == 0:
            print(f'frame {registered.frame}: ok ({time.time()-started:.0f} s)', flush=True)
    print(f'all {n+1} frames identical ({time.time()-started:.0f} s)')


def check_obstacles(bundle, frames=None):
    from go2nav.perception.obstacles import DepthObstacleObserver
    observer = DepthObstacleObserver()
    started = time.time()
    for n, (args, kwargs, result) in enumerate(calls(bundle, 'obstacles')):
        if frames is not None and n >= frames:
            break
        packet = args[0]
        current, _ = result
        got = observer.observe((packet.depth['depth_m'], packet.depth['valid']),
                               (packet.auxiliary_depth['depth_m'], packet.auxiliary_depth['valid']), packet.fast['values'],
                               packet.policy['sensor_state']['sensed']['specific_force']['values'], packet.measured_ns)
        if (got is None) != (current is None) or (got is not None and got.cells != current.occupied):
            print(f'frame {packet.frame}: DIVERGED', None if got is None else len(got.cells),
                  None if current is None else len(current.occupied))
            return observer, packet, result
        if n % 200 == 0:
            print(f'frame {packet.frame}: ok ({time.time()-started:.0f} s)', flush=True)
    print(f'all {n+1} frames identical ({time.time()-started:.0f} s)')


def _original_tracker():
    from lewm.sparse_corner_completion_runtime_development import SparseCornerCompletionMotion
    from lewm.two_cm_floor_extent_development import configure
    configure()
    return SparseCornerCompletionMotion()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('component', choices=['pose', 'registration', 'obstacles'])
    p.add_argument('bundle')
    p.add_argument('--frames', type=int)
    p.add_argument('--shadow', action='store_true')
    a = p.parse_args()
    if a.component == 'pose':
        check_pose(a.bundle, a.frames, a.shadow)
    elif a.component == 'registration':
        check_registration(a.bundle, a.frames)
    else:
        check_obstacles(a.bundle, a.frames)


if __name__ == '__main__':
    main()
