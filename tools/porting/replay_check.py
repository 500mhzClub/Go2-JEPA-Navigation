"""Replay a recorded bundle's component traffic into a ported component and compare, call by call (porting tool).

    python tools/porting/replay_check.py pose outputs/porting/bundles/C1_on_prelim30 [--frames N] [--shadow]

Components:
- `pose`: `go2nav.perception.visual_odometry.VisualOdometry` against the pose worker's `current_pose` (frame,
  reference frame, position, rotation, gyro rotation, promotion and keyframe count; exact equality).
- `obstacles`: `go2nav.perception.obstacles.DepthObstacleObserver` against the obstacle worker's occupied cells.
- `controller`: `go2nav.controllers.controller.NavigationController` (with its own ported perception), fed the recorded
  frames and 50 Hz tick times, against every recorded dispatch request (command and reason) and mission row.
- `mapping`: `go2nav.perception.mapping.RoutingMap`, fed the recorded registered pose, against every field of the
  mapping worker's snapshot.
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
                c['promoted_keyframe'], c['promotion_reason'], c['keyframe_count'], None, None)


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


MAP_FIELDS = ('frame', 'measured_ns', 'floor', 'occupied', 'position_map', 'map_from_initial', 'floor_height',
              'primary_current_floor_cells', 'auxiliary_current_floor_cells', 'fine_occupied', 'current_floor',
              'current_occupied', 'current_fine_occupied')


def check_mapping(bundle, frames=None):
    from go2nav.perception.mapping import RoutingMap
    routing = RoutingMap()
    started = time.time()
    for n, (args, kwargs, result) in enumerate(calls(bundle, 'mapping')):
        if frames is not None and n >= frames:
            break
        packet, evidence = args
        snapshot, _ = result
        pose = evidence['current_pose']
        got = routing.update(pose['frame'], packet.measured_ns, pose['position_initial_body_m'],
                             pose['rotation_initial_body_from_current_body'],
                             (packet.depth['depth_m'], packet.depth['valid']),
                             (packet.auxiliary_depth['depth_m'], packet.auxiliary_depth['valid']),
                             packet.policy['sensor_state']['sensed']['specific_force']['values'])
        diffs = [f for f in MAP_FIELDS if getattr(got, f) != getattr(snapshot, f)]
        if diffs:
            print(f'frame {packet.frame}: DIVERGED {diffs}', {f: (len(getattr(got, f)), len(getattr(snapshot, f)))
                  for f in diffs if isinstance(getattr(got, f), frozenset)})
            return routing, args, result
        if n % 50 == 0:
            print(f'frame {packet.frame}: ok ({time.time()-started:.0f} s)', flush=True)
    print(f'all {n+1} map updates identical ({time.time()-started:.0f} s)')


def sensor_frame(packet):
    from go2nav.controllers.controller import SensorFrame
    return SensorFrame(packet.frame, packet.measured_ns, packet.policy['image']['rgb'],
                       (packet.depth['depth_m'], packet.depth['valid']), packet.auxiliary_rgb['rgb'],
                       (packet.auxiliary_depth['depth_m'], packet.auxiliary_depth['valid']), packet.fast['values'],
                       packet.policy['sensor_state']['sensed']['specific_force']['values'], packet.policy, tuple(packet.history))


C1_MODEL = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
                'go2_short_pulse_command_control_v1_attempt_001/command_only.npz')


def build_controller(bundle):
    import json
    from go2nav.controllers.controller import NavigationController
    from go2nav.controllers.forecasters import CommandHistoryForecaster
    info = json.loads((Path(bundle)/'bundle.json').read_text())
    mission = info['mission']
    arm = mission[mission.index('--controller')+1]
    recovery = mission[mission.index('--recovery')+1] == 'on' if '--recovery' in mission else False
    run = Path(info['run'])
    cues = json.loads((run/'public_task_cues.json').read_text())
    if arm == 'C1':
        forecaster = CommandHistoryForecaster(C1_MODEL)
    elif arm == 'C3':
        forecaster = jepa_forecaster(mission[mission.index('--c3-decoder')+1])
    elif arm == 'C4':
        forecaster = direct_forecaster(mission[mission.index('--c4-weights')+1])
    else:
        raise SystemExit(f'controller check for {arm} not ported yet')
    return NavigationController(forecaster, goal_xy=cues['goal_initial_body_xy_m'], home_xy=cues['return_initial_body_xy_m'],
                                arrival_radius_m=.02, navigation_frames=4800, recovery=recovery), run


T4 = '/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/action_final.pt'
T5B = '/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_readout_v1_attempt_003/maze_data_final.pt'


def jepa_forecaster(decoder_path):
    import json
    from go2nav.controllers.neural import JepaForecaster
    from go2nav.models.motion import load_decoder_and_c4, load_motion_readout
    from go2nav.models.predictor import load_horizon_predictor
    stats = json.loads((REPO/'data/training/control_norm_stats.json').read_text())
    decoder, _ = load_decoder_and_c4(decoder_path, load_motion_readout(T5B))
    return JepaForecaster(load_horizon_predictor(T4), decoder, stats['control_mean'], stats['control_std'])


def direct_forecaster(weights_path):
    import json
    from go2nav.controllers.neural import DirectForecaster
    from go2nav.models.motion import load_decoder_and_c4, load_motion_readout
    stats = json.loads((REPO/'data/training/control_norm_stats.json').read_text())
    _, c4 = load_decoder_and_c4(weights_path, load_motion_readout(T5B))
    return DirectForecaster(c4, stats['control_mean'], stats['control_std'])


def check_controller(bundle, frames=None):
    import json
    controller, run = build_controller(bundle)
    mission_rows = json.loads((run/'mission.json').read_text())
    started, ticks, n = time.time(), 0, 0
    for record in read_stream(Path(bundle)/'controller.stream'):
        if record['kind'] == 'submit':
            packet = record['packet']
            if frames is not None and packet.frame >= frames:
                break
            controller.on_frame(sensor_frame(packet))
            row, expected = controller.mission_rows[-1], mission_rows[packet.frame]
            for key in ('phase', 'phase_transition', 'hold_required', 'terminal', 'quiet_intervals'):
                if row[key] != expected[key]:
                    print(f'frame {packet.frame}: mission {key} {row[key]} != {expected[key]}')
                    return controller, record
            if packet.frame % 100 == 0:
                print(f'frame {packet.frame}: ok ({ticks} ticks, {time.time()-started:.0f} s)', flush=True)
            n += 1
        else:
            got = controller.request(record['now_ns'])
            want = record['result']
            if got['requested_command'] != want['requested_command'] or got['reason'] != want['reason']:
                print(f"tick {record['now_ns']}: DIVERGED {got} != {want['requested_command']} {want['reason']}")
                return controller, record
            ticks += 1
    print(f'all {n} frames and {ticks} ticks identical ({time.time()-started:.0f} s)')
    return controller, None


def _original_tracker():
    from lewm.sparse_corner_completion_runtime_development import SparseCornerCompletionMotion
    from lewm.two_cm_floor_extent_development import configure
    configure()
    return SparseCornerCompletionMotion()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('component', choices=['pose', 'registration', 'obstacles', 'mapping', 'controller'])
    p.add_argument('bundle')
    p.add_argument('--frames', type=int)
    p.add_argument('--shadow', action='store_true')
    a = p.parse_args()
    if a.component == 'pose':
        check_pose(a.bundle, a.frames, a.shadow)
    elif a.component == 'registration':
        check_registration(a.bundle, a.frames)
    elif a.component == 'obstacles':
        check_obstacles(a.bundle, a.frames)
    elif a.component == 'mapping':
        check_mapping(a.bundle, a.frames)
    else:
        check_controller(a.bundle, a.frames)


if __name__ == '__main__':
    main()
