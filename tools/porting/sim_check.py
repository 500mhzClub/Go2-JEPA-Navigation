"""Drive the ported simulator with a recorded mission's commands and compare it with the recording (porting tool).

    python tools/porting/sim_check.py RUN_DIR [--bundle BUNDLE_DIR] [--ticks N]

RUN_DIR is an original mission run folder (specification.json, episode.json, requests.json, public_task_cues.json,
native/physics_trace.npz). The ported `MissionSession` is built from the same spec, settled, and then given every
recorded requested command; at every fifth tick it renders a frame. Compared, exactly:
- every physics-trace row (settling included) against native/physics_trace.npz;
- the settled task cues against public_task_cues.json;
- the executed command of every tick against requests.json;
- with --bundle (a replay bundle of the same mission), every frame's packets against the recorded controller input:
  primary RGB, both depth images and validity, auxiliary RGB, the fast gyro and the policy histories.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(os.environ.get('GO2NAV_SNAPSHOT', Path.home()/'Workspace/LeWMQuad-v3-snapshot-2026-10-06'))
sys.path.insert(0, str(REPO))
TRACE_FIELDS = ('timestamp_s', 'base_pose_world', 'base_twist_world', 'joint_position', 'joint_velocity',
                'requested_command', 'post_slew_applied_command', 'applied_command', 'physics_contact', 'phase')


def compare_row(index, row, trace):
    for key in TRACE_FIELDS:
        if not np.array_equal(np.asarray(row[key]), trace[key][index]):
            return key, np.asarray(row[key]), trace[key][index]
    return None


def recorded_frames(bundle):
    sys.path[:0] = [str(REPO/'tools/porting'), str(SNAPSHOT), str(SNAPSHOT/'lewm_genesis'), str(SNAPSHOT/'lewm_worlds')]
    from replay_bundle import read_stream
    for record in read_stream(Path(bundle)/'controller.stream'):
        if record['kind'] == 'submit':
            yield record['packet']


def compare_frame(got, packet):
    policy, depth, fast, auxiliary_depth, auxiliary_rgb, now = got
    checks = dict(measured_ns=(now, packet.measured_ns), rgb=(policy['image']['rgb'], packet.policy['image']['rgb']),
                  depth=(depth['depth_m'], packet.depth['depth_m']), valid=(depth['valid'], packet.depth['valid']),
                  auxiliary_depth=(auxiliary_depth['depth_m'], packet.auxiliary_depth['depth_m']),
                  auxiliary_valid=(auxiliary_depth['valid'], packet.auxiliary_depth['valid']),
                  auxiliary_rgb=(auxiliary_rgb['rgb'], packet.auxiliary_rgb['rgb']),
                  auxiliary_rgb_sha256=(auxiliary_rgb['auxiliary_depth_sha256'], packet.auxiliary_rgb['auxiliary_depth_sha256']),
                  fast_values=(fast['values'], packet.fast['values']), fast_measured=(fast['measured_ns'], packet.fast['measured_ns']))
    for role in ('sensed', 'control'):
        for name, channel in policy['sensor_state'][role].items():
            for field in ('values', 'valid', 'measured_ns'):
                checks[f'{name}_{field}'] = (channel[field], packet.policy['sensor_state'][role][name][field])
    return [k for k, (a, b) in checks.items() if not np.array_equal(np.asarray(a), np.asarray(b))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--ticks', type=int)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    from go2nav import artifacts
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession, settled_task_cues
    spec = json.loads((a.run/'specification.json').read_text())
    episode = json.loads((a.run/'episode.json').read_text())
    requests = json.loads((a.run/'requests.json').read_text())
    trace = dict(np.load(a.run/'native/physics_trace.npz'))
    scene.init_genesis(spec['procedural_seed'])
    started = time.time()
    session = MissionSession(spec, Path(tempfile.mkdtemp(prefix='sim_check_')), policy_checkpoint=artifacts.path('go2_locomotion_policy'))
    pose = session.settle()
    for i, row in enumerate(session.samples):
        bad = compare_row(i, row, trace)
        if bad:
            print(f'settle sample {i}: DIVERGED {bad[0]}: {bad[1]} != {bad[2]}')
            return session
    cues = json.loads((a.run/'public_task_cues.json').read_text())
    if settled_task_cues(episode, pose) != cues:
        print('task cues DIVERGED', settled_task_cues(episode, pose), cues)
        return session
    print(f'settled: {len(session.samples)} samples identical, cues identical ({time.time()-started:.0f} s)', flush=True)
    frames = recorded_frames(a.bundle) if a.bundle else None
    checked, ticks = len(session.samples), 0
    for tick, request in enumerate(requests):
        if a.ticks is not None and tick >= a.ticks:
            break
        if tick % 5 == 0:
            got = session.frame()
            if frames is not None:
                bad = compare_frame(got, next(frames))
                if bad:
                    print(f'frame {tick//5}: DIVERGED {bad}')
                    return session
        session.phase = 2
        applied = session.step(request['requested_command'])
        if applied != request['applied_command']:
            print(f'tick {tick}: applied DIVERGED {applied} != {request["applied_command"]}')
            return session
        for i in range(checked, len(session.samples)):
            bad = compare_row(i, session.samples[i], trace)
            if bad:
                print(f'tick {tick} sample {i}: DIVERGED {bad[0]}: {bad[1]} != {bad[2]}')
                return session
        checked, ticks = len(session.samples), tick+1
        if tick % 500 == 0:
            print(f'tick {tick}: ok ({checked} samples, {time.time()-started:.0f} s)', flush=True)
    print(f'all {ticks} ticks, {checked} samples and {session.frames} frames identical ({time.time()-started:.0f} s)')
    return session


if __name__ == '__main__':
    main()
