"""Compare a mission run by the clean repository with the original run of the same episode (porting tool, gate G4).

    python tools/porting/mission_check.py ORIGINAL_RUN PORTED_RUN

Exact comparisons: every request (command and reason), executed command and simulator time; every physics-trace row;
the frame times; the mission rows' phase, transitions, holds, quiet intervals and terminal; the outcome (frames, steps,
error). Prints the first difference of each kind.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TRACE_FIELDS = ('timestamp_s', 'base_pose_world', 'base_twist_world', 'joint_position', 'joint_velocity',
                'requested_command', 'applied_command', 'physics_contact')
MISSION_FIELDS = ('phase', 'phase_transition', 'hold_required', 'terminal', 'quiet_intervals')


def load(run, name):
    return json.loads((Path(run)/name).read_text())


def first_request_difference(original, ported):
    for i, (a, b) in enumerate(zip(original, ported)):
        for key in ('requested_command', 'reason', 'applied_command', 'simulator_ns'):
            if a.get(key) != b.get(key):
                return f'request {i} {key}: {a.get(key)} != {b.get(key)}'
    if len(original) != len(ported):
        return f'request count {len(original)} != {len(ported)}'
    return None


def first_trace_difference(original, ported):
    a, b = np.load(Path(original)/'native/physics_trace.npz'), np.load(Path(ported)/'native/physics_trace.npz')
    if len(a['timestamp_s']) != len(b['timestamp_s']):
        n = min(len(a['timestamp_s']), len(b['timestamp_s']))
        head = first_trace_difference_n(a, b, n)
        return head or f'trace length {len(a["timestamp_s"])} != {len(b["timestamp_s"])}'
    return first_trace_difference_n(a, b, len(a['timestamp_s']))


def first_trace_difference_n(a, b, n):
    for key in TRACE_FIELDS:
        unequal = np.flatnonzero(np.any((a[key][:n] != b[key][:n]).reshape(n, -1), axis=1))
        if len(unequal):
            return f'trace {key} first differs at sample {unequal[0]}'
    return None


def first_mission_difference(original, ported):
    for i, (a, b) in enumerate(zip(original, ported)):
        for key in MISSION_FIELDS:
            if a[key] != b[key]:
                return f'mission row {i} {key}: {a[key]} != {b[key]}'
    if len(original) != len(ported):
        return f'mission rows {len(original)} != {len(ported)}'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('original', type=Path)
    p.add_argument('ported', type=Path)
    a = p.parse_args()
    report = dict(
        requests=first_request_difference(load(a.original, 'requests.json'), load(a.ported, 'requests.json')),
        trace=first_trace_difference(a.original, a.ported),
        frames=None if [r['measured_ns'] for r in load(a.original, 'acquisitions.json')] ==
        [r['measured_ns'] for r in load(a.ported, 'acquisitions.json')] else 'frame times differ',
        mission=first_mission_difference(load(a.original, 'mission.json'), load(a.ported, 'mission.json')))
    original, ported = load(a.original, 'result.json'), load(a.ported, 'result.json')
    report['outcome'] = None if all(original[k] == ported[k] for k in ('frames', 'policy_steps')) else \
        f"frames/steps {original['frames']}/{original['policy_steps']} != {ported['frames']}/{ported['policy_steps']}"
    for key, value in report.items():
        print(f'{key}: {"identical" if value is None else value}')
    print('errors:', original['error'], '|', ported['error'])


if __name__ == '__main__':
    main()
