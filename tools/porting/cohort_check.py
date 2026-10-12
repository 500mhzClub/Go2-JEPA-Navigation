"""Compare a cohort rerun in the clean repository with the original cohort, mission by mission (porting tool, G4).

    python tools/porting/cohort_check.py outputs/cohorts/rexit2_ref [--traces]

For each mission the original run is `<capability root>/runs/dev_<cohort>_<arm>_<set><maze:02d>_ep<episode>` (the
original set names: prelim_test, stage2_*, dev_tune). Compared: the evaluation outcome (round trip, beacon, home,
contacts, hard and operating violations, minimum clearance, SPL, failure taxonomy) and, with --traces, the requests and
the physics trace (exact). The original did not evaluate missions that ended in a physical stop (a disallowed contact,
a recording stall stop); for those the recorded ending (error, frames, policy steps) is compared instead.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import original

CAPABILITY = original.path('recovery', '.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
REPO = Path(__file__).resolve().parents[2]
FIELDS = ('round_trip_success', 'beacon_success', 'home_success', 'disallowed_contact_samples', 'failure_and_stall_taxonomy')


def original_run(arm, set_name, maze, episode, original_cohort):
    if set_name == 'prelim_test_v1':
        role = 'prelim_test'
    elif set_name == 'stage2_sets_v1':
        registry = json.loads((REPO/'data/mazes/stage2_sets_v1/registry.json').read_text())
        role = next(e['role'] for e in registry['entries'] if e['maze_id'] == maze)
    else:
        role = set_name
    return CAPABILITY/'runs'/f'dev_{original_cohort}_{arm}_{role}{maze:02d}_ep{episode}'


def outcome(evaluation):
    s = evaluation['safety']
    return dict({k: evaluation[k] for k in FIELDS}, hard=s['hard']['confirmed_violation_samples'],
                operating=s['operating']['confirmed_violation_samples'], min_clearance_m=s['hard']['minimum_separation_lower_m'],
                outbound_spl=evaluation['outbound']['spl'], return_spl=evaluation['return_leg']['spl'])


def traces_differ(a, b):
    ra, rb = (json.loads((r/'requests.json').read_text()) for r in (a, b))
    key = lambda rows: [(x['requested_command'], x['reason'], x.get('applied_command')) for x in rows]
    if key(ra) != key(rb):
        return 'requests differ'
    ta, tb = (np.load(r/'native/physics_trace.npz') for r in (a, b))
    if len(ta['timestamp_s']) != len(tb['timestamp_s']) or any(not np.array_equal(ta[k], tb[k]) for k in ('base_pose_world', 'joint_position')):
        return 'physics trace differs'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cohort', type=Path)
    p.add_argument('--traces', action='store_true')
    a = p.parse_args()
    config = json.loads((a.cohort/'config.json').read_text())
    original_cohort = config['cohort'].get('original_cohort', config['cohort']['name'])
    same = 0
    for arm, set_name, maze, episode, assignment in config['plan']:
        ported = a.cohort/'runs'/assignment
        original = original_run(arm, set_name, maze, episode, original_cohort)
        if not (ported/'result.json').exists() or not (original/'result.json').exists():
            print(f'{assignment}: not comparable (missing run)')
            continue
        if (original/'episode_evaluation.json').exists():
            if not (ported/'episode_evaluation.json').exists():
                print(f'{assignment}: not comparable (missing evaluation)')
                continue
            got, want = (outcome(json.loads((r/'episode_evaluation.json').read_text())) for r in (ported, original))
        else:
            got, want = ({k: json.loads((r/'result.json').read_text())[k] for k in ('error', 'frames', 'policy_steps')}
                         for r in (ported, original))
        diffs = [k for k in want if got[k] != want[k]]
        if a.traces:
            trace = traces_differ(original, ported)
            if trace:
                diffs.append(trace)
        same += not diffs
        print(f'{assignment}: {"identical" if not diffs else diffs}', flush=True)
    print(f'{same} of {len(config["plan"])} missions identical')


if __name__ == '__main__':
    sys.exit(main())
