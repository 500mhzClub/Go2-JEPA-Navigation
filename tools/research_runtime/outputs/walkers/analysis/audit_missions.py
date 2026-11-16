"""Read-only audit of the walker cohorts: outcomes, failure classes, time use, and how far the model's tracks reach."""
import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path('outputs/cohorts')
GROUPS = {
    'ws': ['ws_obst1_c5', 'ws_obst2_c5'], 'wl': ['wl_obst1_c5', 'wl_obst2_c5'],
    'wh': ['wh_obst1_c5', 'wh_obst2_c5'], 'wo': ['wo_obst1_c5', 'wo_obst2_c5'],
    'wf_ref': ['wf_ref_c5'], 'wf_held': ['wf_held_c5'],
}


def one(run):
    run = Path(run)
    ev = json.loads((run/'episode_evaluation.json').read_text()) if (run/'episode_evaluation.json').exists() else {}
    res = json.loads((run/'result.json').read_text())
    plan = json.loads((run/'planning.json').read_text()) if (run/'planning.json').exists() else []
    acted = [p for p in plan if 'action' in p]
    st_counts = collections.Counter(p.get('route_status') for p in acted)
    reasons = collections.Counter(p.get('reason') for p in plan if 'action' not in p)
    rev = fwd = turn = hold = 0
    for p in acted:
        c = p['selection'].get('requested_command') or [0, 0, 0]
        if c[0] < 0:
            rev += 1
        elif c[0] > 0:
            fwd += 1
        elif c[2] != 0:
            turn += 1
        else:
            hold += 1
    # Where the model's confirmed tracks could steer anything: plans on a cut route, and the stand-off's reach.
    cut = [p for p in acted if p.get('route_status') == 'OBSERVED_FLOOR_ROUTE_TO_CUT']
    shadow = {}
    if (run/'walker_shadow.json').exists():
        for row in json.loads((run/'walker_shadow.json').read_text()):
            shadow[row['frame']] = [t for t in row['tracks'] if t.get('confirmed')]
    cut_near = 0
    any_conf = 0
    for p in acted:
        if shadow.get(p['frame']):
            any_conf += 1
    for p in cut:
        target = p['route'].get('target_map_xy_m')
        for t in shadow.get(p['frame'], []):
            if target and np.hypot(t['centroid_m'][0]-target[0], t['centroid_m'][1]-target[1]) < 1.3:
                cut_near += 1
                break
    out = dict(run=run.name, round_trip=bool(ev.get('round_trip_success')), beacon=bool(ev.get('beacon_success')),
               error=res.get('error'), elapsed=res.get('simulated_s'), plans=len(plan), acted=len(acted),
               status=dict(st_counts), reasons=dict(reasons), rev=rev, fwd=fwd, turn=turn, hold=hold,
               cut=len(cut), cut_near=cut_near, any_confirmed=any_conf,
               contacts=ev.get('disallowed_contact_samples', 0),
               outbound=ev.get('outbound'), ret=ev.get('return_leg'),
               taxonomy=ev.get('failure_and_stall_taxonomy'), source_error=ev.get('source_error'))
    tim = run/'stage_timings.json'
    out['plan_ms'] = st.median(t['wall_ns'] for t in json.loads(tim.read_text()))/1e6 if tim.exists() and tim.stat().st_size > 5 else None
    return out


def main():
    all_rows = {}
    with ProcessPoolExecutor(14) as pool:
        for name, cohorts in GROUPS.items():
            runs = [r for c in cohorts for r in sorted((ROOT/c/'runs').iterdir()) if r.is_dir()]
            all_rows[name] = list(pool.map(one, runs, chunksize=4))
            print(name, len(runs), file=sys.stderr, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(all_rows))


if __name__ == '__main__':
    main()
