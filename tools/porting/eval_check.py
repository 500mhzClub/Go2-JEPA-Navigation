"""Check the ported episode reader against the original's evaluations (porting tool).

    python tools/porting/eval_check.py RUN_DIR... [--workers 8]
    python tools/porting/eval_check.py --against ORIGINAL_RUN PORTED_RUN

Default: each RUN_DIR is an original run with its episode_evaluation.json; the ported reader evaluates the run (without
writing) and every outcome field is compared exactly. --against: the ported reader evaluates both runs (an original
run and the clean repository's run of the same episode) and compares the two records.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCALARS = ('beacon_success', 'home_success', 'round_trip_success', 'disallowed_contact_samples', 'stall_by_phase',
           'hold_categories', 'failure_and_stall_taxonomy')
ARRIVAL = ('phase', 'frame', 'native_minimum_distance_m', 'native_maximum_distance_m', 'native_final_distance_m',
           'native_maximum_100ms_speed_m_s', 'all_requested_intervals_zero', 'passed')
LEG = ('success', 'actual_path_m', 'shortest_path_m', 'spl', 'elapsed_s')
SAFETY = ('confirmed_violation_samples', 'unresolved_sampled_samples', 'minimum_separation_lower_m',
          'minimum_separation_upper_m', 'interval_robustness_failure_count')


def differences(a, b):
    out = [k for k in SCALARS if a.get(k) != b.get(k)]
    if len(a.get('arrivals', [])) != len(b.get('arrivals', [])):
        out.append('arrival count')
    else:
        out += [f'arrival {i} {k}' for i, (x, y) in enumerate(zip(a['arrivals'], b['arrivals'])) for k in ARRIVAL if x.get(k) != y.get(k)]
    out += [f'{name} {k}' for name in ('outbound', 'return_leg') for k in LEG if a[name].get(k) != b[name].get(k)]
    out += [f'safety {t} {k}' for t in ('hard', 'operating') for k in SAFETY
            if k in a.get('safety', {}).get(t, {}) and a['safety'][t].get(k) != b.get('safety', {}).get(t, {}).get(k)]
    holds = lambda r: [(h['frame'], h['category'], h.get('override_reason')) for h in r.get('hold_details', [])]
    if holds(a) != holds(b):
        out.append('hold details')
    return out


def check(run):
    from go2nav.eval.episode import evaluate
    run = Path(run)
    original = json.loads((run/'episode_evaluation.json').read_text())
    try:
        ported = evaluate(run, write=False)
    except Exception as exc:  # report and continue with the other runs
        return run.name, [f'error {exc!r}']
    return run.name, differences(original, ported)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--against', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.against:
        from go2nav.eval.episode import evaluate
        original, ported = (evaluate(r, write=False) for r in a.runs)
        found = differences(original, ported)
        print('identical' if not found else found)
        return
    with ProcessPoolExecutor(a.workers) as pool:
        for name, found in pool.map(check, a.runs):
            print(f'{name}: {"identical" if not found else found}', flush=True)


if __name__ == '__main__':
    main()
