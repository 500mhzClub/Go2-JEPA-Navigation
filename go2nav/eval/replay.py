"""Recorded decisions, decided again (docs/dynamic/PLAN.md P19): the cheapest check of a change to `planning.cost`.

A mission run with `--decision cost` saves `decision_inputs.npz`: everything `cost.select` was given at each plan (the
forecasts, the waypoint or scan heading, the range returns, the reflex's obstacle cells, the flags). `replay` gives the
same inputs to the code as it is now and reports where the choice differs from the one the mission made. It takes
seconds where a suite takes hours. It is one step, open loop: it shows that a choice changes at the states a mission
passed through, not what the mission would then have done.

    python -m go2nav.eval.replay RUN [RUN ...] [--where STATUS]     # RUN: a mission's folder, or a cohort's
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from go2nav.planning import cost


def save(path, rows, candidates, arrival_radius_m):
    """The inputs of every plan in one file; the returns and the cells, whose counts vary, are stacked with offsets."""
    def stacked(key, kind):
        parts = [np.zeros((0, 2), kind) if r[key] is None or not len(r[key]) else
                 np.asarray(sorted(r[key]) if key == 'cells' else r[key], kind)[:, :2] for r in rows]
        return np.concatenate(parts), np.cumsum([0]+[len(p) for p in parts])
    points, points_at = stacked('points', np.float64)
    cells, cells_at = stacked('cells', np.int32)
    np.savez_compressed(
        path, frame=np.array([r['frame'] for r in rows]), prediction=np.array([np.asarray(r['prediction']) for r in rows]),
        goal_body=np.array([r['goal_body'] for r in rows], np.float64),
        scan_error=np.array([np.nan if r['scan_error'] is None else r['scan_error'] for r in rows], np.float64),
        target=np.array([r['target'] for r in rows], bool), pulse=np.array([r['pulse'] for r in rows], bool),
        no_points=np.array([r['points'] is None for r in rows]), no_cells=np.array([r['cells'] is None for r in rows]),
        points=points, points_at=points_at, cells=cells, cells_at=cells_at, candidates=np.array(candidates),
        arrival_radius_m=arrival_radius_m)


def load(path):
    """The rows `save` was given, as keyword arguments of `cost.select` (with `frame`)."""
    z = np.load(path)
    candidates = [str(c) for c in z['candidates']]
    for k in range(len(z['frame'])):
        scan = float(z['scan_error'][k])
        yield dict(frame=int(z['frame'][k]), prediction=z['prediction'][k], candidates=candidates,
                   goal_body=z['goal_body'][k], scan_error=None if np.isnan(scan) else scan,
                   points=None if z['no_points'][k] else z['points'][z['points_at'][k]:z['points_at'][k+1]],
                   arrival_radius_m=float(z['arrival_radius_m']), target=bool(z['target'][k]), pulse=bool(z['pulse'][k]),
                   cells=None if z['no_cells'][k] else {tuple(int(v) for v in c)
                                                         for c in z['cells'][z['cells_at'][k]:z['cells_at'][k+1]]})


def replay(run, where=None):
    """One mission: the plans decided again. Returns dict(plans, changed, changes: Counter of (was, now))."""
    run = Path(run)
    made = {r['frame']: r for r in json.loads((run/'planning.json').read_text()) if 'selection' in r}
    plans, changes = 0, collections.Counter()
    for row in load(run/'decision_inputs.npz'):
        record = made.get(row.pop('frame'))
        if record is None or (where and record['route_status'] != where):
            continue
        now = cost.select(row.pop('prediction'), row.pop('candidates'), row.pop('goal_body'), row.pop('scan_error'),
                          row.pop('points'), **row)['action']
        plans += 1
        if now != record['action']:
            changes[(record['action'], now)] += 1
    return dict(plans=plans, changed=sum(changes.values()), changes=changes)


def runs_of(path):
    path = Path(path)
    found = [path] if (path/'decision_inputs.npz').exists() else sorted(p.parent for p in path.glob('runs/*/decision_inputs.npz'))
    return [p for p in found if 'attempt' not in p.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--where', help='only the plans made under this route status')
    args = parser.parse_args()
    total, changes = 0, collections.Counter()
    for run in (r for path in args.runs for r in runs_of(path)):
        out = replay(run, args.where)
        total += out['plans']
        changes.update(out['changes'])
        print(f"{run.name}: {out['changed']} of {out['plans']} plans change")
    print(f'all: {sum(changes.values())} of {total} plans change')
    for (was, now), count in changes.most_common(12):
        print(f'  {count:6d}  {was} -> {now}')


if __name__ == '__main__':
    main()
