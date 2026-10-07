"""Compare regenerated training recordings with the original programme's, trial by trial (porting tool).

    python tools/porting/collection_check.py ROOT [--collection NAME ...] [--workers 12]

ROOT is a `go2nav.data.recordings` output. For every trial of each collection, against the original recording
(`tools/porting/original.py`: the recovery or workspace drive's navigation_development_artifacts_v1/<folder>/<trial>):
- the physics trace: every array equal;
- every RGB frame (primary and auxiliary): equal pixels, and the same frames;
- the policy histories: every array the two have in common equal;
- native depth: equal, for the frames whose original depth was kept (the collection's depth_retention.json lists the
  retired trials; their depth files are absent and are not compared).
Writes ROOT/collection_check.json and prints one line per collection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2nav.data.recordings import FOLDERS, trials  # noqa: E402
import original  # noqa: E402

ARTIFACTS = '.generated/navigation_development_artifacts_v1'


def original_trial(folder, trial):
    for root in ('recovery', 'workspace'):
        path = original.path(root, ARTIFACTS, folder, trial)
        if path.is_dir():
            return path
    return None


def arrays_equal(a_path, b_path):
    with np.load(a_path) as a, np.load(b_path) as b:
        common = sorted(set(a.files) & set(b.files))
        return dict(equal=all(np.array_equal(a[k], b[k]) for k in common), common=len(common),
                    only_regenerated=sorted(set(a.files)-set(b.files)), only_original=sorted(set(b.files)-set(a.files)))


def compare(task):
    collection, trial, root = task
    folder = FOLDERS[collection][0]
    new, old = Path(root)/folder/trial, original_trial(folder, trial)
    if old is None:
        return dict(collection=collection, trial=trial, original='missing')
    row = dict(collection=collection, trial=trial, trace=arrays_equal(new/'physics_trace.npz', old/'physics_trace.npz'))
    if (old/'policy_histories.npz').exists() and (new/'policy_histories.npz').exists():
        row['histories'] = arrays_equal(new/'policy_histories.npz', old/'policy_histories.npz')
    for kind in ('rgb', 'auxiliary_rgb'):
        names = sorted(p.name for p in old.glob(f'{kind}_[0-9][0-9][0-9][0-9].png'))
        produced = sorted(p.name for p in new.glob(f'{kind}_[0-9][0-9][0-9][0-9].png'))
        if names or produced:
            differing = [n for n in names if n in produced and
                         not np.array_equal(np.asarray(Image.open(new/n)), np.asarray(Image.open(old/n)))]
            row[kind] = dict(original=len(names), regenerated=len(produced), same_names=names == produced,
                             differing=len(differing), first_differing=differing[:3])
    depth = sorted(p.name for p in old.glob('native_depth_[0-9][0-9][0-9][0-9].npz'))
    if depth:
        differing = [n for n in depth if not (new/n).exists() or not arrays_equal(new/n, old/n)['equal']]
        row['native_depth'] = dict(compared=len(depth), differing=len(differing), first_differing=differing[:3])
    row['identical'] = (row['trace']['equal'] and row.get('histories', {}).get('equal', True)
                        and all(row[k]['same_names'] and not row[k]['differing'] for k in ('rgb', 'auxiliary_rgb') if k in row)
                        and not row.get('native_depth', {}).get('differing'))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--collection', action='append', choices=sorted(FOLDERS), dest='collections')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    tasks = [(c, t, str(a.root.resolve())) for c in (a.collections or FOLDERS) for t, _ in trials(c)]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(compare, tasks))
    summary = {}
    for row in rows:
        s = summary.setdefault(row['collection'], dict(trials=0, identical=0, original_missing=0, trace_differs=0,
                                                       frames_differ=0, depth_compared=0, depth_differs=0))
        s['trials'] += 1
        if row.get('original') == 'missing':
            s['original_missing'] += 1
            continue
        s['identical'] += row['identical']
        s['trace_differs'] += not row['trace']['equal']
        s['frames_differ'] += any(row[k]['differing'] or not row[k]['same_names'] for k in ('rgb', 'auxiliary_rgb') if k in row)
        s['depth_compared'] += row.get('native_depth', {}).get('compared', 0)
        s['depth_differs'] += row.get('native_depth', {}).get('differing', 0)
    (a.root/'collection_check.json').write_text(json.dumps(dict(summary=summary, trials=rows), indent=1)+'\n')
    for collection, s in summary.items():
        print(collection, json.dumps(s))


if __name__ == '__main__':
    main()
