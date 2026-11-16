"""P7's reproduction check: each smc_repro_* mission against its wo_* record, file by file, exactly."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('outputs/cohorts')
FILES = ('planning.json', 'requests.json', 'walker_shadow.json', 'mission.json', 'poses.json', 'acquisitions.json')
ok = True
for n in (1, 2):
    folder = ROOT/f'smc_repro_obst{n}'/'runs'
    if not folder.exists():
        continue
    for run in sorted(folder.iterdir()):
        name = run.name.replace(f'smc_repro_obst{n}', f'wo_obst{n}_c5')
        record = ROOT/f'wo_obst{n}_c5'/'runs'/name
        if not (run/'result.json').exists():
            print(run.name, 'unfinished')
            ok = False
            continue
        same = {f: hashlib.sha256((run/f).read_bytes()).digest() == hashlib.sha256((record/f).read_bytes()).digest() for f in FILES}
        a, b = np.load(run/'native/physics_trace.npz'), np.load(record/'native/physics_trace.npz')
        same['physics_trace'] = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        x, y = np.load(run/'native/obstacle_trace.npz'), np.load(record/'native/obstacle_trace.npz')
        same['obstacle_trace'] = all(np.array_equal(x[k], y[k]) for k in x.files)
        print(run.name[-22:], 'IDENTICAL' if all(same.values()) else {k: v for k, v in same.items() if not v})
        ok &= all(same.values())
        if not same['planning.json']:
            p, q = (json.loads((r/'planning.json').read_text()) for r in (run, record))
            first = next((i for i, (u, v) in enumerate(zip(p, q)) if u != v), None)
            print('   plans', len(p), len(q), 'first difference at plan', first, 'frame', None if first is None else p[first].get('frame'))
print('REPRODUCED' if ok else 'NOT REPRODUCED')
