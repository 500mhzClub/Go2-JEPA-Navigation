"""The stand-off data gate (docs/dynamic/PLAN.md P14), read from a corpus's recorded traces.

    python -m go2nav.data.standoff_audit CORPUS [--kinds K ...] [--roles R ...] [--workers N] [--out FILE]

From each unit's frames (10 Hz: `steps.npz` pose, `dynamic.npz` walker positions, `meta.json` behaviours):
- A **stand-off** is a stretch of frames in which a walker waits on the robot: it has stood still (moved under 1 mm a
  frame) for at least 1 s, within 0.70 m of the base, with the robot ahead of its last heading. This is the measure of
  `eval.metrics` and of P11's analysis.
- It is **resolved within 10 s** if, within 10 s of its start, the walker is out of the stand-off and more than 1.0 m
  from the robot.
- The robot **moves away** if, within 15 s of its start, the robot stands at least 1 m from where it stood when the
  stand-off began, further from the walker than it was.

Reported overall and by walker behaviour ('reactive', 'non-yielding', 'polite') and by layout family (the unit's set):
stand-offs, their frames, and the shares resolved and moved away. The P14 gate needs at least 25% of stand-off frames
resolved within 10 s, at least 20% of stand-offs with the robot moving away, and every behaviour and family at least 15%
of the stand-off frames.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

FRAME_S, STILL_FRAMES, STAND_OFF_M, STILL_M = .1, 10, .70, 1e-3
RESOLVE_FRAMES, CLEAR_M = 100, 1.0
AWAY_FRAMES, AWAY_M = 150, 1.0


def behaviour(placement):
    if placement.get('behaviour') == 'reactive':
        return 'reactive'
    return 'polite' if placement.get('polite', True) else 'non-yielding'


def waiting_frames(base, walker):
    """(n,) whether this walker waits on the robot at each frame (see the module docstring)."""
    n = len(base)
    out, heading, still = np.zeros(n, bool), np.zeros(2), 0
    for t in range(1, n):
        d = walker[t]-walker[t-1]
        if np.linalg.norm(d) > STILL_M:
            heading, still = d/np.linalg.norm(d), 0
        else:
            still += 1
        r = base[t]-walker[t]
        out[t] = still >= STILL_FRAMES and np.linalg.norm(r) < STAND_OFF_M and heading@r > 0
    return out


def stretches(mask):
    """(start, end) of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def audit_unit(job):
    corpus, unit = job
    folder = Path(corpus)/unit['path']
    meta = json.loads((folder/'meta.json').read_text())
    placement = (meta.get('obstacles') or {}).get('placement') or []
    if not placement:
        return []
    with np.load(folder/'steps.npz') as steps:
        base = steps['pose'][:, :2].astype(float)
    with np.load(folder/'dynamic.npz') as dynamic:
        walkers = dynamic['obstacles_xy'].astype(float)
    rows = []
    for j, o in enumerate(placement[:walkers.shape[1]]):
        w = walkers[:, j]
        waiting = waiting_frames(base, w)
        for start, end in stretches(waiting):
            after = slice(start, min(start+RESOLVE_FRAMES, len(base)))
            free = ~waiting[after] & (np.linalg.norm(w[after]-base[after], axis=1) > CLEAR_M)
            later = slice(start, min(start+AWAY_FRAMES, len(base)))
            moved = np.linalg.norm(base[later]-base[start], axis=1)
            further = np.linalg.norm(base[later]-w[start], axis=1) > np.linalg.norm(base[start]-w[start])
            rows.append(dict(behaviour=behaviour(o), family=meta.get('set', 'unknown'), frames=int(end-start),
                             resolved=bool(free.any()), moved_away=bool((moved >= AWAY_M)[further].any())))
    return rows


def summary(rows):
    def block(rs):
        frames = sum(r['frames'] for r in rs)
        return dict(stand_offs=len(rs), frames=frames,
                    resolved_frame_share=round(sum(r['frames'] for r in rs if r['resolved'])/frames, 4) if frames else None,
                    moved_away_share=round(sum(r['moved_away'] for r in rs)/len(rs), 4) if rs else None)
    out = dict(all=block(rows))
    for key in ('behaviour', 'family'):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        total = sum(r['frames'] for r in rows) or 1
        out[key] = {k: block(v) | dict(frame_share=round(sum(r['frames'] for r in v)/total, 4)) for k, v in sorted(groups.items())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('corpus', type=Path)
    p.add_argument('--kinds', nargs='+')
    p.add_argument('--roles', nargs='+')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    units = [u for u in json.loads((a.corpus/'index_12.json').read_text())['units']
             if (not a.kinds or u['kind'] in a.kinds) and (not a.roles or u['role'] in a.roles)]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = [r for part in pool.map(audit_unit, [(str(a.corpus), u) for u in units], chunksize=8) for r in part]
    report = dict(corpus=str(a.corpus), units=len(units), **summary(rows))
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text+'\n')


if __name__ == '__main__':
    main()
