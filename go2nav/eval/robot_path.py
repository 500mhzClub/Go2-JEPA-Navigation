"""P24 stage 2a (docs/dynamic/PLAN.md): the kinematic model's path under a tape against the Go2's own in Genesis.

    python -m go2nav.eval.robot_path --corpus CORPUS --out FILE [--split lewm_val] [--every 8] [--workers N]

On corpus v2's units (`data.lewm_corpus_v2`: steps.npz with the base pose, and each 10-Hz frame's requested and
executed commands at 50 Hz), every `every`-th frame t0 with 6.4 s of the unit left: P17's kinematic model
(`planning.sampling.integrate`) from the pose and executed command at t0 (the last executed command before the frame),
under the recorded requests of the next 6.4 s (one a frame: they are constant within a frame), against the recorded
poses. The median position error (m) and heading error (rad) at 1.6, 3.2 and 6.4 s, over all starts, over those where
the robot moved more than `MOVING_M` in the 6.4 s, and by the unit's kind. The rule (PLAN P24 2a, set beforehand): the
kinematic model is used if its median position error at 3.2 s is at most M5's walker error there (0.074 m).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

HORIZONS, TAPE, MOVING_M, RULE_M = (16, 32, 64), 64, .3, .074


def _unit(args):
    folder, every = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from go2nav.data.genesis_units import yaw_of
    from go2nav.planning.sampling import integrate, wrap
    meta = json.loads((Path(folder)/'meta.json').read_text())
    steps = np.load(Path(folder)/'steps.npz')
    frames = int(meta['frames'])
    xy, yaw = steps['pose'][:frames, :2], yaw_of(steps['pose'][:frames, 3:7])
    requested, executed = steps['requested'][:frames, 0].astype(float), steps['executed'][:frames].astype(float)
    starts = [t for t in range(8, frames-TAPE, every)]
    if not starts:
        return []
    start = dict(xy=None, yaw=None, executed=None)
    rows = []
    for t in starts:
        start = dict(xy=xy[t], yaw=float(yaw[t]), executed=executed[t-1, -1])
        path, heading = integrate(start, requested[t:t+TAPE][None])
        moved = float(np.linalg.norm(xy[t+TAPE]-xy[t]))
        for h in HORIZONS:
            rows.append(dict(kind=meta['kind'], horizon=h, moving=moved > MOVING_M,
                             position=float(np.linalg.norm(path[0, h-1]-xy[t+h])),
                             heading=float(abs(wrap(heading[0, h-1]-yaw[t+h])))))
    return rows


def measure(corpus, split='lewm_val', every=8, workers=None):
    folders = sorted(str(p) for p in (Path(corpus)/split).iterdir() if (p/'steps.npz').exists())
    with ProcessPoolExecutor(workers or os.cpu_count()) as pool:
        rows = [r for part in pool.map(_unit, [(f, every) for f in folders], chunksize=4) for r in part]
    report = dict(units=len(folders), starts=sum(r['horizon'] == HORIZONS[0] for r in rows))
    groups = [('all', lambda r: True), ('moving', lambda r: r['moving'])]
    groups += [(f'kind={k}', (lambda k: lambda r: r['kind'] == k)(k)) for k in sorted({r['kind'] for r in rows})]
    for name, keep in groups:
        for h in HORIZONS:
            sel = [r for r in rows if r['horizon'] == h and keep(r)]
            report[f'{name} {h/10:.1f}s'] = dict(count=len(sel), position_m=round(float(np.median([r['position'] for r in sel])), 4),
                                                 heading_rad=round(float(np.median([r['heading'] for r in sel])), 4))
    report['rule_m'] = RULE_M
    report['kinematic_model_used'] = report['all 3.2s']['position_m'] <= RULE_M
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--split', default='lewm_val')
    p.add_argument('--every', type=int, default=8)
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    report = measure(a.corpus, a.split, a.every, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
