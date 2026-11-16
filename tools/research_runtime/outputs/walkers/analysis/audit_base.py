"""Base rates behind the pose-loss finding, the C1/C5 comparison, and which cohorts cost what."""
import collections
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path('outputs/cohorts')


def rate(run):
    run = Path(run)
    trace = np.load(run/'native/physics_trace.npz')
    obs = np.load(run/'native/obstacle_trace.npz')
    n = min(len(trace['base_pose_world']), len(obs['xy']))
    p = trace['base_pose_world'][:n, :2][::50]
    w = obs['xy'][:n][::50]
    d = np.linalg.norm(w-p[:, None], axis=2).min(1)
    cmd = trace['requested_command'][:n][::50]
    turning = (np.abs(cmd[:, 2]) > .1) & (np.abs(cmd[:, 0]) < .02)
    close = d < .8
    return float(close.mean()), float((close & turning).mean()), float(turning.mean()), len(d)


def main():
    runs = {}
    for name, cohorts in dict(two=['wh_obst2_c5', 'wo_obst2_c5', 'ws_obst2_c5'], one=['wh_obst1_c5', 'wo_obst1_c5', 'ws_obst1_c5'],
                              four=['wf_held_c5', 'wf_ref_c5']).items():
        runs[name] = [r for c in cohorts for r in sorted((ROOT/c/'runs').iterdir())
                      if (r/'native/obstacle_trace.npz').exists() and (r/'native/physics_trace.npz').exists()]
    with ProcessPoolExecutor(14) as pool:
        for name, rs in runs.items():
            out = list(pool.map(rate, rs, chunksize=4))
            w = np.array([o[3] for o in out], float)
            a = np.array([o[:3] for o in out])
            m = (a*w[:, None]).sum(0)/w.sum()
            print(f'{name} walkers, {len(rs)} missions: time with a walker centre within 0.8 m {100*m[0]:.1f}%; '
                  f'and turning in place {100*m[1]:.1f}%; turning in place at all {100*m[2]:.1f}%')

    def table(cohorts):
        out = {}
        for c in cohorts:
            for row in json.loads((ROOT/c/'result.json').read_text())['rows']:
                m = re.search(r'_(\d+)_ep(\d)$', row['assignment'])
                out[(re.search(r'obst(\d)', c).group(1), m.group(1), m.group(2))] = row
        return out

    def compare(a, b, label):
        A, B = table(a), table(b)
        k = set(A) & set(B)
        print(f"{label}: {sum(A[x]['round_trip'] for x in k)} -> {sum(B[x]['round_trip'] for x in k)} of {len(k)}; gained "
              f"{sum(B[x]['round_trip'] and not A[x]['round_trip'] for x in k)} lost {sum(A[x]['round_trip'] and not B[x]['round_trip'] for x in k)}")

    print()
    compare(['cuf_obst1_c1', 'cuf_obst2_c1'], ['ws_obst1_c5', 'ws_obst2_c5'], 'C1 ridge -> C5 world-model forecast, same planner, hand tracker')
    compare(['tp_lat_obst1_c1', 'tp_lat_obst2_c1'], ['cuf_obst1_c1', 'cuf_obst2_c1'], 'mover trails -> close-up with fix (C1)')
    print()
    for c in ('wo_obst1_c5', 'wo_obst2_c5', 'wf_held_c5', 'cuf_obst1_c1'):
        walls = [json.loads(l)['wall_s'] for r in (ROOT/c/'runs').iterdir() if (r/'progress.jsonl').exists()
                 for l in (r/'progress.jsonl').read_text().splitlines()[-1:]]
        sims = [json.loads(l)['simulated_s'] for r in (ROOT/c/'runs').iterdir() if (r/'progress.jsonl').exists()
                for l in (r/'progress.jsonl').read_text().splitlines()[-1:]]
        print(c, 'mean wall s per mission', round(np.mean(walls)), 'mean simulated s', round(np.mean(sims)), 'realtime factor', round(np.sum(sims)/np.sum(walls), 2))


if __name__ == '__main__':
    main()
