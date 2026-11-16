"""Review of 11 November (second): how predictable are the walkers in principle? Val windows of P20's corpus with a
walker in sight within 3 m; the episode is replayed exactly to the window and branched K times under the recorded
tape, each branch with fresh dice for every walker (the rules known, the draws not). For the nearest walker in sight:
the distance between two independent futures (what a sampler of the truth would be off by), the error of the futures'
mean (the best a deterministic predictor can do under squared error), and the error of carrying it on at its true
velocity over the last frame. Medians at 1.6, 3.2 and 6.4 s, over all windows and by walker kind."""
import copy, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CORPUS = Path('/mnt/workspace_drive/go2nav_ring/corpus_v1'); MAZES = Path('/mnt/workspace_drive/go2nav_lewm_v2/mazes')
K, H = 8, (16, 32, 64)

def window(args):
    index, t0 = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from go2nav.data import ring_corpus as rc
    meta = json.loads((CORPUS/'val'/f'{index:05d}.json').read_text())
    with np.load(CORPUS/'val'/f'{index:05d}.npz') as d:
        requested, sight, truth, pose = d['requested'], d['in_sight'], d['walkers_xy'], d['pose']
    world = rc.replay_world(MAZES, meta, requested, t0)
    assert np.allclose(pose[t0], [*world.xy, world.yaw], atol=1e-4), (index, t0)
    near = np.where(sight[t0][:len(world.walkers)], np.linalg.norm(world.walkers_xy()-world.xy, axis=1), np.inf)
    j = int(near.argmin())
    velocity, start = world.walkers_velocity[j].copy(), world.walkers_xy()[j].copy()
    futures = []
    for k in range(K):
        branch = copy.deepcopy(world)
        for n, w in enumerate(branch.walkers):
            w.rng = np.random.default_rng([index, t0, k, n])
        path = []
        for f in range(64):
            branch.step(requested[t0+f])
            if branch.contact is not None:
                break
            path.append(branch.walkers_xy()[j])
        if len(path) < 64:
            return None
        futures.append(np.array(path))
    futures = np.array(futures)                                    # (K, 64, 2)
    out = dict(kind=meta['kind'])
    for h in H:
        ends = futures[:, h-1]
        out[f'pair@{h}'] = float(np.linalg.norm(ends[0]-ends[1]))
        out[f'mean@{h}'] = float(np.linalg.norm(ends[0]-ends[1:].mean(0)))
        out[f'velocity@{h}'] = float(np.linalg.norm(ends[0]-(start+velocity*h*.1)))
        out[f'frozen@{h}'] = float(np.linalg.norm(ends[0]-start))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(0); jobs = []
    for p in sorted((CORPUS/'val').glob('*.json')):
        meta = json.loads(p.read_text())
        if meta['walkers'] == 0: continue
        with np.load(p.with_suffix('.npz')) as d: pose, truth, sight = d['pose'], d['walkers_xy'], d['in_sight']
        close = np.nanmin(np.where(sight, np.linalg.norm(truth-pose[:, None, :2], axis=-1), np.inf), axis=1)
        frames = [t for t in range(8, len(pose)-64) if close[t] <= 3.]
        for t in (rng.choice(frames, min(2, len(frames)), replace=False) if frames else []): jobs.append((int(p.stem), int(t)))
    with ProcessPoolExecutor(10) as pool: rows = [r for r in pool.map(window, jobs) if r]
    print(len(rows), 'windows of', len(jobs))
    for name, sel in [('all', rows)]+[(k, [r for r in rows if r['kind'] == k]) for k in ('polite', 'rude', 'reactive')]:
        print(name.ljust(9), len(sel), '  '.join(f"{h/10:.1f}s: two futures {np.median([r[f'pair@{h}'] for r in sel]):.2f}, best deterministic {np.median([r[f'mean@{h}'] for r in sel]):.2f}, true velocity {np.median([r[f'velocity@{h}'] for r in sel]):.2f}, frozen {np.median([r[f'frozen@{h}'] for r in sel]):.2f}" for h in H))
    json.dump(rows, open('outputs/review_1111/noise_floor.json', 'w'))
