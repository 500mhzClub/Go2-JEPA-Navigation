"""P22, memory: in stand-offs (a walker waiting on the robot at t0, labels only to choose), the exact future under a
hold tape against each model's likeliest prediction under the same tape, by the walker's kind and by how long it had
been waiting. argv: model specs (objects:PATH or modes:PATH)."""
import copy, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CORPUS = Path('/mnt/workspace_drive/go2nav_ring/corpus_v1'); MAZES = Path('/mnt/workspace_drive/go2nav_lewm_v2/mazes')
H = (16, 32)

def window(args):
    index, t0, j, waited, specs = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch; torch.set_num_threads(1)
    from go2nav.data import ring_corpus as rc
    from go2nav.eval.object_measures import _setup, _matched
    from go2nav.planning.sampling import integrate
    meta = json.loads((CORPUS/'val'/f'{index:05d}.json').read_text())
    with np.load(CORPUS/'val'/f'{index:05d}.npz') as d:
        ranges, pose, requested = d['ranges'].astype(np.float32), d['pose'], d['requested']
    world = rc.replay_world(MAZES, meta, requested, t0)
    tape = np.zeros((1, 64, 3), np.float32)
    branch = copy.deepcopy(world); path = []
    for f in range(32):
        branch.step(tape[0, f]); path.append(branch.walkers_xy()[j])
    start = world.walkers_xy()[j]
    out = dict(kind=meta['kind'], waited=waited, true={h: float(np.linalg.norm(path[h-1]-start)) for h in H})
    xy, yaw = integrate(dict(xy=world.xy, yaw=world.yaw, executed=world.executed), tape)
    for spec in specs:
        geometry, fields, model, _ = _setup(meta, spec)
        for t in range(t0+1):
            model.observe(t*.1, (ranges[t], ranges[t] > 0), pose[t, :2], float(pose[t, 2]), geometry)
            if t < t0: model.record(requested[t])
        now = model.frames[-1]; k = _matched(now['xy'], now['seen'], start)
        if k is None: return None
        pred = model.top(None, xy, yaw, None, geometry, requested=tape)[0][:, k]
        out[spec] = {h: float(np.linalg.norm(pred[h-1]-start)) for h in H}
    return out

if __name__ == '__main__':
    specs = sys.argv[1:]; rng = np.random.default_rng(0); jobs = []
    for p in sorted((CORPUS/'val').glob('*.json')):
        meta = json.loads(p.read_text())
        if meta['walkers'] == 0: continue
        with np.load(p.with_suffix('.npz')) as d: waiting = d['waiting']
        run = np.zeros_like(waiting, int)
        for t in range(1, len(waiting)): run[t] = np.where(waiting[t], run[t-1]+1, 0)
        cand = [(t, j) for t in range(8, len(waiting)-64) for j in np.flatnonzero(waiting[t]) if t % 5 == 0]
        for t, j in (rng.permutation(cand)[:4] if cand else []):
            jobs.append((int(p.stem), int(t), int(j), float(run[t, j]*.1), specs))
    with ProcessPoolExecutor(26) as pool: rows = [r for r in pool.map(window, jobs) if r]
    print(len(rows), 'stand-off windows of', len(jobs))
    for kind in ('polite', 'rude', 'reactive'):
        for lo, hi in ((0, 2), (2, 5), (5, 1e9)):
            sel = [r for r in rows if r['kind'] == kind and lo <= r['waited'] < hi]
            if not sel: continue
            line = f"{kind:9s} waited {lo}-{hi if hi < 1e9 else 'more'} s ({len(sel):3d}): true moved 1.6 s {np.median([r['true'][16] for r in sel]):.2f} m, 3.2 s {np.median([r['true'][32] for r in sel]):.2f}"
            for s in specs:
                line += f" | {Path(s.split(':')[1]).parent.name}: {np.median([r[s][16] for r in sel]):.2f}, {np.median([r[s][32] for r in sel]):.2f}"
            print(line)
    json.dump(rows, open('outputs/walker_memory/standoff.json', 'w'))
