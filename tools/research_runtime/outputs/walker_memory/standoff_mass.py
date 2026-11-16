"""P22 item 10 follow-up: the stand-off windows of standoff.py, read through every future rather than the likeliest.
Per model: the probability on futures that move more than MOVE_M by 1.6 and 3.2 s, the probability-weighted
displacement, and the walker's memory features at t0 (for N3). argv: model specs (modes:PATH)."""
import copy, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CORPUS = Path('/mnt/workspace_drive/go2nav_ring/corpus_v1'); MAZES = Path('/mnt/workspace_drive/go2nav_lewm_v2/mazes')
H = {16: 4, 32: 8}               # frames -> model steps
MOVE_M = .15

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
        knots, p, _ = model._futures(xy, yaw, tape)
        knots, p = knots[0, k], p[0, k]                                       # (modes, steps+1, 2), (modes,)
        moved = {h: np.linalg.norm(knots[:, s]-knots[:, 0], axis=-1) for h, s in H.items()}
        out[spec] = dict(mass={h: float(p[moved[h] > MOVE_M].sum()) for h in H},
                         expected={h: float((p*moved[h]).sum()) for h in H},
                         memory=now['memory'][k].tolist())
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
    caps = (30., 20., 20., 3.)
    for kind in ('polite', 'rude', 'reactive'):
        for lo, hi in ((0, 2), (2, 5), (5, 1e9)):
            sel = [r for r in rows if r['kind'] == kind and lo <= r['waited'] < hi]
            if not sel: continue
            share = {h: np.mean([r['true'][h] > MOVE_M for r in sel]) for h in H}
            line = f"{kind:9s} {lo}-{hi if hi < 1e9 else 'more'} s ({len(sel):3d}): truly moves {share[16]:.2f}, {share[32]:.2f}"
            for s in specs:
                name = Path(s.split(':')[1]).parent.name
                line += (f" | {name} p(move) {np.mean([r[s]['mass'][16] for r in sel]):.2f}, {np.mean([r[s]['mass'][32] for r in sel]):.2f}"
                         f" exp {np.mean([r[s]['expected'][16] for r in sel]):.2f}, {np.mean([r[s]['expected'][32] for r in sel]):.2f}")
            m = np.array([r[specs[-1]]['memory'] for r in sel])*caps
            line += f" | memory median age {np.median(m[:, 0]):.1f} s, still {np.median(m[:, 1]):.1f} s, near {np.median(m[:, 2]):.1f} s, walked {np.median(m[:, 3]):.2f} m"
            print(line)
    json.dump(rows, open('outputs/walker_memory/standoff_mass.json', 'w'))
