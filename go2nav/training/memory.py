"""Train the walker memory (docs/dynamic/PLAN.md, P9, step 3) on corpus v2, resumably, and score it as the baselines.

    python -m go2nav.training.memory cache ROOT --world-model outputs/lewm/go2_v2_walkers/lewm.pt --out outputs/memory/states
    python -m go2nav.training.memory train ROOT --states outputs/memory/states --out outputs/memory/fixed
    python -m go2nav.training.memory evaluate ROOT --states outputs/memory/states --memory outputs/memory/fixed/memory.pt
    python -m go2nav.training.memory calibrate ROOT --states outputs/memory/states --memory outputs/memory/fixed/memory.pt

**Stage A, the fixed encoder.** `cache` encodes every second frame of each unit with labels (`data.memory_labels`)
once with the world model (bf16, its states (13, 192) as float16, `STATES/<role>/<unit>.npy`), and `train` fits the
memory (`models.memory`) on those states: batches of `batch` sequences of `length` steps (5 Hz) from random units
and starts, the first `burn_in` steps unscored, binary cross-entropy on every cell of the labelled grid at every
later step. The pose change between steps comes from the corpus's true poses, with Gaussian noise of
`pose_noise_m` and `yaw_noise_rad` per step standing in for the visual odometry's error. AdamW, warm-up then cosine
decay, bf16 autocast; every `checkpoint_every` steps the model, optimiser, scheduler and step are written atomically
and a rerun continues from there (the batches are a function of the seed and the step).

**Evaluation** (`evaluate`, and at the end of training): the memory runs over each validation unit from its first
step, and its occupancy at each labelled frame is scored as `eval.memory_baselines` scores the baselines (average
precision on the cells out of view, the rear zone, walkers that do not yield, and the cells in view), with the
baselines on the same frames, and on the frames of `eval.memory_probe` (12.8 s of history and on) beside the
probe's sets.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from go2nav.data import lewm_corpus as v1
from go2nav.data import memory_labels as ml
from go2nav.eval import memory_baselines as mb
from go2nav.models.memory import STEP_FRAMES, MemoryConfig, WalkerMemory, load_memory, save_memory

DEFAULTS = dict(seed=2026110601, batch=32, length=100, burn_in=10, steps=12000, lr=3e-4, weight_decay=1e-2,
                warmup=500, grad_clip=1., pose_noise_m=.01, yaw_noise_rad=.005, checkpoint_every=1000, log_every=100)
MEMORY_FRACTION = .85


def units(root, role):
    return sorted(f for f in (Path(root)/'units'/role).iterdir() if (f/ml.FILE).exists())


def cache(root, world_model, out, device='cuda'):
    from go2nav.models.lewm import load_world_model, normalise_frames
    model = load_world_model(world_model, device)
    for role in ('lewm_val', 'lewm_train'):
        (out/role).mkdir(parents=True, exist_ok=True)
        folders = units(root, role)
        start = time.time()
        for n, folder in enumerate(folders):
            target = out/role/f'{folder.name}.npy'
            if target.exists():
                continue
            frames = np.load(folder/'frames.npy', mmap_mode='r')
            t = np.arange(0, len(frames), STEP_FRAMES)
            states = np.zeros((len(t), model.config.tokens, model.config.latent), np.float16)
            with torch.inference_mode():
                for s in range(0, len(t), 450):
                    x = torch.from_numpy(np.ascontiguousarray(frames[t[s:s+450]])).to(device)
                    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                        states[s:s+450] = model.encode(normalise_frames(x)[:, None])[:, 0].float().cpu().numpy()
            tmp = out/role/f'{folder.name}.tmp.npy'
            np.save(tmp, states)
            os.replace(tmp, target)
            if (n+1) % 200 == 0:
                print(f'{role}: {n+1} of {len(folders)} units, {time.time()-start:.0f} s', flush=True)


def motions(pose):
    """The pose of each step in the body frame of the step before (steps, 3: x, y, yaw); zero at the first."""
    p = pose[::STEP_FRAMES]
    yaw = np.array([float(v1.planar_yaw(q[3:7])) for q in p])
    d = np.diff(p[:, :2], axis=0)
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    out = np.zeros((len(p), 3), np.float32)
    out[1:, 0], out[1:, 1] = c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1]
    out[1:, 2] = (np.diff(yaw)+np.pi) % (2*np.pi)-np.pi
    return out


class Units:
    """Every unit's cached states (memory-mapped), step motions and labels, for drawing training sequences."""

    def __init__(self, root, states, role):
        self.items = []
        for folder in units(root, role):
            path = Path(states)/role/f'{folder.name}.npy'
            if not path.exists():
                continue
            with np.load(folder/ml.FILE) as d:
                occupied = d['occupied']
            with np.load(folder/'steps.npz') as steps:
                motion = motions(steps['pose'])
            self.items.append((path, len(occupied), motion, occupied))   # states opened per draw: 4,000 open maps exceed the file limit

    def batch(self, rng, size, length):
        states, motion, labels = [], [], []
        long = [i for i, item in enumerate(self.items) if item[1] >= length]    # a unit can end early (a stop)
        for _ in range(size):
            path, n, m, o = self.items[long[rng.integers(len(long))]]
            start = int(rng.integers(0, n-length+1))
            states.append(np.array(np.load(path, mmap_mode='r')[start:start+length]))
            motion.append(m[start:start+length])
            labels.append(o[start:start+length])
        return (torch.from_numpy(np.stack(states)), torch.from_numpy(np.stack(motion)),
                torch.from_numpy(np.stack(labels)))


def run_memory(model, states, motion, device, keep=None):
    """The memory over whole sequences (B, steps, ...): occupancy logits at every step (B, steps, CELLS, CELLS)."""
    memory = model.initial(len(states), device)
    out = []
    for k in range(states.size(1)):
        memory = model.step(memory, states[:, k].float(), motion[:, k])
        out.append(model.occupancy(memory))
    return torch.stack(out, 1)


def train(root, states, out, config, device='cuda'):
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, torch.device(device).index or 0)
    data = Units(root, states, 'lewm_train')
    print(f'{len(data.items)} training units', flush=True)
    torch.manual_seed(config['seed'])
    model = WalkerMemory(MemoryConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), config['lr'], weight_decay=config['weight_decay'])
    warm, total = config['warmup'], config['steps']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s+1)/warm)*.5*(1+math.cos(math.pi*min(s, total)/total)))
    step = 0
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimiser'])
        sched.load_state_dict(state['scheduler'])
        step = state['step']
        print(f'resumed at step {step}', flush=True)
    log = open(out/'metrics.jsonl', 'a')
    start, losses = time.time(), []
    model.train()
    while step < total:
        rng = np.random.default_rng([config['seed'], step])
        s, m, y = data.batch(rng, config['batch'], config['length'])
        noise = rng.normal(size=m.shape).astype(np.float32)*np.array([config['pose_noise_m']]*2+[config['yaw_noise_rad']], np.float32)
        s, m, y = s.to(device), (m+torch.from_numpy(noise)).to(device), y.to(device).float()
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits = run_memory(model, s, m, device)
        b = config['burn_in']
        loss = F.binary_cross_entropy_with_logits(logits[:, b:].float(), y[:, b:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        opt.step()
        sched.step()
        step += 1
        losses.append(float(loss))
        if step % config['log_every'] == 0:
            log.write(json.dumps(dict(step=step, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0],
                                      s_per_step=(time.time()-start)/len(losses)))+'\n')
            log.flush()
            print(f'step {step} loss {np.mean(losses):.5f}', flush=True)
            start, losses = time.time(), []
        if step % config['checkpoint_every'] == 0 or step == total:
            tmp = out/'checkpoint.tmp.pt'
            torch.save(dict(model=model.state_dict(), optimiser=opt.state_dict(), scheduler=sched.state_dict(), step=step), tmp)
            os.replace(tmp, checkpoint)
    save_memory(out/'memory.pt', model, extra=dict(training=config))
    return out/'memory.pt'


def evaluate(root, states, memory_path, out, device='cuda', probe=None):
    """Score the memory against the baselines on the validation units (see the module docstring)."""
    model = load_memory(memory_path, device)
    prior = mb.prior_rates(root, None)
    from go2nav.eval.memory_probe import OFFSETS, EVERY
    parts = {view: [] for view in ('out_of_view', 'in_view', 'out_of_view_probe_frames', 'in_view_probe_frames')}
    for folder in units(root, 'lewm_val'):
        path = Path(states)/'lewm_val'/f'{folder.name}.npy'
        if not path.exists():
            continue
        grids = mb.unit_grids(folder, prior)
        with np.load(folder/'steps.npz') as steps:
            motion = motions(steps['pose'])
        s = torch.from_numpy(np.load(path))[None].to(device)
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits = run_memory(model, s, torch.from_numpy(motion)[None].to(device), device)
        grids['scores']['memory'] = torch.sigmoid(logits[0].float()).flatten(1).cpu().numpy()[:len(grids['t'])]
        rows = np.flatnonzero(grids['t'] >= max(OFFSETS))[::EVERY]
        for view, out_view in (('out_of_view', True), ('in_view', False)):
            parts[view].append(cells_with(grids, None, out_view))
            parts[f'{view}_probe_frames'].append(cells_with(grids, rows, out_view))
    names = ('memory', *mb.BASELINES)
    result = {}
    for view, data in parts.items():
        merged = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
        zones = dict(all=np.ones(len(merged['label']), bool), rear=merged['rear'], do_not_yield=merged['rude'],
                     rear_do_not_yield=merged['rear'] & merged['rude'])
        result[view] = {zone: {n: round(mb.average_precision(merged[n][m], merged['label'][m]), 4) for n in names}
                        for zone, m in zones.items()}
        if view == 'out_of_view':
            age = merged['age']
            result[view]['ap_by_age'] = {label: {n: round(mb.average_precision(merged[n][~merged['label'] | sel],
                                                                               merged['label'][~merged['label'] | sel]), 4)
                                                 for n in ('memory', 'constant_velocity', 'hold')}
                                         for label, _, _ in mb.AGE_BINS
                                         for sel in [mb.in_bin(age, label)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))
    return result


def calibrate(root, states, memory_path, device='cuda', units_n=200, precision=.5, seed=0):
    """The missions' threshold (PLAN P9, step 4): the lowest whose precision on the cells out of view of `units_n`
    training units is at least `precision`. Written into the memory checkpoint as `threshold`."""
    model = load_memory(memory_path, device)
    folders = [f for f in units(root, 'lewm_train') if (Path(states)/'lewm_train'/f'{f.name}.npy').exists()]
    rng = np.random.default_rng(seed)
    scores, labels = [], []
    for i in sorted(rng.choice(len(folders), min(units_n, len(folders)), replace=False)):
        folder = folders[i]
        with np.load(folder/ml.FILE) as d:
            occupied, visible = d['occupied'].reshape(len(d['t']), -1), d['visible'].reshape(len(d['t']), -1)
        with np.load(folder/'steps.npz') as steps:
            motion = motions(steps['pose'])
        s = torch.from_numpy(np.load(Path(states)/'lewm_train'/f'{folder.name}.npy'))[None].to(device)
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            logits = run_memory(model, s, torch.from_numpy(motion)[None].to(device), device)
        p = torch.sigmoid(logits[0].float()).flatten(1).cpu().numpy()[:len(occupied)]
        scores.append(p[~visible])
        labels.append(occupied[~visible])
    score, label = np.concatenate(scores), np.concatenate(labels)
    order = np.argsort(-score, kind='stable')
    hits = np.cumsum(label[order])
    precisions = hits/np.arange(1, len(order)+1)
    ok = np.flatnonzero(precisions >= precision)
    threshold = float(score[order][ok[-1]]) if len(ok) else 1.
    claimed = score >= threshold
    detail = dict(threshold=round(threshold, 4), precision=round(float(label[claimed].mean()), 4) if claimed.any() else None,
                  recall=round(float(claimed[label].mean()), 4), units=len(scores))
    state = torch.load(memory_path, map_location='cpu', weights_only=False)
    state['threshold'], state['calibration'] = threshold, detail
    tmp = Path(memory_path).with_suffix('.tmp.pt')
    torch.save(state, tmp)
    os.replace(tmp, memory_path)
    print(json.dumps(detail))
    return detail


def cells_with(grids, rows, out):
    """`memory_baselines.cells` with the memory's scores too."""
    rows_ = slice(None) if rows is None else rows
    d = mb.cells(grids, rows, out)
    mask = ~grids['visible'][rows_] if out else grids['visible'][rows_]
    d['memory'] = grids['scores']['memory'][rows_][mask]
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache')
    c.add_argument('root', type=Path)
    c.add_argument('--world-model', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('train')
    t.add_argument('root', type=Path)
    t.add_argument('--states', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--set', action='append', default=[], help='KEY=VALUE over the defaults')
    e = sub.add_parser('evaluate')
    e.add_argument('root', type=Path)
    e.add_argument('--states', type=Path, required=True)
    e.add_argument('--memory', type=Path, required=True)
    e.add_argument('--out', type=Path)
    k = sub.add_parser('calibrate')
    k.add_argument('root', type=Path)
    k.add_argument('--states', type=Path, required=True)
    k.add_argument('--memory', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'cache':
        cache(a.root, a.world_model, a.out)
    elif a.command == 'train':
        config = dict(DEFAULTS)
        for item in a.set:
            key, value = item.split('=', 1)
            config[key] = type(DEFAULTS[key])(value)
        path = train(a.root, a.states, a.out, config)
        evaluate(a.root, a.states, path, a.out/'evaluation.json')
    elif a.command == 'calibrate':
        calibrate(a.root, a.states, a.memory)
    else:
        evaluate(a.root, a.states, a.memory, a.out or a.memory.parent/'evaluation.json')


if __name__ == '__main__':
    main()
