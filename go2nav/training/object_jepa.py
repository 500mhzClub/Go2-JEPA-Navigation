"""Training the object-level predictor (`models.object_jepa`; docs/dynamic/PLAN.md P22) on P20's corpus and the ring
tracker's objects (`data.object_tracks`).

    python -m go2nav.training.object_jepa train --corpus CORPUS --out DIR [--config FILE] [--set key=value ...]

**Windows** (all on the GPU): `history` steps of 0.4 s ending at the start step t0, then 2 steps after it. Objects are
the slots whose track at t0 is the same track at the step (so a slot reused by a new track is another object); a
target counts where that track is seen at the target step. Every window's slots are shuffled.

**Loss:** the squared error of each object's predicted position at the step after every history step
(teacher-forced), plus that of the steps after it rolled out on the model's own predictions, to `rollout` steps after
t0 (2 by default, Terver et al.; run M4 takes 6, since the measures read 16), each object's walls looked up again at
its predicted position. With `distance` (M5 on) the error is the distance, not its square. With probability `mask_probability` a window masks one
object present through the history (C-JEPA's object masking; `mask_probability: 0` is run M1) in the teacher-forced
pass; the rollout starts from an unmasked pass (the first M0 rolled out the masked guess, which tripled its rollout
error).

**Validation** (`validate`, `DIR/validation.jsonl`): fixed val windows rolled out 16 steps (6.4 s) under the recorded
commands; the median error at 1.6, 3.2 and 6.4 s against the tracker's later positions of the same track, beside
carrying each object on at the tracker's velocity (M0 and M1's own validation files used its displacement over the
last step, a weaker baseline). Labels are not read: this is the training's own
check; PLAN P22's measures are `eval.object_measures`.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from go2nav.models.object_jepa import (OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING, STEP_FRAMES, STEP_S,
                                       ObjectConfig, ObjectJEPA, actions_input, object_features, offsets,
                                       robot_features)

DEFAULTS = dict(seed=2026111104, model={}, data=dict(batch=512, mask_probability=.5, rollout=2, distance=False),
                optimiser=dict(lr=3e-4, weight_decay=1e-3, warmup=500, steps=20000, grad_clip=1.),
                runtime=dict(log_every=100, validate_every=2000, checkpoint_every=1000, validation_windows=8192))
VALIDATION_STEPS, REPORT = 16, (4, 8, 16)


def merged(base, extra):
    out = dict(base)
    for key, value in extra.items():
        out[key] = merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def load_config(path, overrides=()):
    config = merged(DEFAULTS, yaml.safe_load(Path(path).read_text()) if path else {})
    for item in overrides:
        key, _, value = item.partition('=')
        node = config
        *parents, last = key.split('.')
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return config


class Rasters:
    """Every layout's wall raster on the device, padded to one size, looked up bilinearly per window."""

    def __init__(self, folder, mazes, device):
        from go2nav.planning.wall_raster import WallRaster
        loaded = {m: WallRaster.load(Path(folder)/f'{m:04d}.npz') for m in mazes}
        self.index = {m: k for k, m in enumerate(sorted(loaded))}
        height = max(r.distance.shape[0] for r in loaded.values())
        width = max(r.distance.shape[1] for r in loaded.values())
        grid = np.zeros((len(loaded), height, width), np.float32)
        low, shape = np.zeros((len(loaded), 2)), np.zeros((len(loaded), 2))
        for m, k in self.index.items():
            r = loaded[m]
            grid[k, :r.distance.shape[0], :r.distance.shape[1]] = r.distance
            low[k], shape[k] = r.low, r.distance.shape
        self.step = loaded[next(iter(loaded))].step
        self.grid = torch.as_tensor(grid, device=device)
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.shape = torch.as_tensor(shape, dtype=torch.long, device=device)

    def lookup(self, layout, points):
        """The walls' distance at points (B, ..., 2) of layouts (B,) (indices into this set)."""
        lead = points.shape[1:-1]
        p = points.reshape(points.shape[0], -1, 2)
        q = (p-self.low[layout][:, None])/self.step
        limit = (self.shape[layout]-2)[:, None]
        i = torch.minimum(torch.clamp(torch.floor(q).long(), min=0), limit)
        f = (q-i).clamp(0., 1.)
        g = self.grid[layout]
        rows = torch.arange(len(layout), device=points.device)[:, None]
        a, b = g[rows, i[..., 0], i[..., 1]], g[rows, i[..., 0]+1, i[..., 1]]
        c, d = g[rows, i[..., 0], i[..., 1]+1], g[rows, i[..., 0]+1, i[..., 1]+1]
        out = (a*(1-f[..., 0])+b*f[..., 0])*(1-f[..., 1])+(c*(1-f[..., 0])+d*f[..., 0])*f[..., 1]
        return out.reshape(points.shape[0], *lead)


def rotate(v, yaw):
    """Vectors (..., 2) turned by yaw (...)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c*v[..., 0]-s*v[..., 1], s*v[..., 0]+c*v[..., 1]), -1)


class Split:
    """One split's frames on the device: the robot's pose and requests, the objects and each frame's layout."""

    def __init__(self, corpus, split, rasters, device, frames):
        corpus = Path(corpus)
        packed = corpus/'packed'
        self.pose = torch.as_tensor(np.load(packed/f'{split}_pose.npy'), device=device)
        self.requested = torch.as_tensor(np.load(packed/f'{split}_requested.npy'), device=device)
        starts = np.load(packed/f'{split}_starts.npy')
        files = sorted((corpus/split).glob('*.npz'))
        xy, seen, ids, layout, velocity, memory = [], [], [], [], [], []
        for k, f in enumerate(files):
            with np.load(corpus/'objects'/split/f.name) as data:
                xy.append(data['xy']), seen.append(data['seen']), ids.append(data['id'])
                velocity.append(data['velocity'])
                memory.append(data['memory'] if 'memory' in data.files else np.zeros((*data['seen'].shape, 4), np.float32))
            maze = json.loads(f.with_suffix('.json').read_text())['maze']
            layout.append(np.full(starts[k+1]-starts[k], rasters.index[maze], np.int64))
        self.xy = torch.as_tensor(np.concatenate(xy), device=device)
        self.velocity = torch.as_tensor(np.concatenate(velocity), device=device)
        self.memory = torch.as_tensor(np.concatenate(memory), device=device)
        self.seen = torch.as_tensor(np.concatenate(seen), device=device)
        self.id = torch.as_tensor(np.concatenate(ids), device=device)
        self.layout = torch.as_tensor(np.concatenate(layout), device=device)
        self.windows = torch.as_tensor(np.concatenate([np.arange(a, b-frames+1) for a, b in zip(starts[:-1], starts[1:])
                                                       if b-a >= frames]), device=device)
        self.rasters, self.device = rasters, device

    def window(self, starts, history, after, generator=None, object_grid=(OBJECT_GRID, OBJECT_SPACING)):
        """Tokens of windows starting at `starts` (B,): history steps, then `after` steps. Returns a dict of tensors.
        `object_grid`: the size and spacing of the walls' grid round each object."""
        steps = history+after
        frame = starts[:, None]+STEP_FRAMES*torch.arange(steps, device=self.device)                    # (B, T)
        t0 = frame[:, history-1]
        origin, heading = self.pose[t0, :2], self.pose[t0, 2]
        layout = self.layout[t0]
        to_body = lambda world: rotate(world-origin[:, None, None], -heading[:, None, None])
        world = self.xy[frame]                                                                           # (B, T, S, 2)
        same = (self.id[frame] == self.id[t0][:, None]) & (self.id[t0][:, None] >= 0)
        seen = self.seen[frame] & same
        position = to_body(world)
        velocity = rotate(self.velocity[frame].float(), -heading[:, None, None])*seen[..., None]         # the tracker's
        memory = self.memory[frame].float()*same[..., None]
        grid = rotate(offsets(*object_grid, self.device), heading[:, None])                              # (B, G, 2)
        walls = self.rasters.lookup(layout, torch.nan_to_num(world)[..., None, :]+grid[:, None, None])
        robot_world, robot_yaw = self.pose[frame, :2], self.pose[frame, 2]
        robot_grid = rotate(offsets(ROBOT_GRID, ROBOT_SPACING, self.device)[None, None], robot_yaw[..., None])
        robot_walls = self.rasters.lookup(layout, robot_world[:, :, None]+robot_grid)
        robot_xy = rotate(robot_world-origin[:, None], -heading[:, None])
        robot_rel = robot_yaw-heading[:, None]
        commands = self.requested[frame[:, :-1, None]+torch.arange(STEP_FRAMES, device=self.device)]   # (B, T-1, 4, 3)
        if generator is not None:                                       # shuffle the slots: identity, not place
            order = torch.argsort(torch.rand(*same.shape[::2], generator=generator, device=self.device), -1)
            take = lambda x: torch.gather(x, 2, order[:, None, :, None].expand(-1, x.shape[1], -1, *x.shape[3:])
                                          if x.dim() == 4 else order[:, None, :].expand(-1, x.shape[1], -1))
            position, walls, same, seen = take(position), take(walls), take(same), take(seen)
            velocity, memory = take(velocity), take(memory)
        return dict(position=position, walls=walls, present=same, seen=seen, velocity=velocity, memory=memory,
                    robot=robot_features(robot_xy, robot_rel, robot_walls), actions=actions_input(commands),
                    layout=layout, origin=origin, heading=heading)

    def object_walls(self, w, position):
        """The walls round objects at body-frame positions (B, S, 2) of windows `w`."""
        world = rotate(position, w['heading'][:, None])+w['origin'][:, None]
        grid = rotate(offsets(OBJECT_GRID, OBJECT_SPACING, self.device), w['heading'][:, None])
        return self.rasters.lookup(w['layout'], world[..., None, :]+grid[:, None])


class Branches(Split):
    """One split's branched windows (`data.branches`) on the device in `Split`'s layout: each branch a segment of its
    own whose one window starts at the segment's start, so the branch point is the history's last step. The tracker's
    velocity and memory are held in half precision (a segment is mostly branch frames: about 10 GB for the train
    split in full precision would be 13)."""

    def __init__(self, corpus, split, rasters, device, frames, episodes=None, folder='branches'):
        """`episodes`: only the episodes numbered below it (a fixed part of a build still running); `folder`: the
        corpus's folder of them (`data.planner_branches` writes 'planner_branches')."""
        from go2nav.data.branches import SEGMENT
        if frames != SEGMENT:
            raise ValueError(f'branched windows are {SEGMENT} frames; this model reads {frames}')
        corpus = Path(corpus)
        parts = {k: [] for k in ('pose', 'requested', 'xy', 'seen', 'id', 'velocity', 'memory', 'layout')}
        for f in sorted((corpus/folder/split).glob('[0-9][0-9][0-9][0-9][0-9].npz')):
            if episodes is not None and int(f.stem) >= episodes:
                continue
            with np.load(f) as d:
                b = len(d['t0'])
                if not b:
                    continue
                parts['requested'].append(np.concatenate((d['requested'], np.zeros((b, 1, 3), np.float32)), 1)
                                          .reshape(b*SEGMENT, 3))
                for k in ('pose', 'xy', 'seen', 'id'):
                    parts[k].append(d[k].reshape(b*SEGMENT, *d[k].shape[2:]))
                for k in ('velocity', 'memory'):
                    parts[k].append(d[k].reshape(b*SEGMENT, *d[k].shape[2:]).astype(np.float16))
            maze = json.loads((corpus/split/f'{f.stem}.json').read_text())['maze']
            parts['layout'].append(np.full(b*SEGMENT, rasters.index[maze], np.int64))
        for k, v in parts.items():
            setattr(self, k, torch.as_tensor(np.concatenate(v), device=device))
        self.windows = torch.arange(0, len(self.pose), SEGMENT, device=device)
        self.rasters, self.device = rasters, device


def joined(a, b):
    """Two windows' dicts as one batch."""
    return {k: torch.cat((a[k], b[k])) for k in a}


def losses(model, split, w, history, masked, rollout=2, distance=False):
    """Teacher-forced and `rollout`-step rollout errors over the counted objects: squared (metres squared), or with
    `distance` the distance itself (metres), so that far steps and rare large misses do not carry the loss."""
    cost = (lambda e: (e.pow(2).sum(-1)+1e-8).sqrt()) if distance else (lambda e: e.pow(2).sum(-1))
    position, present, seen = w['position'], w['present'], w['seen']
    residual = model.config.residual
    speed = w['velocity'][:, :history] if residual else torch.zeros_like(position[:, :history])
    objects = object_features(position[:, :history], seen[:, :history], present[:, :history], w['walls'][:, :history],
                              speed if residual else None)
    delta = model(objects, w['robot'][:, :history], w['actions'][:, :history], masked)
    shown = torch.nan_to_num(position[:, :history])
    base, carried = shown, speed
    if masked is not None:                       # a masked object is predicted from its first step, carried on from there
        steps = torch.arange(history, device=shown.device)[None, :, None, None]
        late = masked[:, None, :, None] & (steps > 0)
        base = torch.where(late, shown[:, :1]+speed[:, :1]*STEP_S*steps, shown)
        carried = torch.where(late, speed[:, :1].expand_as(speed), speed)
    predicted = base+model.advance(delta, carried)
    target, count = torch.nan_to_num(position[:, 1:history+1]), seen[:, 1:history+1] & present[:, :history]
    teacher = (cost(predicted-target)*count).sum()/count.sum().clamp(min=1)
    if masked is not None and masked.any():                                     # the rollout starts unmasked: masking
        clean = model(objects, w['robot'][:, :history], w['actions'][:, :history])  # is for the history (C-JEPA)
        predicted = shown+model.advance(clean, speed)
    at = predicted[:, -1]                                                        # step t0+1, from the true history
    alive = present[:, history-1]
    moving = (at-shown[:, -1])/STEP_S if residual else None
    window, error, counted = objects, 0., 0
    for k in range(1, rollout):                                                  # steps t0+2 .. t0+rollout, on its own
        new = object_features(at, alive, alive, split.object_walls(w, at), moving)
        window = torch.cat((window[:, 1:], new[:, None]), 1)
        move = model.advance(model(window, w['robot'][:, k:k+history], w['actions'][:, k:k+history])[:, -1], moving)
        at = at+move
        if residual:
            moving = move/STEP_S
        count_k = seen[:, history+k] & alive
        error = error+(cost(at-torch.nan_to_num(position[:, history+k]))*count_k).sum()
        counted = counted+count_k.sum()
    roll = error/torch.as_tensor(counted, device=at.device).clamp(min=1)
    return teacher+roll, dict(teacher=teacher, rollout=roll)


@torch.no_grad()
def validate(model, split, starts, history, batch=2048):
    """See the module docstring: median errors (m) at each reported step, the model's and the last-step velocity's."""
    model.eval()
    errors = {k: [] for k in REPORT}
    carried = {k: [] for k in REPORT}
    for i in range(0, len(starts), batch):
        w = split.window(starts[i:i+batch], history, VALIDATION_STEPS)
        h = history
        objects = object_features(w['position'][:, :h], w['seen'][:, :h], w['present'][:, :h], w['walls'][:, :h],
                                  w['velocity'][:, :h] if model.config.residual else None)
        predicted = model.rollout(objects, w['robot'][:, :h], w['actions'], w['robot'][:, h:],
                                  lambda p: split.object_walls(w, p), VALIDATION_STEPS)
        now = torch.nan_to_num(w['position'][:, h-1])
        velocity = w['velocity'][:, h-1]*STEP_S                                  # a step's worth of the tracker's velocity
        for k in REPORT:
            counted = w['seen'][:, h-1+k] & w['present'][:, h-1]
            truth = torch.nan_to_num(w['position'][:, h-1+k])
            errors[k].append((predicted[:, k-1]-truth).norm(dim=-1)[counted])
            carried[k].append((now+k*velocity-truth).norm(dim=-1)[counted])
    model.train()
    out = {}
    for k in REPORT:
        out[f'model@{k*.4:.1f}s'] = round(float(torch.cat(errors[k]).median()), 4)
        out[f'velocity@{k*.4:.1f}s'] = round(float(torch.cat(carried[k]).median()), 4)
        out[f'count@{k*.4:.1f}s'] = int(torch.cat(errors[k]).numel())
    return out


def schedule(warmup, total):
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    return factor


def train(corpus, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    model_config = ObjectConfig(**config['model'])
    model = ObjectJEPA(model_config).to(device)
    h = model_config.history
    run, optim, data = config['runtime'], config['optimiser'], config['data']
    mazes = sorted({json.loads(p.read_text())['maze'] for split in ('train', 'val')
                    for p in (Path(corpus)/split).glob('*.json')})
    rasters = Rasters(Path(corpus)/'objects'/'rasters', mazes, device)
    train_split = Split(corpus, 'train', rasters, device, STEP_FRAMES*(h-1+data['rollout'])+1)
    val_split = Split(corpus, 'val', rasters, device, STEP_FRAMES*(h-1+VALIDATION_STEPS)+1)
    pick = torch.as_tensor(np.random.default_rng(config['seed']).choice(len(val_split.windows),
                                                                        run['validation_windows'], replace=False))
    val_starts = val_split.windows[pick.to(device)]
    optimiser = torch.optim.AdamW(model.parameters(), lr=optim['lr'], weight_decay=optim['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(optim['warmup'], optim['steps']))
    step, checkpoint = 0, out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model']), optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
    log, vlog = (out/'train.jsonl').open('a'), (out/'validation.jsonl').open('a')
    generator = torch.Generator(device=device)
    sums, started, n = {}, time.time(), 0
    model.train()
    while step < optim['steps']:
        generator.manual_seed(config['seed']*1_000_003+step)
        starts = train_split.windows[torch.randint(len(train_split.windows), (data['batch'],), device=device,
                                                   generator=generator)]
        w = train_split.window(starts, h, data['rollout'], generator)
        masked = None
        if data['mask_probability'] > 0:
            through = w['present'][:, :h].all(1)                                         # (B, S)
            choice = torch.rand(through.shape, device=device, generator=generator)*through
            one = torch.nn.functional.one_hot(choice.argmax(-1), through.shape[1]).bool() & through
            masked = one & (torch.rand(len(starts), 1, device=device, generator=generator) < data['mask_probability'])
        total, terms = losses(model, train_split, w, h, masked, data['rollout'], data['distance'])
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim['grad_clip'])
        optimiser.step(), scheduler.step()
        step, n = step+1, n+1
        for key, value in terms.items():
            sums[key] = sums.get(key, 0.)+float(value)
        sums['grad_norm'] = sums.get('grad_norm', 0.)+float(norm)
        if step % run['log_every'] == 0:
            log.write(json.dumps(dict(step=step, lr=scheduler.get_last_lr()[0], s_per_step=round((time.time()-started)/n, 4),
                                      **{k: round(v/n, 6) for k, v in sums.items()}))+'\n')
            log.flush()
            sums, started, n = {}, time.time(), 0
        if step % run['validate_every'] == 0 or step == optim['steps']:
            vlog.write(json.dumps(dict(step=step, **validate(model, val_split, val_starts, h)))+'\n')
            vlog.flush()
        if step % run['checkpoint_every'] == 0 or step == optim['steps']:
            partial = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step, config=config), partial)
            partial.rename(checkpoint)
    torch.save(dict(model=model.state_dict(), config=model_config.to_dict(), step=step), out/'model.pt')
    return step


def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    model = ObjectJEPA(ObjectConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--config', type=Path)
    t.add_argument('--set', action='append', default=[])
    a = p.parse_args()
    print(train(a.corpus, a.out, load_config(a.config, a.set)))


if __name__ == '__main__':
    main()
