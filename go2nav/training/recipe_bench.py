"""P27's JEPA runs (docs/dynamic/PLAN.md P27 item 3): the recipe ladder on `data.reactive_balls`.

    python -m go2nav.training.recipe_bench train --data ROOT --variant t2 --recipe r1 --out DIR [--seed S] [--set k=v ...]

**Base** (every rung): `models.lewm.LeWorldModel` at `training.balls`'s L0 sizes (patch 8, width 64, 4 blocks of 4
heads; latent 128; a predictor of 4 blocks over 3 history frames, no dropout) on 64 x 64 frames in [-1, 1]; an action
is one step's (dx, dy) over 3 px. The objective is `models.lewm.losses` on windows of 3 + 8 frames: the teacher-forced
next-embedding MSE, the MSE of an 8-step rollout, and 0.09 x SIGReg.

**Recipes** (`RECIPES`): r0, one [CLS] latent; r1, every 8 x 8 patch token predicted (`grid=(8, 8)`: 65 tokens a frame);
r2, r1 with each patch token's prediction error weighted by the change of its target patch's pixels (`motion_losses`);
r4, the object-token JEPA (`models.object_tokens`).

**Training:** AdamW, warm-up then cosine, clip 1, bf16 autocast (SIGReg in float32, 128 projections), on the GPU only;
6,000 steps at batch 64 (a patch-token step costs 0.2-0.35 s, and a run with its measures must fit 45 minutes). A step
draws its windows from (seed, step).
DIR/checkpoint.pt every `checkpoint_every` steps (rerunning the same command resumes). At the end the BatchNorm
projectors' running statistics are recomputed as a cumulative average over training windows (the handoff's trap), and
DIR/model.pt is written.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go2nav.data import reactive_balls as data
from go2nav.models.lewm import LeWMConfig, LeWorldModel, SIGReg, losses
from go2nav.models.object_tokens import ObjectTokenConfig, ObjectTokenJEPA, token_losses
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule, to_pixels

HISTORY = 3
DEFAULTS = dict(steps=6000, batch=64, rollout=8, lr=3e-4, weight_decay=1e-3, warmup=500, grad_clip=1.,
                sigreg_weight=.09, sigreg_projections=128, bf16=True, log_every=250, checkpoint_every=2000,
                memory_fraction=.42, recalibration_batches=200, policy='interactive')
RECIPES = dict(r0=dict(), r1=dict(model=dict(grid=(8, 8))), r2=dict(model=dict(grid=(8, 8)), motion_floor=.1),
               r4=dict(kind='objects'))


def model_config(**overrides):
    base = dict(image_hw=(world.SIZE, world.SIZE), patch=8, width=64, depth=4, heads=4, mlp=256, latent=128,
                history=HISTORY, block_steps=1, command_dim=2, predictor_depth=4, predictor_heads=4,
                predictor_head_dim=16, predictor_mlp=256, predictor_dropout=0.)
    return LeWMConfig(**(base | overrides))


def patch_motion(pixels, patch):
    """Frames (B, T, 3, H, W) to (B, T-1, G): the mean absolute change of each patch's pixels from frame t-1 to frame t,
    for t = 1..T-1, the patches in row-major order."""
    change = (pixels[:, 1:]-pixels[:, :-1]).abs().mean(2)
    return F.avg_pool2d(change, patch).flatten(2)


def motion_losses(model, pixels, actions, *, rollout_steps, sigreg, lam, floor, generator=None):
    """R2: `models.lewm.losses` for a patch-grid model (one grid token a patch), with each grid token's squared error
    weighted by w = floor + (1 - floor) x its target patch's pixel change over the batch's mean change (so the mean
    weight is 1): a token whose patch a disc enters or leaves counts for more than a still one. [CLS]'s terms and
    SIGReg are unchanged."""
    n, cls = model.config.history, model.cls
    z = model.encode(pixels)
    change = patch_motion(pixels.float(), model.config.patch)
    w = floor+(1-floor)*change/change.mean().clamp_min(1e-8)                 # (B, T-1, G): the weight of frame t's tokens
    pred, rolled = model.predict(z[:, :n], actions[:, :n]), model.rollout(z[:, :n], actions, rollout_steps)
    error = lambda out, first: ((out[:, :, 1:]-z[:, first:first+out.size(1), 1:]).float().pow(2).mean(-1)
                                * w[:, first-1:first-1+out.size(1)]).mean()
    terms = dict(pred=(cls(pred)-cls(z[:, 1:n+1])).pow(2).mean(), grid_pred=error(pred, 1),
                 rollout=(cls(rolled)-cls(z[:, n:n+rollout_steps])).pow(2).mean(), grid_rollout=error(rolled, n))
    terms['sigreg'] = (sigreg(cls(z).float().transpose(0, 1), generator)
                       + sigreg(z[:, :, 1:].float().flatten(1, 2).transpose(0, 1), generator))/2
    terms['loss'] = terms['pred']+terms['rollout']+terms['grid_pred']+terms['grid_rollout']+lam*terms['sigreg']
    return terms


class Windows:
    """A split on the GPU: `batch(episodes, starts, length)` gives the pixels (B, length, 3, 64, 64) and the actions
    (B, length-1, 2) over 3 px of windows starting at those frames."""

    def __init__(self, root, policy, split, variant, device):
        frames, labels, self.branches = data.load(root, policy, split, variant, mmap=False)
        self.frames = torch.from_numpy(frames).to(device)
        self.actions = torch.from_numpy(labels['actions']).to(device)/world.MAX_ACTION
        self.labels = labels
        self.episodes, self.length = frames.shape[:2]

    def batch(self, episodes, starts, length):
        steps = starts[:, None]+torch.arange(length, device=starts.device)
        e = episodes[:, None]
        return to_pixels(self.frames[e, steps]), self.actions[e, steps[:, :-1]]

    def draw(self, count, length, generator):
        device = self.frames.device
        episodes = torch.randint(0, self.episodes, (count,), generator=generator, device=device)
        starts = torch.randint(0, self.length-length+1, (count,), generator=generator, device=device)
        return self.batch(episodes, starts, length)


class BranchWindows:
    """The interactive training split's exact branches on the GPU (the first `steps` frames of each tape): `draw`
    gives, for `count` random moments, the history's pixels (B, 3, 3, 64, 64) and, for two different tapes of each, the
    future pixels (B, steps, 3, 64, 64) and the actions (B, 2 + steps, 2) from the first history frame on."""

    def __init__(self, root, variant, device, steps):
        self.windows = Windows(root, 'interactive', 'train', variant, device)
        frames = data.load(root, 'interactive', 'train', variant, branch_frames=True)[2]['frames']
        self.frames = torch.from_numpy(np.ascontiguousarray(frames[:, :, :, :steps])).to(device)
        branches = self.windows.branches
        self.t0 = torch.from_numpy(branches['t0'].astype(np.int64)).to(device)
        self.actions = torch.from_numpy(branches['actions'][:, :, :, :steps]).to(device)/world.MAX_ACTION

    def draw(self, count, generator):
        device = self.frames.device
        draw = lambda high: torch.randint(0, high, (count,), generator=generator, device=device)
        e, m, a = draw(self.frames.shape[0]), draw(self.frames.shape[1]), draw(len(world.TAPES))
        b = (a+1+draw(len(world.TAPES)-1)) % len(world.TAPES)
        frames = self.t0[e, m][:, None]+torch.arange(1-HISTORY, 1, device=device)
        history, past = to_pixels(self.windows.frames[e[:, None], frames]), self.windows.actions[e[:, None], frames[:, :-1]]
        return (history, [to_pixels(self.frames[e, m, x]) for x in (a, b)],
                [torch.cat((past, self.actions[e, m, x]), 1) for x in (a, b)])


def branch_losses(model, history, futures, tapes):
    """R3's terms on one history under two tapes: `branch`, the MSE of each tape's rollout against that tape's encoded
    true future; `paired`, the MSE of the predicted difference between the tapes against the true one,
    (pred_A - pred_B) against (z_A - z_B)."""
    zh, (za, zb) = model.encode(history), (model.encode(f) for f in futures)
    ra, rb = (model.rollout(zh, tape, za.size(1)) for tape in tapes)
    return dict(branch=((ra-za).pow(2).mean()+(rb-zb).pow(2).mean())/2, paired=((ra-rb)-(za-zb)).pow(2).mean())


def save(path, model, meta, extra=None):
    state = dict(config=asdict(model.config), model_state_dict=model.state_dict(), **meta, **(extra or {}))
    partial = Path(str(path)+'.partial')
    torch.save(state, partial)
    os.replace(partial, path)


def load(path, device='cpu'):
    """A trained run's (model in eval mode, checkpoint dict)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(state['config'])
    config['grid'] = tuple(config.get('grid', ()))
    if state.get('kind') == 'objects':
        model = ObjectTokenJEPA(ObjectTokenConfig(**config))
    else:
        config['image_hw'] = tuple(config['image_hw'])
        model = LeWorldModel(LeWMConfig(**config))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False), state


def build(recipe):
    options = RECIPES[recipe]
    if options.get('kind') == 'objects':
        return ObjectTokenJEPA(ObjectTokenConfig(**options.get('model', {})))
    return LeWorldModel(model_config(**options.get('model', {})))


@torch.no_grad()
def recalibrate(model, objective, batches):
    """The BatchNorm layers' running statistics as the cumulative average over `batches` training batches, each passed
    through the whole objective (so the predictor's projectors see teacher-forced and rolled-out inputs)."""
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i in range(batches):
        objective(i)
    model.eval()


def train(root, variant, recipe, out, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111600+seed
    torch.manual_seed(run_seed)
    options = RECIPES[recipe]
    model = build(recipe).to(device)
    params = list(model.parameters())
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    trainset = Windows(root, cfg['policy'], 'train', variant, device)
    branchset = BranchWindows(root, variant, device, cfg['rollout']) if 'branches' in options else None
    length, total = HISTORY+cfg['rollout'], cfg['steps']
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], total))
    meta = dict(recipe=recipe, variant=variant, seed=seed, train=cfg, kind=options.get('kind', 'lewm'))
    step, checkpoint, metrics = 0, out/'checkpoint.pt', out/'metrics.jsonl'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        if metrics.exists():
            rows = [r for r in metrics.read_text().splitlines() if json.loads(r).get('step', 0) <= step]
            metrics.write_text(''.join(r+'\n' for r in rows))
    (out/'config.json').write_text(json.dumps(dict(**meta, run_seed=run_seed, model=asdict(model.config)), indent=1))

    def objective(index, offset=0):
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+offset+index)
        pixels, actions = trainset.draw(cfg['batch']//2 if branchset else cfg['batch'], length, g)
        shared = dict(rollout_steps=cfg['rollout'], sigreg=sigreg, lam=cfg['sigreg_weight'], generator=g)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            if options.get('kind') == 'objects':
                terms = token_losses(model, pixels, actions, **shared)
            elif 'motion_floor' in options:
                terms = motion_losses(model, pixels, actions, floor=options['motion_floor'], **shared)
            else:
                terms = losses(model, pixels, actions, **shared)
            if branchset:       # R3: half the batch is windows; the other half is a quarter as many moments, 2 tapes each
                terms.update(branch_losses(model, *branchset.draw(cfg['batch']//4, g)))
                terms['loss'] = terms['loss']+terms['branch']+options['branches']['paired']*terms['paired']
            return terms

    model.train()
    log, started = metrics.open('a'), time.time()
    while step < total:
        terms = objective(step)
        optimiser.zero_grad(set_to_none=True)
        terms['loss'].backward()
        grad = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        if torch.isfinite(terms['loss']):
            optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            row = dict(step=step, lr=scheduler.get_last_lr()[0], grad=float(grad),
                       steps_per_s=round(cfg['log_every']/(time.time()-started), 1),
                       **{k: round(float(v.detach()), 6) for k, v in terms.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            started = time.time()
        if step % cfg['checkpoint_every'] == 0 and step < total:
            save(checkpoint, model, meta, dict(optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                                               step=step))
    recalibrate(model, lambda i: objective(i, offset=5_000_000), cfg['recalibration_batches'])
    save(out/'model.pt', model, meta, dict(step=step))
    print(f'done: {out}/model.pt at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--variant', choices=['t1', 't2'], required=True)
    t.add_argument('--recipe', choices=sorted(RECIPES), required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.variant, args.recipe, args.out, args.seed, overrides)


if __name__ == '__main__':
    main()
