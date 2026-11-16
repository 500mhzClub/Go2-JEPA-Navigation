"""P27's JEPA runs (docs/dynamic/PLAN.md P27 item 3): the recipe ladder on `data.reactive_balls`.

    python -m go2nav.training.recipe_bench train --data ROOT --variant t2 --recipe r1 --out DIR [--seed S] [--set k=v ...]

**Base** (every rung): `models.lewm.LeWorldModel` at `training.balls`'s L0 sizes (patch 8, width 64, 4 blocks of 4
heads; latent 128; a predictor of 4 blocks over 3 history frames, no dropout) on 64 x 64 frames in [-1, 1]; an action
is one step's (dx, dy) over 3 px. The objective is `models.lewm.losses` on windows of 3 + 8 frames: the teacher-forced
next-embedding MSE, the MSE of an 8-step rollout, and 0.09 x SIGReg.

**Recipes** (`RECIPES`): r0, one [CLS] latent; r1, every 8 x 8 patch token predicted (`grid=(8, 8)`: 65 tokens a frame).

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

import torch
from torch import nn

from go2nav.data import reactive_balls as data
from go2nav.models.lewm import LeWMConfig, LeWorldModel, SIGReg, losses
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule, to_pixels

HISTORY = 3
DEFAULTS = dict(steps=6000, batch=64, rollout=8, lr=3e-4, weight_decay=1e-3, warmup=500, grad_clip=1.,
                sigreg_weight=.09, sigreg_projections=128, bf16=True, log_every=250, checkpoint_every=2000,
                memory_fraction=.42, recalibration_batches=200, policy='interactive')
RECIPES = dict(r0=dict(), r1=dict(grid=(8, 8)))


def model_config(**overrides):
    base = dict(image_hw=(world.SIZE, world.SIZE), patch=8, width=64, depth=4, heads=4, mlp=256, latent=128,
                history=HISTORY, block_steps=1, command_dim=2, predictor_depth=4, predictor_heads=4,
                predictor_head_dim=16, predictor_mlp=256, predictor_dropout=0.)
    return LeWMConfig(**(base | overrides))


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


def save(path, model, meta, extra=None):
    state = dict(config=asdict(model.config), model_state_dict=model.state_dict(), **meta, **(extra or {}))
    partial = Path(str(path)+'.partial')
    torch.save(state, partial)
    os.replace(partial, path)


def load(path, device='cpu'):
    """A trained run's (model in eval mode, checkpoint dict)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(state['config'])
    config['image_hw'], config['grid'] = tuple(config['image_hw']), tuple(config.get('grid', ()))
    model = LeWorldModel(LeWMConfig(**config))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False), state


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
    model = LeWorldModel(model_config(**RECIPES[recipe])).to(device)
    params = list(model.parameters())
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    trainset = Windows(root, cfg['policy'], 'train', variant, device)
    length, total = HISTORY+cfg['rollout'], cfg['steps']
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], total))
    meta = dict(recipe=recipe, variant=variant, seed=seed, train=cfg)
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
        pixels, actions = trainset.draw(cfg['batch'], length, g)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            return losses(model, pixels, actions, rollout_steps=cfg['rollout'], sigreg=sigreg, lam=cfg['sigreg_weight'],
                          generator=g)

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
                       **{k: round(float(v), 6) for k, v in terms.items()})
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
