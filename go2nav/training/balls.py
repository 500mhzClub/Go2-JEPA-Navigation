"""P25's LeWM runs (docs/dynamic/PLAN.md P25 item 2), on `data.balls`: L0, LeWM as ported here at Zhu et al.'s n-ball
sizes, and L1, L0 with their Fourier head on the balls' boxes (the supervised reference).

    python -m go2nav.training.balls train --data ROOT --n 3 --model l0 --out DIR [--seed S] [--set key=value ...]

**Model** (`model_config`): `models.lewm.LeWorldModel` on 64 x 64 frames in [-1, 1]:
- encoder: patch 8, width 64, 4 blocks of 4 heads, MLP 256, [CLS] to a 128 latent;
- predictor: over 3 frames, 4 blocks of 4 heads of 16, MLP 256, at width 128 (the port has no input projection; the
  paper's predictor is 64 wide);
- an action is one step's (dx, dy) over 3 px.

**Objective,** the paper's (as `models.lewm.losses` with one step): the teacher-forced next-embedding MSE over windows
of 4 frames and 3 actions, plus 0.09 x SIGReg over the encoded frames. L1 adds 0.1 x the Fourier head's MSE on the
encoded states and 0.1 x on the predicted ones. Its target (`fourier_target`) is the sum over every ball of
[cos(W'b), sin(W'b)], b = (x, y, w, h) the ball's box in pixels and W (4 x 9) fixed Gaussian frequencies of deviation
0.15 rad/px (the paper does not list its own).

**Training:** AdamW 5e-5, weight decay 1e-3, 2,000 steps of warm-up then cosine, clip 1 (LeWM's official settings), in
float32, at batch 64, for 200 epochs. An epoch is every window of every training episode once (1,000 x 97), in an order
drawn from (seed, epoch), so the run is a pure function of its seed. The frames live on the GPU as uint8.

**Resume:** DIR/checkpoint.pt (model, head, optimiser, scheduler, step and RNG states) every `checkpoint_every`
steps, written atomically; rerunning the same command continues from it, and metrics.jsonl is cut back to it.
**Validation** every `validate_every` steps and at the end: the same terms on 4,096 fixed validation windows.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from go2nav.data import balls as data
from go2nav.models.lewm import LeWMConfig, LeWorldModel, SIGReg
from go2nav.sim import balls as world

WINDOW, HISTORY = 4, 3
FREQUENCIES, FOURIER_SCALE, FOURIER_SEED = 9, .15, 2026111502
DEFAULTS = dict(batch=64, epochs=200, lr=5e-5, weight_decay=1e-3, warmup=2000, grad_clip=1., sigreg_weight=.09,
                sigreg_projections=1024, aux_weight=.1, log_every=500, validate_every=20000, checkpoint_every=5000,
                validation_windows=4096, memory_fraction=.15)


def model_config():
    return LeWMConfig(image_hw=(world.SIZE, world.SIZE), patch=8, width=64, depth=4, heads=4, mlp=256, latent=128,
                      history=HISTORY, block_steps=1, command_dim=2, predictor_depth=4, predictor_heads=4,
                      predictor_head_dim=16, predictor_mlp=256)


class FourierHead(nn.Module):
    """L1's training-only head: a state (..., 128) to the spectral target (..., 18), a 128-64-18 MLP."""

    def __init__(self, latent=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 2*FREQUENCIES))

    def forward(self, z):
        return self.net(z)


def frequencies(device=None):
    g = torch.Generator().manual_seed(FOURIER_SEED)
    return (torch.randn(4, FREQUENCIES, generator=g)*FOURIER_SCALE).to(device)


def fourier_target(red, white, w):
    """The sum over the balls of [cos(W'b), sin(W'b)]: red (..., 2) and white (..., n, 2) centres to (..., 18)."""
    centres = torch.cat((red[..., None, :], white), -2)
    size = torch.full_like(centres, 2*world.RADIUS)
    phase = torch.cat((centres, size), -1)@w
    return torch.cat((phase.cos(), phase.sin()), -1).sum(-2)


def to_pixels(frames):
    """uint8 (..., 64, 64, 3) to float (..., 3, 64, 64) in [-1, 1]."""
    return frames.float().div(127.5).sub(1.).movedim(-1, -3)


class Windows:
    """A split on the GPU and its windows of 4 frames: `batch(episodes, starts)` gives the pixels (B, 4, 3, 64, 64), the
    actions (B, 3, 2) over 3 px, and the red (B, 4, 2) and white (B, 4, n, 2) centres."""

    def __init__(self, root, n, split, device):
        frames, labels = data.load(root, n, split, mmap=False)
        self.frames = torch.from_numpy(frames).to(device)
        self.labels = {k: torch.from_numpy(v).to(device) for k, v in labels.items()}
        self.episodes, self.length = frames.shape[:2]
        self.starts = self.length-WINDOW+1

    def batch(self, episodes, starts):
        steps = starts[:, None]+torch.arange(WINDOW, device=starts.device)
        e = episodes[:, None]
        return (to_pixels(self.frames[e, steps]), self.labels['actions'][e, steps[:, :HISTORY]]/world.MAX_ACTION,
                self.labels['red'][e, steps], self.labels['white'][e, steps])


def terms(model, head, sigreg, w, pixels, actions, red, white, cfg, generator=None):
    z = model.encode(pixels)
    pred = model.predict(z[:, :HISTORY], actions)
    out = dict(pred=(pred-z[:, 1:]).pow(2).mean(), sigreg=sigreg(z.float().transpose(0, 1), generator))
    out['loss'] = out['pred']+cfg['sigreg_weight']*out['sigreg']
    if head is not None:
        target = fourier_target(red, white, w)
        out['aux_encoded'] = (head(z)-target).pow(2).mean()
        out['aux_predicted'] = (head(pred)-target[:, 1:]).pow(2).mean()
        out['loss'] = out['loss']+cfg['aux_weight']*(out['aux_encoded']+out['aux_predicted'])
    return out


def schedule(warmup, total):
    def f(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    return f


def build(kind, seed, device):
    torch.manual_seed(seed)
    model = LeWorldModel(model_config()).to(device)
    head = FourierHead().to(device) if kind == 'l1' else None
    return model, head


def save(path, model, head, kind, n, cfg, extra=None):
    state = dict(config=asdict(model.config), model_state_dict=model.state_dict(), kind=kind, n=n, train=cfg,
                 head_state_dict=None if head is None else head.state_dict(), **(extra or {}))
    partial = Path(str(path)+'.partial')
    torch.save(state, partial)
    os.replace(partial, path)


def load(path, device='cpu'):
    """A trained run's (model, head or None, checkpoint dict), in eval mode."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(state['config'])
    config['image_hw'], config['grid'] = tuple(config['image_hw']), tuple(config.get('grid', ()))
    model = LeWorldModel(LeWMConfig(**config))
    model.load_state_dict(state['model_state_dict'])
    head = None
    if state.get('head_state_dict') is not None:
        head = FourierHead()
        head.load_state_dict(state['head_state_dict'])
        head = head.to(device).eval()
    return model.to(device).eval(), head, state


@torch.no_grad()
def validate(model, head, sigreg, w, val, cfg, device):
    model.eval()
    if head is not None:
        head.eval()
    g = torch.Generator(device=device).manual_seed(7)
    count = cfg['validation_windows']
    episodes = torch.randint(0, val.episodes, (count,), generator=g, device=device)
    starts = torch.randint(0, val.starts, (count,), generator=g, device=device)
    sums, size = {}, cfg['batch']     # SIGReg's statistic grows with the batch: validate at the training batch
    for i in range(0, count, size):
        batch = val.batch(episodes[i:i+size], starts[i:i+size])
        for k, v in terms(model, head, sigreg, w, *batch, cfg, torch.Generator(device=device).manual_seed(i)).items():
            sums[k] = sums.get(k, 0.)+float(v)*len(batch[0])
    model.train()
    if head is not None:
        head.train()
    return {k: v/count for k, v in sums.items()}


def train(root, n, kind, out, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111500+seed
    model, head = build(kind, run_seed, device)
    params = list(model.parameters())+([] if head is None else list(head.parameters()))
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    w = frequencies(device)
    trainset, val = Windows(root, n, 'train', device), Windows(root, n, 'val', device)
    per_epoch = trainset.episodes*trainset.starts//cfg['batch']
    total = cfg['epochs']*per_epoch
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], total))
    step, checkpoint, metrics = 0, out/'checkpoint.pt', out/'metrics.jsonl'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        if head is not None:
            head.load_state_dict(state['head_state_dict'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        step = state['step']
        if metrics.exists():
            rows = [r for r in metrics.read_text().splitlines() if json.loads(r).get('step', 0) <= step]
            metrics.write_text(''.join(r+'\n' for r in rows))
    (out/'config.json').write_text(json.dumps(dict(kind=kind, n=n, seed=seed, run_seed=run_seed, train=cfg,
                                                   model=asdict(model.config), total_steps=total), indent=1))
    model.train()
    log, started, epoch, order = metrics.open('a'), time.time(), None, None
    while step < total:
        if step//per_epoch != epoch:
            epoch = step//per_epoch
            g = torch.Generator(device=device).manual_seed(run_seed*1000+epoch)
            order = torch.randperm(trainset.episodes*trainset.starts, generator=g, device=device)
        index = order[(step % per_epoch)*cfg['batch']:(step % per_epoch+1)*cfg['batch']]
        batch = trainset.batch(index//trainset.starts, index % trainset.starts)
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+step)
        out_terms = terms(model, head, sigreg, w, *batch, cfg, g)
        optimiser.zero_grad(set_to_none=True)
        out_terms['loss'].backward()
        grad = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        if torch.isfinite(out_terms['loss']):
            optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            row = dict(step=step, epoch=epoch, lr=scheduler.get_last_lr()[0], grad=float(grad),
                       steps_per_s=round(cfg['log_every']/(time.time()-started), 1),
                       **{k: round(float(v), 6) for k, v in out_terms.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            started = time.time()
        if step % cfg['validate_every'] == 0 or step == total:
            row = dict(step=step, validation=validate(model, head, sigreg, w, val, cfg, device))
            log.write(json.dumps(row)+'\n')
            log.flush()
        if step % cfg['checkpoint_every'] == 0 or step == total:
            save(checkpoint, model, head, kind, n, cfg, dict(optimiser=optimiser.state_dict(),
                 scheduler=scheduler.state_dict(), step=step, rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state()))
    save(out/'model.pt', model, head, kind, n, cfg, dict(step=step))
    print(f'done: {out}/model.pt at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--n', type=int, required=True)
    t.add_argument('--model', choices=['l0', 'l1'], required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.n, args.model, args.out, args.seed, overrides)


if __name__ == '__main__':
    main()
