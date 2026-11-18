"""P33 (docs/dynamic/PLAN.md P33): Q3 [object tokens, residual predictor] on the hard toy, over an engineered tracker or
learned keypoints (`data.hard_tracks`), trained identically.

    python -m go2nav.training.hard_tokens train --data ROOT --estimator blobs|keypoints --out DIR [--seed S] [--set k=v ...]

A token per object slot and frame: the agent (by its colour) first, then the `data.hard_tracks.SLOTS` obstacle slots
of the estimator's chains through the window (3 history frames, 8 after). The loss is Q3's on the tokens still followed
(`hard_losses`: the agent always; an obstacle slot where its chain is seen), with SIGReg on those tokens pooled. Windows
are drawn from (seed, step); bf16 autocast; AdamW 3e-4 with warm-up then cosine; batch 64. DIR/checkpoint.pt every
`checkpoint_every` steps (rerunning resumes); at the end the BatchNorm statistics are recomputed and DIR/model.pt written.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from go2nav.data import hard_tracks as ht
from go2nav.data import reactive_balls as data
from go2nav.models.lewm import SIGReg
from go2nav.models.object_tokens import ObjectTokenConfig, ObjectTokenJEPA
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule, to_pixels
from go2nav.training.recipe_bench import recalibrate, save

DEFAULTS = dict(steps=36000, batch=64, rollout=8, lr=3e-4, weight_decay=1e-3, warmup=500, grad_clip=1., sigreg_weight=.09,
                sigreg_projections=128, bf16=True, log_every=250, checkpoint_every=2000, memory_fraction=.35,
                recalibration_batches=200, memory=False)
HISTORY = 3


class HardWindows:
    """A split's frames and actions on the GPU and an estimator's tracks (numpy): `batch(e, t0, after)` gives the pixels
    (B, 3+after, 3, 64, 64), actions (B, 2+after, 2) over 3 px, positions (B, T, 1+S, 2), seen (B, T, 1+S) and the
    loss's validity (B, T, 1+S)."""

    def __init__(self, root, split, estimator, device):
        frames, self.labels, self.branches = data.load(root, 'interactive', split, 't2', mmap=False)
        self.frames = torch.from_numpy(frames).to(device)
        self.actions = torch.from_numpy(self.labels['actions']).to(device)/world.MAX_ACTION
        self.tracks = ht.load(root, split, estimator)
        self.episodes, self.length = frames.shape[:2]
        self.device = device

    def batch(self, e, t0, after):
        """e, t0 numpy (B,)."""
        pos, seen, present = ht.chain(self.tracks['xy'], self.tracks['ahead'], self.tracks['back'], e, t0,
                                      before=HISTORY-1, after=after)
        frames = t0[:, None]+np.arange(1-HISTORY, after+1)
        agent = self.tracks['agent'][e[:, None], frames]
        located = np.concatenate((agent[:, :, None], pos), 2)
        seen = np.concatenate((np.ones(seen.shape[:2]+(1,), bool), seen), 2)
        valid = seen & np.concatenate((np.ones((len(e), 1), bool), present), 1)[:, None]
        t = lambda x: torch.as_tensor(x, device=self.device)
        rows, steps = t(e)[:, None], t(frames)
        return dict(pixels=to_pixels(self.frames[rows, steps]), actions=self.actions[rows, steps[:, :-1]],
                    located=t(located).float(), seen=t(seen), valid=t(valid), e=e, t0=t0)


def hard_losses(model, b, *, rollout_steps, sigreg, lam, generator=None):
    n = HISTORY
    z = model.encode(b['pixels'], b['located'], b['seen'])
    pred = model.predict(z[:, :n], b['actions'][:, :n])
    rolled = model.rollout(z[:, :n], b['actions'], rollout_steps)
    w = b['valid'].float()
    mse = lambda out, target, m: ((out-target).float().pow(2).mean(-1)*m).sum()/m.sum().clamp(min=1)
    terms = dict(pred=mse(pred, z[:, 1:n+1], w[:, 1:n+1]), rollout=mse(rolled, z[:, n:n+rollout_steps], w[:, n:n+rollout_steps]))
    tokens = z.float()[b['valid']]
    size = len(z)
    order = torch.randperm(len(tokens), generator=generator, device=z.device)[:len(tokens)//size*size]
    terms['sigreg'] = sigreg(tokens[order].unflatten(0, (-1, size)), generator)
    terms['loss'] = terms['pred']+terms['rollout']+lam*terms['sigreg']
    return terms


def train(root, estimator, out, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111933+seed
    torch.manual_seed(run_seed)
    model = ObjectTokenJEPA(ObjectTokenConfig(residual=True, tokens=1+ht.SLOTS)).to(device)
    params = list(model.parameters())
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    windows = HardWindows(root, 'train', estimator, device)
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], cfg['steps']))
    meta = dict(recipe=f'p33_{estimator}', kind='objects', estimator=estimator, seed=seed, train=cfg)
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
        rng = np.random.default_rng((run_seed, offset, index))
        e = rng.integers(0, windows.episodes, cfg['batch'])
        t0 = rng.integers(HISTORY-1, windows.length-cfg['rollout'], cfg['batch'])
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+offset+index)
        b = windows.batch(e, t0, cfg['rollout'])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            return hard_losses(model, b, rollout_steps=cfg['rollout'], sigreg=sigreg, lam=cfg['sigreg_weight'], generator=g)

    model.train()
    log, started = metrics.open('a'), time.time()
    while step < cfg['steps']:
        terms = objective(step)
        optimiser.zero_grad(set_to_none=True)
        terms['loss'].backward()
        grad = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        if torch.isfinite(terms['loss']):
            optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            log.write(json.dumps(dict(step=step, lr=scheduler.get_last_lr()[0], grad=float(grad),
                                      steps_per_s=round(cfg['log_every']/(time.time()-started), 1),
                                      **{k: round(float(v.detach()), 6) for k, v in terms.items()}))+'\n')
            log.flush()
            started = time.time()
        if step % cfg['checkpoint_every'] == 0 and step < cfg['steps']:
            save(checkpoint, model, meta, dict(optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(), step=step))
    recalibrate(model, lambda i: objective(i, offset=5_000_000), cfg['recalibration_batches'])
    save(out/'model.pt', model, meta, dict(step=step))
    print(f'done: {out}/model.pt at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--estimator', choices=['blobs', 'keypoints', 'detector'], required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.estimator, args.out, args.seed, overrides)


if __name__ == '__main__':
    main()
