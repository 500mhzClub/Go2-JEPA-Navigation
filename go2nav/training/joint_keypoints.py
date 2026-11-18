"""P35 (docs/dynamic/PLAN.md P35): the locator and the world model trained together on the hard toy.

    python -m go2nav.training.joint_keypoints train --data ROOT --locator LOCATOR.pt --out DIR [--set k=v ...]

Each step adds two losses:
- the locator's own (`models.keypoints.Transporter`): the rebuilt frame of a pair 1-8 frames apart, plus the
  separation at `separation_scale` px, as P33's round 2;
- Q3's (`training.hard_tokens.hard_losses`) on windows of 3 history and `rollout` future frames whose obstacle tokens
  sit at the locator's keypoints, joined frame to frame by the nearest within `data.hard_tracks.GATE_PX` (the joins
  without gradient), the agent's by colour. The gradient passes through the crops and positions into the locator.
The locator starts from LOCATOR.pt; Q3 from scratch. bf16 autocast for Q3, float32 for the locator; AdamW 3e-4 (the
locator 1e-4), warm-up then cosine. At the end DIR/locator.pt (the keypoint network, as `training.keypoints` writes
it) and DIR/model.pt (Q3, as `training.hard_tokens` writes it, its `estimator` 'joint').
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from go2nav.data import hard_tracks as ht
from go2nav.data import reactive_balls as data
from go2nav.models.keypoints import KeypointConfig, Transporter, separation
from go2nav.models.lewm import SIGReg
from go2nav.models.object_tokens import ObjectTokenConfig, ObjectTokenJEPA, detect
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule, to_pixels
from go2nav.training.hard_tokens import HISTORY, hard_losses
from go2nav.training.recipe_bench import recalibrate, save

DEFAULTS = dict(steps=8000, batch=32, pairs=64, rollout=4, lr=3e-4, locator_lr=1e-4, weight_decay=1e-3, warmup=300,
                grad_clip=1., sigreg_weight=.09, sigreg_projections=128, jepa_weight=1., separation_weight=.01,
                separation_scale=2., gap=(1, 8), log_every=250, checkpoint_every=2000, memory_fraction=.4,
                recalibration_batches=100)


def nearest(a, b, gate):
    """Keypoints a, b (N, K, 2): each of a to the nearest of b within `gate` (N, K), -1 if none (no gradient)."""
    d = torch.cdist(a.detach().float(), b.detach().float())
    best, which = d.min(-1)
    return torch.where(best < gate, which, torch.full_like(which, -1))


def chained(kp, h):
    """Keypoints of a window's frames (B, T, K, 2) to each t0 keypoint's position at every frame (B, T, K, 2), and
    whether its chain reaches that frame (B, T, K); t0 is frame h. Unseen history frames hold the next seen position."""
    b, t, k, _ = kp.shape
    index = torch.full((b, t, k), -1, dtype=torch.long, device=kp.device)
    index[:, h] = torch.arange(k, device=kp.device)
    follow = lambda table, prev: torch.where(prev >= 0, torch.gather(table, 1, prev.clamp(min=0)), prev)
    for s in range(h+1, t):
        index[:, s] = follow(nearest(kp[:, s-1], kp[:, s], ht.GATE_PX), index[:, s-1])
    for s in range(h-1, -1, -1):
        index[:, s] = follow(nearest(kp[:, s+1], kp[:, s], ht.GATE_PX), index[:, s+1])
    seen = index >= 0
    pos = torch.gather(kp, 2, index.clamp(min=0)[..., None].expand(-1, -1, -1, 2))
    out = [pos[:, s] for s in range(t)]
    for s in range(h-1, -1, -1):
        out[s] = torch.where(seen[:, s, :, None], out[s], out[s+1])
    return torch.stack(out, 1), seen


def window_batch(transporter, frames, actions, e, t0, after):
    """Q3's batch over the locator's keypoints, with the gradient into them."""
    steps = t0[:, None]+torch.arange(1-HISTORY, after+1, device=frames.device)
    pixels = to_pixels(frames[e[:, None], steps])                                            # (B, T, 3, 64, 64)
    flat = pixels.flatten(0, 1)
    kp = transporter.keypoints(flat).unflatten(0, pixels.shape[:2])                          # (B, T, K, 2)
    pos, seen = chained(kp, HISTORY-1)
    with torch.no_grad():
        agent = detect(flat)[0][:, 0].unflatten(0, pixels.shape[:2])
    located = torch.cat((agent[:, :, None], pos), 2)
    seen = torch.cat((torch.ones_like(seen[:, :, :1]), seen), 2)
    return dict(pixels=pixels, actions=actions[e[:, None], steps[:, :-1]], located=located, seen=seen, valid=seen)


def train(root, locator, out, overrides=None, seed=0):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111935+seed
    torch.manual_seed(run_seed)
    state = torch.load(locator, map_location='cpu', weights_only=False)
    config = KeypointConfig(**state['config'])
    transporter = Transporter(config).to(device)
    transporter.load_state_dict(state['transporter'])
    model = ObjectTokenJEPA(ObjectTokenConfig(residual=True, tokens=1+config.keypoints)).to(device)
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    frames_np, labels, _ = data.load(root, 'interactive', 'train', 't2', mmap=False)
    frames = torch.from_numpy(frames_np).to(device)
    actions = torch.from_numpy(labels['actions']).to(device)/world.MAX_ACTION
    episodes, length = frames.shape[:2]
    groups = [dict(params=list(model.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay']),
              dict(params=list(transporter.parameters()), lr=cfg['locator_lr'], weight_decay=0.)]
    optimiser = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], cfg['steps']))
    params = groups[0]['params']+groups[1]['params']
    meta = dict(recipe='p35_joint', kind='objects', estimator='joint', seed=seed, train=cfg, locator_start=str(locator))
    (out/'config.json').write_text(json.dumps(dict(**meta, model=asdict(model.config), locator=asdict(config)), indent=1))

    def objective(index, offset=0):
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+offset+index)
        draw = lambda high, n: torch.randint(0, high, (n,), generator=g, device=device)
        # the locator's own loss on pairs
        gap = torch.randint(cfg['gap'][0], cfg['gap'][1]+1, (cfg['pairs'],), generator=g, device=device)
        pe = draw(episodes, cfg['pairs'])
        ps = (torch.rand(cfg['pairs'], generator=g, device=device)*(length-gap)).long()
        source, target = to_pixels(frames[pe, ps]), to_pixels(frames[pe, ps+gap])
        rebuilt, kp = transporter(source, target, config.sigma)
        terms = dict(rebuild=F.mse_loss(rebuilt, target), separation=separation(kp, cfg['separation_scale']))
        # Q3's loss over the locator's keypoints
        e, t0 = draw(episodes, cfg['batch']), HISTORY-1+draw(length-cfg['rollout']-HISTORY+1, cfg['batch'])
        b = window_batch(transporter, frames, actions, e, t0, cfg['rollout'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            jepa = hard_losses(model, b, rollout_steps=cfg['rollout'], sigreg=sigreg, lam=cfg['sigreg_weight'], generator=g)
        terms.update(jepa)
        terms['jepa'] = jepa['loss']
        terms['total'] = terms['rebuild']+cfg['separation_weight']*terms['separation']+cfg['jepa_weight']*jepa['loss']
        return terms

    model.train(), transporter.train()
    log, started, step = (out/'metrics.jsonl').open('a'), time.time(), 0
    while step < cfg['steps']:
        terms = objective(step)
        optimiser.zero_grad(set_to_none=True)
        terms['total'].backward()
        grad = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        if torch.isfinite(terms['total']):
            optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            log.write(json.dumps(dict(step=step, grad=float(grad), steps_per_s=round(cfg['log_every']/(time.time()-started), 2),
                                      **{k: round(float(v.detach()), 6) for k, v in terms.items()}))+'\n')
            log.flush()
            started = time.time()
    transporter.eval()
    recalibrate(model, lambda i: objective(i, offset=5_000_000), cfg['recalibration_batches'])
    torch.save(dict(config=asdict(config), keypoints=transporter.keypoints.state_dict(), transporter=transporter.state_dict(),
                    step=step), out/'locator.pt')
    save(out/'model.pt', model, meta, dict(step=step))
    print(f'done: {out} at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--locator', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.locator, args.out, overrides)


if __name__ == '__main__':
    main()
