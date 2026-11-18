"""P37's detector (docs/dynamic/PLAN.md P37): `models.shared_detector.MaskTransporter` trained on pairs of the hard
toy's frames.

    python -m go2nav.training.shared_detector train --data ROOT --out DIR [--seed S] [--set k=v ...]

A step draws `batch` pairs (s, t) of frames 1-8 apart in the interactive training split, from (seed, step), as P30's
locator; the loss is the mean squared pixel error of the rebuilt frame t plus `mask_weight` times the masks' mean (of
s's and t's). Adam with warm-up then cosine, float32, on the GPU. DIR/checkpoint.pt every `checkpoint_every` steps
(rerunning resumes); at the end DIR/detector.pt holds the mask network and its config. No labels are read.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from go2nav.data import reactive_balls as data
from go2nav.models.keypoints import KeypointConfig
from go2nav.models.shared_detector import MaskTransporter
from go2nav.training.balls import schedule, to_pixels

DEFAULTS = dict(steps=10000, batch=64, lr=1e-3, warmup=500, gap=(1, 8), mask_weight=.02, log_every=250,
                checkpoint_every=2000, memory_fraction=.3, policy='interactive')


def train(root, out, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111937+seed
    torch.manual_seed(run_seed)
    config = KeypointConfig(keypoints=1)
    model = MaskTransporter(config).to(device)
    frames = torch.from_numpy(data.load(root, cfg['policy'], 'train', 't2', mmap=False)[0]).to(device)
    episodes, length = frames.shape[:2]
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], cfg['steps']))
    step, checkpoint, metrics = 0, out/'checkpoint.pt', out/'metrics.jsonl'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        if metrics.exists():
            rows = [r for r in metrics.read_text().splitlines() if json.loads(r).get('step', 0) <= step]
            metrics.write_text(''.join(r+'\n' for r in rows))
    (out/'config.json').write_text(json.dumps(dict(train=cfg, model=asdict(config), seed=seed, run_seed=run_seed), indent=1))
    log, started = metrics.open('a'), time.time()
    model.train()
    while step < cfg['steps']:
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+step)
        e = torch.randint(0, episodes, (cfg['batch'],), generator=g, device=device)
        gap = torch.randint(cfg['gap'][0], cfg['gap'][1]+1, (cfg['batch'],), generator=g, device=device)
        s = (torch.rand(cfg['batch'], generator=g, device=device)*(length-gap)).long()
        source, target = to_pixels(frames[e, s]), to_pixels(frames[e, s+gap])
        rebuilt, ms, mt = model(source, target)
        rebuild, cover = F.mse_loss(rebuilt, target), (ms.mean()+mt.mean())/2
        loss = rebuild+cfg['mask_weight']*cover
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            log.write(json.dumps(dict(step=step, loss=round(float(loss.detach()), 6), rebuild=round(float(rebuild.detach()), 6),
                                      mask_mean=round(float(cover.detach()), 5),
                                      mask_over_half=round(float((mt.detach() > .5).float().mean()), 5),
                                      lr=scheduler.get_last_lr()[0],
                                      steps_per_s=round(cfg['log_every']/(time.time()-started), 1)))+'\n')
            log.flush()
            started = time.time()
        if step % cfg['checkpoint_every'] == 0 and step < cfg['steps']:
            partial = checkpoint.with_suffix('.partial')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step), partial)
            os.replace(partial, checkpoint)
    torch.save(dict(config=asdict(config), mask=model.mask.state_dict(), transporter=model.state_dict(), step=step),
               out/'detector.pt')
    print(f'done: {out}/detector.pt at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.out, args.seed, overrides)


if __name__ == '__main__':
    main()
