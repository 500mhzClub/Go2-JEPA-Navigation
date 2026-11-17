"""P30's locator (docs/dynamic/PLAN.md P30): `models.keypoints.Transporter` trained on pairs of T2 frames.

    python -m go2nav.training.keypoints train --data ROOT --out DIR [--variant t2] [--seed S] [--set k=v ...]

A step draws `batch` pairs (s, t) of frames 1-8 apart in the interactive training split, from (seed, step); the loss
is the mean squared pixel error of the rebuilt frame t, plus `separation_weight` x `models.keypoints.separation`. The masks' deviation falls linearly from `sigma_start` to
`sigma` over the first `anneal` of the steps, then stays. Adam with warm-up then cosine, float32, on the GPU. DIR/
checkpoint.pt every `checkpoint_every` steps (rerunning resumes). At the end the stage-1 measures are taken
(`models.keypoints.measures`, on 20,000 training frames for the assignment and offsets and on every validation frame),
and DIR/locator.pt holds the keypoint network, the keypoint of each object (`order`), the offsets and the measures;
DIR/measures.json the measures.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from go2nav.data import reactive_balls as data
from go2nav.models.keypoints import KeypointConfig, Transporter, assign, measures, separation
from go2nav.training.balls import schedule, to_pixels

DEFAULTS = dict(steps=10000, batch=64, lr=1e-3, warmup=500, gap=(1, 8), keypoints=4, sigma=2., log_every=250,
                checkpoint_every=2000, memory_fraction=.3, policy='interactive', assignment_frames=20000,
                separation_weight=.01, sigma_start=8., anneal=.5)


@torch.no_grad()
def locate(net, frames, device, batch=2048):
    """Keypoints (N, K, 2) of uint8 frames (N, 64, 64, 3) (numpy or a tensor)."""
    out = []
    for i in range(0, len(frames), batch):
        part = torch.as_tensor(np.asarray(frames[i:i+batch])).to(device)
        out.append(net(to_pixels(part)).cpu().numpy())
    return np.concatenate(out)


def positions(labels):
    return np.concatenate((labels['agent'][:, :, None], labels['obstacle']), 2)               # (E, T, 4, 2)


def score(net, root, variant, policy, device, frames=20000, seed=0):
    """Assignment and offsets on training frames, measures on every validation frame."""
    rng = np.random.default_rng(2026111890+seed)
    train_frames, train_labels, _ = data.load(root, policy, 'train', variant)
    e, t = rng.integers(0, train_frames.shape[0], frames), rng.integers(0, train_frames.shape[1], frames)
    order_rows = np.lexsort((t, e))
    e, t = e[order_rows], t[order_rows]
    kp = locate(net, train_frames[e, t], device)
    truth = positions(train_labels)[e, t]
    order = assign(kp, truth)
    offsets = (truth-kp[:, order]).mean(0)
    val_frames, val_labels, _ = data.load(root, policy, 'val', variant)
    vkp = locate(net, val_frames.reshape(-1, *val_frames.shape[2:]), device)
    out = measures(vkp, positions(val_labels).reshape(-1, 4, 2), order, offsets)
    out['train_corrected'] = measures(kp, truth, order, offsets)['corrected']
    return order, offsets, out


def train(root, out, variant='t2', seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111880+seed
    torch.manual_seed(run_seed)
    config = KeypointConfig(keypoints=cfg['keypoints'], sigma=cfg['sigma'])
    model = Transporter(config).to(device)
    frames = torch.from_numpy(data.load(root, cfg['policy'], 'train', variant, mmap=False)[0]).to(device)
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
    (out/'config.json').write_text(json.dumps(dict(train=cfg, model=asdict(config), variant=variant, seed=seed,
                                                   run_seed=run_seed), indent=1))
    log, started = metrics.open('a'), time.time()
    model.train()
    while step < cfg['steps']:
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+step)
        e = torch.randint(0, episodes, (cfg['batch'],), generator=g, device=device)
        gap = torch.randint(cfg['gap'][0], cfg['gap'][1]+1, (cfg['batch'],), generator=g, device=device)
        s = (torch.rand(cfg['batch'], generator=g, device=device)*(length-gap)).long()
        source, target = to_pixels(frames[e, s]), to_pixels(frames[e, s+gap])
        sigma = cfg['sigma']+(cfg['sigma_start']-cfg['sigma'])*max(0., 1-step/(cfg['anneal']*cfg['steps']))
        rebuilt, kp = model(source, target, sigma)
        rebuild, apart = F.mse_loss(rebuilt, target), separation(kp)
        loss = rebuild+cfg['separation_weight']*apart
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            log.write(json.dumps(dict(step=step, loss=round(float(loss.detach()), 6), rebuild=round(float(rebuild.detach()), 6),
                                      separation=round(float(apart.detach()), 4), sigma=round(sigma, 2), lr=scheduler.get_last_lr()[0],
                                      steps_per_s=round(cfg['log_every']/(time.time()-started), 1)))+'\n')
            log.flush()
            started = time.time()
        if step % cfg['checkpoint_every'] == 0 and step < cfg['steps']:
            partial = checkpoint.with_suffix('.partial')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step), partial)
            os.replace(partial, checkpoint)
    del frames
    model.eval()
    order, offsets, result = score(model.keypoints, root, variant, cfg['policy'], device, cfg['assignment_frames'], seed)
    torch.save(dict(config=asdict(config), keypoints=model.keypoints.state_dict(), transporter=model.state_dict(),
                    order=torch.as_tensor(order), offsets=torch.as_tensor(offsets), measures=result, step=step),
               out/'locator.pt')
    (out/'measures.json').write_text(json.dumps(dict(order=order.tolist(), offsets=offsets.tolist(), **result), indent=1))
    print(json.dumps(result['corrected']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--variant', choices=['t1', 't2'], default='t2')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.out, args.variant, args.seed, overrides)


if __name__ == '__main__':
    main()
