"""P31 stage 2 (docs/dynamic/PLAN.md P31 item 4): P28's model with Q3 on the ring corpus's windows, over the tracker's
objects or the learned keypoints (`data.ring_windows`).

    python -m go2nav.training.ring_windows train --corpus CORPUS --kind tracker|learned --out DIR
        [--keypoints DIR/keypoints_train.npy] [--seed S] [--set k=v ...]

Windows of 3 history steps and 8 future steps of 0.4 s from the recorded training episodes, drawn from (seed, step).
`models.ring_tokens.token_losses` (teacher-forced and 8-step rollout MSE on the tokens with a target, 0.09 x SIGReg),
the residual predictor, bf16 autocast, AdamW 3e-4 with warm-up then cosine, batch 64, 32,000 steps. The targets are
the model's own tokens of the future steps: no true positions enter training. DIR/checkpoint.pt every
`checkpoint_every` steps (rerunning resumes); at the end the BatchNorm statistics are recomputed, the objective is
logged once more in evaluation mode, and DIR/model.pt is written.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from go2nav.data.ring_windows import Frames
from go2nav.models.lewm import SIGReg
from go2nav.models.ring_tokens import RingTokenConfig, RingTokenJEPA, token_losses
from go2nav.training.balls import schedule
from go2nav.training.recipe_bench import recalibrate, save

DEFAULTS = dict(steps=32000, batch=64, future=8, lr=3e-4, weight_decay=1e-3, warmup=500, grad_clip=1.,
                sigreg_weight=.09, sigreg_projections=128, bf16=True, log_every=250, checkpoint_every=2000,
                memory_fraction=.5, recalibration_batches=200)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = RingTokenJEPA(RingTokenConfig(**state['config']))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False), state


def train(corpus_root, kind, out, keypoints=None, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111895+seed
    torch.manual_seed(run_seed)
    frames = Frames(corpus_root, 'train', device, keypoints=keypoints if kind == 'learned' else None, labels=False)
    slots = frames.keypoints if kind == 'learned' else 6
    model = RingTokenJEPA(RingTokenConfig(residual=True, slots=slots)).to(device)
    params = list(model.parameters())
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    windows = torch.as_tensor(frames.windows(cfg['future']), device=device)
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], cfg['steps']))
    meta = dict(recipe=f'p31_{kind}', kind=kind, keypoints=keypoints, seed=seed, train=cfg)
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
    (out/'config.json').write_text(json.dumps(dict(**meta, run_seed=run_seed, windows=len(windows),
                                                   model=asdict(model.config)), indent=1))

    def objective(index, offset=0):
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+offset+index)
        t0 = windows[torch.randint(0, len(windows), (cfg['batch'],), generator=g, device=device)]
        b = frames.batch(kind, t0, cfg['future'])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            return token_losses(model, b, rollout_steps=cfg['future'], sigreg=sigreg, lam=cfg['sigreg_weight'], generator=g)

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
    with torch.no_grad():
        rows = [objective(i, offset=6_000_000) for i in range(20)]
    log.write(json.dumps(dict(step=step, evaluation={k: round(sum(float(r[k]) for r in rows)/len(rows), 6)
                                                     for k in rows[0]}))+'\n')
    log.flush()
    save(out/'model.pt', model, meta, dict(step=step))
    print(f'done: {out}/model.pt at step {step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True)
    t.add_argument('--kind', choices=['tracker', 'learned'], required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--keypoints')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.corpus, args.kind, args.out, args.keypoints, args.seed, overrides)


if __name__ == '__main__':
    main()
