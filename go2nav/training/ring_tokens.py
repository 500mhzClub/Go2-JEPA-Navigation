"""P28's runs (docs/dynamic/PLAN.md P28): the object-token JEPA (`models.ring_tokens`) on the F004 scenes
(`data.ring_scenes`).

    python -m go2nav.training.ring_tokens train --data ROOT --corpus CORPUS --out DIR [--seed S] [--set k=v ...]

P27's R4 settings: AdamW 3e-4 with warm-up then cosine, clip 1, bf16 autocast (SIGReg in float32, 128 projections),
batch 64, an 8-step rollout loss, 0.09 x SIGReg; 32,000 steps (15 a second, inside 45 minutes with the measures). A sample is a training scene under one of its 7
futures, drawn from (seed, step). DIR/checkpoint.pt every `checkpoint_every` steps (rerunning resumes); at the end
the BatchNorm statistics are recomputed over training batches (`training.recipe_bench.recalibrate`) and DIR/model.pt
is written, after the objective is logged once more in evaluation mode (`evaluation` in metrics.jsonl).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from go2nav.data import ring_scenes
from go2nav.models.lewm import SIGReg
from go2nav.models.ring_tokens import RingTokenConfig, RingTokenJEPA, token_losses
from go2nav.training.balls import schedule
from go2nav.training.recipe_bench import recalibrate, save

DEFAULTS = dict(steps=32000, batch=64, rollout=8, lr=3e-4, weight_decay=1e-3, warmup=500, grad_clip=1.,
                sigreg_weight=.09, sigreg_projections=128, bf16=True, log_every=250, checkpoint_every=2000,
                memory_fraction=.4, recalibration_batches=200)
RECIPES = dict(p28=dict(), p28q3=dict(model=dict(residual=True)))


def load(path, device='cpu'):
    """A trained run's (model in eval mode, checkpoint dict)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = RingTokenJEPA(RingTokenConfig(**state['config']))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device).eval().requires_grad_(False), state


def train(root, corpus, out, recipe='p28', seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    options = RECIPES[recipe]
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111680+seed
    torch.manual_seed(run_seed)
    model = RingTokenJEPA(RingTokenConfig(**options.get('model', {}))).to(device)
    params = list(model.parameters())
    sigreg = SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    scenes = ring_scenes.Scenes(root, 'train', ring_scenes.rasters_for(root, corpus, device), device)
    total = cfg['steps']
    optimiser = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], total))
    meta = dict(recipe=recipe, seed=seed, train=cfg)
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
        draw = lambda high: torch.randint(0, high, (cfg['batch'],), generator=g, device=device)
        b = scenes.batch(draw(scenes.count), draw(ring_scenes.FUTURES))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            return token_losses(model, b, rollout_steps=cfg['rollout'], sigreg=sigreg, lam=cfg['sigreg_weight'],
                                generator=g, **options.get('objective', {}))

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
    with torch.no_grad():       # the objective again with the running statistics: far from training's means a mismatch
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
    t.add_argument('--data', required=True)
    t.add_argument('--corpus', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--recipe', choices=sorted(RECIPES), default='p28')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.data, args.corpus, args.out, args.recipe, args.seed, overrides)


if __name__ == '__main__':
    main()
