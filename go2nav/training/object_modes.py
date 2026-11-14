"""Training the several-futures predictor (`models.object_modes`; docs/dynamic/PLAN.md P22 item 8) on P20's corpus and
the ring tracker's objects, on the GPU.

    python -m go2nav.training.object_modes train --corpus CORPUS --out DIR [--config FILE] [--set key=value ...]

Windows, splits and the walls' rasters are `training.object_jepa`'s: `history` steps ending at the start step, then
the model's `steps` after it; the targets are the tracker's later positions of the same track, where it is seen.
With `data.branches` set (PLAN P22 item 11), that share of each batch is branched windows (the module
`data.branches`: one moment of an episode under several random tapes), validated apart as `branch_...`;
`data.branch_episodes`: only train episodes numbered below it. With `data.planner_branches` set (item 13), that share
is windows branched from the states a planner reached under its own tapes (`data.planner_branches`), validated apart
as `planner_...`. With `init`, training starts from that model's weights (a fine-tune; the model's settings must
match).
AdamW with warm-up and cosine decay. Resumable from DIR/checkpoint.pt; the weights at the end: DIR/model.pt.

**Validation** (`DIR/validation.jsonl`, fixed val windows, the recorded commands as the tape): at 1.6, 3.2 and 6.4 s
the median error of the likeliest future (`top`), of the nearest of the futures (`nearest`), and of the tracker's
constant velocity; and `covered`: the share of walkers for which some future of probability `P_MIN` or more ends
within `COVER_M` of where the walker is. Labels are not read; PLAN P22's measures are `eval.object_measures`.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from go2nav.models.object_jepa import STEP_FRAMES, STEP_S, object_features
from go2nav.models.object_modes import WIDE_GRID, WIDE_SPACING, ModesConfig, ObjectModes, carried, losses
from go2nav.training.object_jepa import Branches, Rasters, Split, joined, merged, schedule

DEFAULTS = dict(seed=2026111201, init=None, model={}, data=dict(batch=512, branches=0., branch_episodes=None,
                                                                planner_branches=0.),
                optimiser=dict(lr=3e-4, weight_decay=1e-3, warmup=500, steps=30000, grad_clip=1.),
                runtime=dict(log_every=100, validate_every=2500, checkpoint_every=1000, validation_windows=8192))
REPORT, P_MIN, COVER_M = (4, 8, 16), .1, .15


def load_config(path, overrides=()):
    import yaml
    config = merged(DEFAULTS, yaml.safe_load(Path(path).read_text()) if path else {})
    for item in overrides:
        key, _, value = item.partition('=')
        node = config
        *parents, last = key.split('.')
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return config


def inputs(w, history, memory=False):
    """A window's model inputs and targets: objects, robot, future_robot, target, counted."""
    h = history
    objects = object_features(w['position'][:, :h], w['seen'][:, :h], w['present'][:, :h], w['walls'][:, :h],
                              w['velocity'][:, :h], w['memory'][:, :h] if memory else None)
    actions = torch.cat((w['actions'], torch.zeros_like(w['actions'][:, :1])), 1)          # none after the last step
    robot = torch.cat((w['robot'], actions), -1)
    alive = w['present'][:, h-1]
    return objects, robot[:, :h], robot[:, h:], torch.nan_to_num(w['position'][:, h:]), w['seen'][:, h:] & alive[:, None]


@torch.no_grad()
def validate(model, split, starts, batch=2048):
    model.eval()
    c = model.config
    rows = {k: dict(top=[], nearest=[], velocity=[], covered=[]) for k in REPORT if k <= c.steps}
    for i in range(0, len(starts), batch):
        w = split.window(starts[i:i+batch], c.history, c.steps, object_grid=(WIDE_GRID, WIDE_SPACING))
        objects, robot, future, target, counted = inputs(w, c.history, c.memory)
        positions, p = model.futures(objects, robot, future)                               # (B, S, K, T, 2), (B, S, K)
        top = torch.gather(positions, 2, p.argmax(-1)[:, :, None, None, None].expand(-1, -1, 1, c.steps, 2))[:, :, 0]
        base = carried(objects, c.steps)
        for k, row in rows.items():
            keep = counted[:, k-1]
            truth = target[:, k-1]
            off = (positions[:, :, :, k-1]-truth[:, :, None]).norm(dim=-1)                 # (B, S, K)
            row['top'].append((top[:, :, k-1]-truth).norm(dim=-1)[keep])
            row['nearest'].append(off.min(-1).values[keep])
            row['velocity'].append((base[:, :, k-1]-truth).norm(dim=-1)[keep])
            row['covered'].append((((off <= COVER_M) & (p >= P_MIN)).any(-1))[keep].float())
    model.train()
    out = {}
    for k, row in rows.items():
        t = f'{k*STEP_S:.1f}s'
        out.update({f'top@{t}': round(float(torch.cat(row['top']).median()), 4),
                    f'nearest@{t}': round(float(torch.cat(row['nearest']).median()), 4),
                    f'velocity@{t}': round(float(torch.cat(row['velocity']).median()), 4),
                    f'covered@{t}': round(float(torch.cat(row['covered']).mean()), 4)})
    return out


def train(corpus, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    device = torch.device('cuda:0')
    torch.manual_seed(config['seed'])
    model_config = ModesConfig(**config['model'])
    model = ObjectModes(model_config).to(device)
    if config.get('init'):
        start = torch.load(config['init'], map_location=device, weights_only=False)
        if start['config'] != model_config.to_dict():
            raise ValueError(f"{config['init']} has settings {start['config']}, this run {model_config.to_dict()}")
        model.load_state_dict(start['model'])
    h, steps = model_config.history, model_config.steps
    run, optim, data = config['runtime'], config['optimiser'], config['data']
    mazes = sorted({json.loads(p.read_text())['maze'] for split in ('train', 'val')
                    for p in (Path(corpus)/split).glob('*.json')})
    rasters = Rasters(Path(corpus)/'objects'/'rasters', mazes, device)
    frames = STEP_FRAMES*(h-1+steps)+1
    train_split, val_split = Split(corpus, 'train', rasters, device, frames), Split(corpus, 'val', rasters, device, frames)
    pick = torch.as_tensor(np.random.default_rng(config['seed']).choice(len(val_split.windows),
                                                                        run['validation_windows'], replace=False))
    val_starts = val_split.windows[pick.to(device)]
    share = data['branches']
    if share:
        branch_train = Branches(corpus, 'train', rasters, device, frames, data['branch_episodes'])
        branch_val = Branches(corpus, 'val', rasters, device, frames)
        pick = torch.as_tensor(np.random.default_rng(config['seed']).choice(
            len(branch_val.windows), min(run['validation_windows'], len(branch_val.windows)), replace=False))
        branch_starts = branch_val.windows[pick.to(device)]
        (out/'branches.json').write_text(json.dumps(dict(train=len(branch_train.windows), val=len(branch_val.windows)))+'\n')
    planner_share = data.get('planner_branches', 0.)
    if planner_share:
        planner_train = Branches(corpus, 'train', rasters, device, frames, folder='planner_branches')
        planner_val = Branches(corpus, 'val', rasters, device, frames, folder='planner_branches')
        pick = torch.as_tensor(np.random.default_rng(config['seed']).choice(
            len(planner_val.windows), min(run['validation_windows'], len(planner_val.windows)), replace=False))
        planner_starts = planner_val.windows[pick.to(device)]
        (out/'planner_branches.json').write_text(json.dumps(dict(train=len(planner_train.windows),
                                                                 val=len(planner_val.windows)))+'\n')
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
        branched, planned = int(round(data['batch']*share)), int(round(data['batch']*planner_share))
        starts = train_split.windows[torch.randint(len(train_split.windows), (data['batch']-branched-planned,),
                                                   device=device, generator=generator)]
        w = train_split.window(starts, h, steps, generator, object_grid=(WIDE_GRID, WIDE_SPACING))
        if branched:
            starts = branch_train.windows[torch.randint(len(branch_train.windows), (branched,), device=device,
                                                        generator=generator)]
            w = joined(w, branch_train.window(starts, h, steps, generator, object_grid=(WIDE_GRID, WIDE_SPACING)))
        if planned:
            starts = planner_train.windows[torch.randint(len(planner_train.windows), (planned,), device=device,
                                                         generator=generator)]
            w = joined(w, planner_train.window(starts, h, steps, generator, object_grid=(WIDE_GRID, WIDE_SPACING)))
        total, terms = losses(model, *inputs(w, h, model_config.memory))
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
                                      **{k: round(v/n, 5) for k, v in sums.items()}))+'\n')
            log.flush()
            sums, started, n = {}, time.time(), 0
        if step % run['validate_every'] == 0 or step == optim['steps']:
            report = validate(model, val_split, val_starts)
            if share:
                report.update({f'branch_{k}': v for k, v in validate(model, branch_val, branch_starts).items()})
            if planner_share:
                report.update({f'planner_{k}': v for k, v in validate(model, planner_val, planner_starts).items()})
            vlog.write(json.dumps(dict(step=step, **report))+'\n')
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
    model = ObjectModes(ModesConfig(**state['config'])).to(device)
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
