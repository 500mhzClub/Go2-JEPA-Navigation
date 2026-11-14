"""P22b's value trainer (docs/dynamic/PLAN.md): the cost to go of `data.value_missions`'s plans, from the window round
the robot (`models.value`), as a correction to the frozen rule's distance term.

    python -m go2nav.training.value --data DATA/packed [MORE/packed ...] --out RUN [--steps 20000] [--batch 1024]

On the GPU only (the dedicated R9700; it refuses the CPU). Each moment's window is drawn on the GPU from its stored
state and its layout's grids. A squared-error loss (in units of `models.value.BASE_S`), so that the value is the
expected cost to go, as the planner's other terms are expectations (approved 14 November, in place of a Huber loss,
which learns the typical outcome and would pass over a rare failure); Adam with a warm-up and a cosine decay. Every
`EVERY` steps (`evaluate`): on the validation split, the mean squared error of the value and of the distance term
alone, over all moments and over those with a walker within `NEAR_M`, with the medians beside them; a checkpoint
(RUN/last.pt, which a rerun resumes from); and the best by the value's mean squared error over all moments
(RUN/model.pt). RUN/train.log has a line per evaluation.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from go2nav.models import value as vm

EVERY, NEAR_M, VAL_LIMIT, CHUNK = 1000, 3., 200_000, 4096
FIELDS = ('xy', 'yaw', 'slots_xy', 'slots_v', 'memory', 'present', 'base', 'own', 'cost', 'leg', 'grid')


def load(folders, split, device):
    """Moments of `split` from packed folders (`data.value_missions.pack`), and their grids, on `device`."""
    parts, walls, distance, low, offset = [], [], [], [], 0
    for folder in map(Path, folders):
        grids = np.load(folder/'grids.npz')
        data = np.load(folder/f'{split}.npz')
        parts.append({k: data[k] if k != 'grid' else data[k]+offset for k in FIELDS})
        walls.append(grids['walls'])
        distance.append(grids['distance'])
        low.append(grids['low'])
        offset += len(grids['walls'])
    moments = {k: torch.as_tensor(np.concatenate([p[k] for p in parts]), device=device) for k in FIELDS}
    moments['leg'], moments['grid'] = moments['leg'].long(), moments['grid'].long()
    return moments, (torch.as_tensor(np.concatenate(walls), device=device),
                     torch.as_tensor(np.concatenate(distance), device=device),
                     torch.as_tensor(np.concatenate(low), device=device))


def windows(moments, grids, idx):
    walls, distance, low = grids
    g, m = moments['grid'][idx], moments
    return vm.draw(walls[g], distance[g, m['leg'][idx]], low[g], m['xy'][idx], m['yaw'][idx], m['own'][idx],
                   m['slots_xy'][idx], m['slots_v'][idx], m['memory'][idx], m['present'][idx])


def near(moments, idx):
    """Moments with a walker in a slot within `NEAR_M` of the robot."""
    gap = torch.linalg.norm(moments['slots_xy'][idx]-moments['xy'][idx, None], dim=-1)
    return ((gap < NEAR_M) & moments['present'][idx]).any(1)


@torch.no_grad()
def evaluate(net, moments, grids, limit=VAL_LIMIT):
    """Mean squared errors (s^2), with the medians of the absolute errors (s), of the value and of the distance term
    alone, over all moments and those near walkers; `ratio`: the value's mean squared error over the distance term's."""
    count = len(moments['cost'])
    idx_all = torch.arange(count, device=moments['cost'].device)
    if count > limit:
        idx_all = idx_all[torch.randperm(count, generator=torch.Generator().manual_seed(0))[:limit].to(idx_all.device)]
    value, close = [], []
    for chunk in idx_all.split(CHUNK):
        value.append(moments['base'][chunk]+net(windows(moments, grids, chunk), moments['base'][chunk],
                                               moments['own'][chunk]))
        close.append(near(moments, chunk))
    value, close = torch.cat(value), torch.cat(close)
    cost, base = moments['cost'][idx_all], moments['base'][idx_all]
    out = {}
    for name, keep in (('all', torch.ones_like(close)), ('near', close)):
        a, b = (value-cost)[keep], (base-cost)[keep]
        out[name] = dict(moments=int(keep.sum()), value_mse=round(float((a**2).mean()), 1),
                         distance_term_mse=round(float((b**2).mean()), 1),
                         value_median_s=round(float(a.abs().median()), 2),
                         distance_term_median_s=round(float(b.abs().median()), 2))
        out[name]['ratio'] = round(out[name]['value_mse']/max(out[name]['distance_term_mse'], 1e-9), 3)
    return out


def train(folders, out, steps=20000, batch=1024, lr=3e-4, warmup=500, seed=0):
    if not torch.cuda.is_available():
        raise RuntimeError('the value trains on the dedicated GPU only')
    device = torch.device('cuda:0')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    train_set, grids = load(folders, 'train', device)
    val_set, val_grids = load(folders, 'val', device)
    net = vm.ValueNet().to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1., (s+1)/warmup)*.5*(1+math.cos(math.pi*min(
        s, steps)/steps)))
    start, best = 0, math.inf
    if (out/'last.pt').exists():
        state = torch.load(out/'last.pt', map_location=device, weights_only=False)
        net.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        schedule.load_state_dict(state['schedule'])
        start, best = state['step'], state['best']
    log = (out/'train.log').open('a')
    log.write(json.dumps(dict(start=start, folders=[str(f) for f in folders], train=len(train_set['cost']),
                              val=len(val_set['cost']), steps=steps, batch=batch, lr=lr))+'\n')
    log.flush()
    generator = torch.Generator(device=device).manual_seed(seed+start)
    began, total = time.time(), 0.
    for step in range(start, steps):
        idx = torch.randint(0, len(train_set['cost']), (batch,), device=device, generator=generator)
        correction = net(windows(train_set, grids, idx), train_set['base'][idx], train_set['own'][idx])
        loss = F.mse_loss((train_set['base'][idx]+correction)/vm.BASE_S, train_set['cost'][idx]/vm.BASE_S)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimiser.step()
        schedule.step()
        total += float(loss)
        if (step+1) % EVERY == 0 or step+1 == steps:
            net.eval()
            report = evaluate(net, val_set, val_grids)
            net.train()
            line = dict(step=step+1, loss=round(total/EVERY, 4), seconds=round(time.time()-began, 1), **report)
            total = 0.
            if report['all']['value_mse'] < best:
                best = report['all']['value_mse']
                vm.save(out/'model.pt', net, dict(step=step+1, val=report))
                line['best'] = True
            torch.save(dict(model=net.state_dict(), optimiser=optimiser.state_dict(), schedule=schedule.state_dict(),
                            step=step+1, best=best), out/'last.pt.tmp')
            (out/'last.pt.tmp').rename(out/'last.pt')
            log.write(json.dumps(line)+'\n')
            log.flush()
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, type=Path, nargs='+')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    a = p.parse_args()
    print(train(a.data, a.out, a.steps, a.batch, a.lr))


if __name__ == '__main__':
    main()
