"""P22b's value trainer (docs/dynamic/PLAN.md): the cost to go, from the window round the robot (`models.value`), as a
correction to the frozen rule's distance term.

    python -m go2nav.training.value --data DATA0/packed --branches BRANCHES/packed --out RUN [--steps 8000]

On the GPU only (the dedicated R9700; it refuses the CPU). Each state's window is drawn on the GPU from its stored state
and its layout's grids. A squared-error loss in units of `models.value.BASE_S`, so that the value is the expected cost
to go, as the planner's other terms are expectations (approved 14 November in place of a Huber loss). Item 8 (approved
14 November, after V0): three terms, weighted equally:
- the branches' cuts (`data.value_branches`) and the costs realised from them;
- round 0's moments (`data.value_missions`) and the costs they realised;
- consistency along round 0's missions: a moment's value against the seconds to its leg's next plan plus that plan's
  value (held fixed), the last plan of a leg against its realised cost.
Adam with a warm-up and a cosine decay. Every `EVERY` steps (`evaluate`, `ranking`): on the validation splits, the mean
squared errors of the value and of the distance term alone (round 0's moments, over all and near walkers), the branches'
squared error, and the ranking (`ranking`: the planner's choice among each moment's tapes by the value against the
frozen rule's, scored by the realised cost of the chosen tape less the best's); a checkpoint (RUN/last.pt, which a
rerun resumes from); the best by the validation loss, the sum of the moments' and branches' squared errors
(RUN/model.pt), so that the ranking bar is not the statistic chosen on. RUN/train.log has a line per evaluation.
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

EVERY, NEAR_M, VAL_LIMIT, CHUNK, FRAME_S = 500, 3., 200_000, 4096, .1
MOMENT_FIELDS = ('xy', 'yaw', 'slots_xy', 'slots_v', 'memory', 'present', 'base', 'own', 'cost', 'leg', 'grid', 'frame',
                 'mission')
BRANCH_FIELDS = ('rule', 'clean', 'terminal', 'pred_xy', 'pred_yaw', 'pred_own', 'pred_slots_xy', 'pred_slots_v',
                 'pred_memory', 'pred_present', 'cut_reached', 'cut_xy', 'cut_yaw', 'cut_slots_xy', 'cut_slots_v',
                 'cut_memory', 'cut_present', 'cut_base', 'cut_own', 'cost_cut', 'cost_moment', 'moment', 'grid', 'leg',
                 'hold')


def _grids(folder, device):
    grids = np.load(Path(folder)/'grids.npz')
    return tuple(torch.as_tensor(grids[k], device=device) for k in ('walls', 'distance', 'low'))


def load(folders, split, device):
    """Round 0's moments of `split` from packed folders (`data.value_missions.pack`) with their grids, on `device`;
    `next`: the index of the leg's next plan (-1 at a leg's last), `step_s`: the seconds to it."""
    parts, grids, offset = [], [], 0
    for folder in map(Path, folders):
        data = np.load(folder/f'{split}.npz')
        g = _grids(folder, device)
        parts.append({k: data[k] if k != 'grid' else data[k]+offset for k in MOMENT_FIELDS})
        grids.append(g)
        offset += len(g[0])
    m = {k: np.concatenate([p[k] for p in parts]) for k in MOMENT_FIELDS}
    same = (m['mission'][1:] == m['mission'][:-1]) & (m['leg'][1:] == m['leg'][:-1])
    m['next'] = np.append(np.where(same, np.arange(1, len(same)+1), -1), -1)
    m['step_s'] = np.append(np.where(same, (m['frame'][1:]-m['frame'][:-1])*FRAME_S, 0.), 0.).astype(np.float32)
    moments = {k: torch.as_tensor(v, device=device) for k, v in m.items()}
    for k in ('leg', 'grid', 'next'):
        moments[k] = moments[k].long()
    return moments, tuple(torch.cat([g[i] for g in grids]) for i in range(3))


def load_branches(folder, split, device):
    """The branches of `split` (`data.value_branches.pack`) with their grids, on `device`."""
    data = np.load(Path(folder)/f'{split}.npz')
    b = {k: torch.as_tensor(data[k], device=device) for k in BRANCH_FIELDS}
    for k in ('moment', 'grid', 'leg'):
        b[k] = b[k].long()
    return b, _grids(folder, device)


def windows(data, grids, idx, prefix=''):
    """Windows of rows `idx` of moments (no prefix) or of branches' cuts (`cut_`) or predicted cuts (`pred_`)."""
    walls, distance, low = grids
    g, p = data['grid'][idx], prefix
    return vm.draw(walls[g], distance[g, data['leg'][idx]], low[g], data[p+'xy'][idx], data[p+'yaw'][idx],
                   data[p+'own'][idx], data[p+'slots_xy'][idx], data[p+'slots_v'][idx], data[p+'memory'][idx],
                   data[p+'present'][idx])


def value_of(net, data, grids, idx, prefix='', base=None):
    """The value (seconds) of rows `idx`: their distance term plus the network's correction."""
    base = data['base' if not prefix else prefix+'base'][idx] if base is None else base
    return base+net(windows(data, grids, idx, prefix), base, data[prefix+'own' if prefix else 'own'][idx])


def near(moments, idx):
    """Moments with a walker in a slot within `NEAR_M` of the robot."""
    gap = torch.linalg.norm(moments['slots_xy'][idx]-moments['xy'][idx, None], dim=-1)
    return ((gap < NEAR_M) & moments['present'][idx]).any(1)


@torch.no_grad()
def evaluate(net, moments, grids, limit=VAL_LIMIT):
    """Mean squared errors (s^2), with the medians of the absolute errors (s), of the value and of the distance term
    alone over round 0's moments, all and near walkers; `ratio`: the value's mean squared error over the term's."""
    count = len(moments['cost'])
    idx_all = torch.arange(count, device=moments['cost'].device)
    if count > limit:
        idx_all = idx_all[torch.randperm(count, generator=torch.Generator().manual_seed(0))[:limit].to(idx_all.device)]
    value = torch.cat([value_of(net, moments, grids, c) for c in idx_all.split(CHUNK)])
    close = torch.cat([near(moments, c) for c in idx_all.split(CHUNK)])
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


@torch.no_grad()
def branch_mse(net, branches, grids):
    """The value's and the distance term's mean squared errors (s^2) at the branches' true cuts."""
    idx = torch.nonzero(branches['cut_reached'])[:, 0]
    value = torch.cat([value_of(net, branches, grids, c, 'cut_') for c in idx.split(CHUNK)])
    cost, base = branches['cost_cut'][idx], branches['cut_base'][idx]
    return dict(tapes=len(idx), value_mse=round(float(((value-cost)**2).mean()), 1),
                distance_term_mse=round(float(((base-cost)**2).mean()), 1))


@torch.no_grad()
def ranking(net, branches, grids):
    """At each branched moment, the planner's choice among its tapes by the frozen rule's cost at the cut's horizon
    (`rule`) and by that cost with the value's correction at each tape's predicted cut (`rule` + `clean` x correction),
    each scored by the chosen tape's realised cost from the moment less the best tape's (regret, seconds)."""
    idx = torch.arange(len(branches['rule']), device=branches['rule'].device)
    correction = torch.cat([value_of(net, branches, grids, c, 'pred_', branches['terminal'][c])-branches['terminal'][c]
                            for c in idx.split(CHUNK)])
    scored = branches['rule']+branches['clean']*correction
    moments = branches['moment']
    count = int(moments.max())+1
    regrets, rho, holds = {'rule': [], 'value': []}, [], [0, 0]
    cost = branches['cost_moment']
    for m in range(count):
        rows = torch.nonzero(moments == m)[:, 0]
        if len(rows) < 2:
            continue
        c, best = cost[rows], cost[rows].min()
        for name, s in (('rule', branches['rule'][rows]), ('value', scored[rows])):
            regrets[name].append(float(c[int(torch.argmin(s))]-best))
        rank = lambda a: torch.argsort(torch.argsort(a)).float()
        a, b = rank(scored[rows]), rank(c)
        if a.std() > 0 and b.std() > 0:
            rho.append(float(((a-a.mean())*(b-b.mean())).mean()/(a.std(unbiased=False)*b.std(unbiased=False))))
        hold = branches['hold'][rows]
        holds[0] += int(hold[int(torch.argmin(scored[rows]))] and not hold[int(torch.argmin(branches['rule'][rows]))])
        holds[1] += 1
    rule_regret, value_regret = float(np.mean(regrets['rule'])), float(np.mean(regrets['value']))
    return dict(moments=len(regrets['rule']), rule_regret_s=round(rule_regret, 2), value_regret_s=round(value_regret, 2),
                ratio=round(value_regret/max(rule_regret, 1e-9), 3), rank_correlation=round(float(np.mean(rho)), 3),
                holds_where_the_rule_does_not=round(holds[0]/max(holds[1], 1), 3))


def train(folders, branches_folder, out, steps=8000, batch=1024, lr=3e-4, warmup=500, seed=0):
    if not torch.cuda.is_available():
        raise RuntimeError('the value trains on the dedicated GPU only')
    device = torch.device('cuda:0')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    moments, grids = load(folders, 'train', device)
    val_moments, val_grids = load(folders, 'val', device)
    branches, branch_grids = load_branches(branches_folder, 'train', device)
    val_branches, val_branch_grids = load_branches(branches_folder, 'val', device)
    reached = torch.nonzero(branches['cut_reached'])[:, 0]
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
    log.write(json.dumps(dict(start=start, folders=[str(f) for f in folders], branches=str(branches_folder),
                              moments=len(moments['cost']), branch_cuts=len(reached), steps=steps, batch=batch, lr=lr))
              + '\n')
    log.flush()
    generator = torch.Generator(device=device).manual_seed(seed+start)
    began, totals, scale = time.time(), np.zeros(3), vm.BASE_S
    for step in range(start, steps):
        ib = reached[torch.randint(0, len(reached), (batch,), device=device, generator=generator)]
        loss_branch = F.mse_loss(value_of(net, branches, branch_grids, ib, 'cut_')/scale, branches['cost_cut'][ib]/scale)
        im = torch.randint(0, len(moments['cost']), (batch,), device=device, generator=generator)
        value = value_of(net, moments, grids, im)
        loss_mc = F.mse_loss(value/scale, moments['cost'][im]/scale)
        following, has = moments['next'][im], moments['next'][im] >= 0
        target = moments['cost'][im].clone()
        if has.any():
            with torch.no_grad():
                target[has] = moments['step_s'][im][has]+value_of(net, moments, grids, following[has])
        loss_td = F.mse_loss(value/scale, target/scale)
        loss = loss_branch+loss_mc+loss_td
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimiser.step()
        schedule.step()
        totals += (float(loss_branch), float(loss_mc), float(loss_td))
        if (step+1) % EVERY == 0 or step+1 == steps:
            net.eval()
            report = dict(moments=evaluate(net, val_moments, val_grids),
                          branches=branch_mse(net, val_branches, val_branch_grids),
                          ranking=ranking(net, val_branches, val_branch_grids))
            net.train()
            selection = report['moments']['all']['value_mse']+report['branches']['value_mse']
            line = dict(step=step+1, losses=dict(zip(('branch', 'moments', 'consistency'), np.round(totals/EVERY, 4))),
                        seconds=round(time.time()-began, 1), selection=round(selection, 1), **report)
            totals[:] = 0
            if selection < best:
                best = selection
                vm.save(out/'model.pt', net, dict(step=step+1, val=report))
                line['best'] = True
            torch.save(dict(model=net.state_dict(), optimiser=optimiser.state_dict(), schedule=schedule.state_dict(),
                            step=step+1, best=best), out/'last.pt.tmp')
            (out/'last.pt.tmp').rename(out/'last.pt')
            log.write(json.dumps(line, default=float)+'\n')
            log.flush()
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, type=Path, nargs='+')
    p.add_argument('--branches', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    a = p.parse_args()
    print(train(a.data, a.branches, a.out, a.steps, a.batch, a.lr))


if __name__ == '__main__':
    main()
