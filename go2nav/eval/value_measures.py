"""P22b's measures (a), (a′) and (b) (docs/dynamic/PLAN.md) of a trained value (`models.value`, `training.value`).

    python -m go2nav.eval.value_measures offline --data DATA/packed [...] --model RUN/model.pt --out FILE
    python -m go2nav.eval.value_measures ranking --branches BRANCHES/packed --model RUN/model.pt --out FILE
    python -m go2nav.eval.value_measures traps --bank BANK --model RUN/model.pt --out FILE [--source true_rude4]

- `ranking` (a′, item 8): at the validation branches' moments (`data.value_branches`), the planner's choice among the
  tapes by the value against the frozen rule's at the same cut, each scored by the realised cost of the chosen tape less
  the best's (`training.value.ranking`). Bar: the value's mean regret at most half the frozen rule's. On the GPU.

- `offline` (a): on the validation layouts' moments, the mean squared error of the cost to go of the value and of the
  frozen rule's distance term alone, over all moments and over those with a walker within 3 m, with the medians beside
  them (`training.value.evaluate`, every moment). Bar (approved 14 November in place of the medians): the value's mean
  squared error at most half the distance term's in both. On the GPU.
- `traps` (b): the timed-out missions of a decision bank source (`eval.decision_bank`), replayed from their recorded
  requests; every second, whether the robot is trapped (`eval.deadlocks.trapped`) and the value's cost to go at its
  state (its walkers from the value's own tracker on the scan). Bar: while trapped, a median above the failure price
  (the frozen rule's walker price) in at least 9 of 11. Read beside it: the value `BEFORE_S` before the robot was
  trapped for good (the last stretch trapped to the end). On the CPU.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BEFORE_S = 5.


def offline(folders, model, out):
    import torch
    from go2nav.models import value as vm
    from go2nav.training.value import evaluate, load
    if not torch.cuda.is_available():
        raise RuntimeError('the offline measure runs on the GPU')
    device = torch.device('cuda:0')
    moments, grids = load(folders, 'val', device)
    report = evaluate(vm.load(model, device), moments, grids, limit=len(moments['cost']))
    report['passes'] = all(report[k]['ratio'] <= .5 for k in ('all', 'near'))
    Path(out).write_text(json.dumps(dict(model=str(model), data=[str(f) for f in folders], **report), indent=1)+'\n')
    return report


def ranking(branches, model, out):
    import torch
    from go2nav.models import value as vm
    from go2nav.training.value import load_branches, ranking as rank
    if not torch.cuda.is_available():
        raise RuntimeError('the ranking measure runs on the GPU')
    device = torch.device('cuda:0')
    data, grids = load_branches(branches, 'val', device)
    report = rank(vm.load(model, device), data, grids)
    report['passes'] = report['ratio'] <= .5
    Path(out).write_text(json.dumps(dict(model=str(model), branches=str(branches), **report), indent=1)+'\n')
    return report


def _trap(args):
    json_file, model = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from go2nav.eval.decision_bank import SOURCES, mission_world
    from go2nav.eval.deadlocks import trapped
    from go2nav.eval.kinematic_missions import ARRIVAL_RADIUS_M
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.fields import Fields
    from go2nav.planning.value import Value, distance_term
    from go2nav.sim.kinematic import FRAME_S
    meta = json.loads(Path(json_file).read_text())
    steps = np.load(Path(json_file).with_suffix('.npz'))['steps']
    set_name, maze, episode, walkers = meta['key']
    world = mission_world(set_name, maze, episode, walkers, SOURCES[meta['source']][1])
    _, record = episode_spec(set_name, maze, episode)
    home, beacon = np.asarray(record['home_se2_world'][:2], float), np.asarray(record['beacon_xy_world'], float)
    fields = Fields(world.geometry)
    value = Value(model, device='cpu')
    value.bind(fields)
    reached, seconds = False, []
    for frame, s in enumerate(steps):
        value.observe(frame*FRAME_S, world.scan(), world.xy, world.yaw, world.geometry)
        if frame % 10 == 0:
            target = home if reached else beacon
            now = value.walkers.now
            at = np.where(now['present'][:, None], now['xy'], np.nan)
            base = distance_term(fields, target, world.xy[None], at[None])
            v = float(base[0]+value.correction(target, world.xy[None], np.array([world.yaw]), np.zeros((1, 2, 0, 2)),
                                               base)[0])
            seconds.append((frame/10, trapped(world, target)[1], v, float(base[0])))
        world.step(s)
        reached = reached or float(np.linalg.norm(world.xy-beacon)) < ARRIVAL_RADIUS_M
    caught = [v for _, t, v, _ in seconds if t]
    free = [i for i, (_, t, _, _) in enumerate(seconds) if not t]
    began = free[-1]+1 if free else 0                          # the last stretch trapped to the end
    before = max(began-int(BEFORE_S), 0)
    return dict(mission=Path(json_file).stem, trapped_s=len(caught),
                value_trapped_median_s=round(float(np.median(caught)), 1) if caught else None,
                distance_term_trapped_median_s=round(float(np.median([b for _, t, _, b in seconds if t])), 1)
                if caught else None,
                value_before_s=round(seconds[before][2], 1), distance_term_before_s=round(seconds[before][3], 1))


def traps(bank, model, out, source='true_rude4', workers=None):
    price = float(os.environ.get('GO2NAV_WALKER_PRICE', 'nan'))
    if not np.isfinite(price):
        raise RuntimeError('the frozen rule is not set (GO2NAV_WALKER_PRICE)')
    files = sorted((Path(bank)/source).glob('[0-9]*.json'))
    timeouts = [f for f in files if json.loads(f.read_text())['exit'] == 'timeout']
    with ProcessPoolExecutor(workers or len(timeouts)) as pool:
        rows = list(pool.map(_trap, [(str(f), str(model)) for f in timeouts]))
    above = sum(r['value_trapped_median_s'] is not None and r['value_trapped_median_s'] > price for r in rows)
    report = dict(model=str(model), source=source, timeouts=len(rows), price_s=price, above_price=above,
                  passes=above >= 9 if len(rows) == 11 else None, missions=rows)
    Path(out).write_text(json.dumps(report, indent=1)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    o = sub.add_parser('offline')
    o.add_argument('--data', required=True, type=Path, nargs='+')
    o.add_argument('--model', required=True, type=Path)
    o.add_argument('--out', required=True, type=Path)
    r = sub.add_parser('ranking')
    r.add_argument('--branches', required=True, type=Path)
    r.add_argument('--model', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    t = sub.add_parser('traps')
    t.add_argument('--bank', required=True, type=Path)
    t.add_argument('--model', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--source', default='true_rude4')
    t.add_argument('--workers', type=int)
    a = p.parse_args()
    if a.command == 'offline':
        report = offline(a.data, a.model, a.out)
    elif a.command == 'ranking':
        report = ranking(a.branches, a.model, a.out)
    else:
        report = traps(a.bank, a.model, a.out, a.source, a.workers)
        report = {k: v for k, v in report.items() if k != 'missions'}
    print(json.dumps(report))


if __name__ == '__main__':
    main()
