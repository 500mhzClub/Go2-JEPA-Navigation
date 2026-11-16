"""P22b item 8's noise floor (approved 15 November): how well could any value choose among a branched moment's tapes?

    python -m go2nav.eval.value_noise run --branches BRANCHES --out OUT [--samples 3] [--workers N]
    python -m go2nav.eval.value_noise summary OUT

At each validation moment of the branches (`data.value_branches`), round 0's mission is run again exactly to the
moment and the same six tapes are proposed (checked by name). Each is run again `samples` times from an exact copy of
the world with the walkers' dice drawn anew (the same new dice for every tape of a sample), for the cut and then under
the frozen rule's planner to the end of the leg, as the branches were. As a check, the first tape is also run with the
original dice and must cost what its branch did.

`summary` scores choosers as the ranking bar (a′) scores the value: the cost, in the branches' own sample, of the tape
chosen, less the best tape's (regret, seconds), averaged over moments:
- `rule`: the frozen rule's choice;
- `expected`: the tape with the least mean cost over the new samples (it knows each tape's expected cost, estimated
  apart from the sample it is scored on): what any value could at best do, short of the noise in its estimate;
- `expected_in_sample`: the same with the branches' own sample among the means (optimistic);
- `random`: a tape at random (the mean over the tapes).
If `expected` cannot halve the rule's regret, bar (a′) cannot be met by any value at this cut.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

JOB = ('index', 'split', 'maze', 'episode', 'walkers', 'kind', 'placement_seed')


def _moment(args):
    branch_json, out, samples = args
    target = Path(out)/(Path(branch_json).stem+'.npz')
    if target.exists():
        return str(target)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from go2nav.data import value_branches as vb
    from go2nav.data.value_missions import mazes_root
    meta = json.loads(Path(branch_json).read_text())
    original = dict(np.load(Path(branch_json).with_suffix('.npz')))
    fail_s = float(os.environ['GO2NAV_WALKER_PRICE'])
    episode = vb.Episode({k: meta[k] for k in JOB}, mazes_root())
    while episode.begin():
        if episode.planning() and episode.frame == meta['frame']:
            tapes, names, rule, *_ = vb.candidates(episode)
            if names != meta['names'] or not np.allclose(rule, original['rule']):
                raise RuntimeError(f'{branch_json}: the tapes differ from the branches\'')
            again = vb.branch(episode, tapes[0], fail_s)[2]
            if not np.isclose(again, original['cost_moment'][0]):
                raise RuntimeError(f'{branch_json}: the original dice cost {again}, the branch {original["cost_moment"][0]}')
            costs = np.full((samples, len(tapes)), np.nan)
            for s in range(samples):
                for k, tape in enumerate(tapes):
                    seeded = episode.copy()
                    for i, walker in enumerate(seeded.world.walkers):
                        walker.rng = np.random.default_rng([meta['index'], meta['frame'], s, i, 13])
                    costs[s, k] = vb.branch(seeded, tape, fail_s)[2]
            np.savez(target, rule=rule, cost=original['cost_moment'], samples=costs, hold=np.array(
                [n == 'hold' for n in names]))
            return str(target)
        if not episode.finish():
            break
    raise RuntimeError(f'{branch_json}: the moment was not reached')


def run(branches, out, samples=3, workers=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((Path(branches)/'val').glob('[0-9]*_[0-9].json'))
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        list(pool.map(_moment, [(str(f), str(out), samples) for f in files], chunksize=1))
    return summary(out)


def summary(out):
    rows = [dict(np.load(f)) for f in sorted(Path(out).glob('[0-9]*_[0-9].npz'))]
    regrets = {k: [] for k in ('rule', 'expected', 'expected_in_sample', 'random')}
    persistent, disasters = 0, 0
    for r in rows:
        cost, best = r['cost'], r['cost'].min()
        rule = int(np.argmin(r['rule']))
        expected = int(np.argmin(r['samples'].mean(0)))
        in_sample = int(np.argmin(np.vstack((cost[None], r['samples'])).mean(0)))
        regrets['rule'].append(cost[rule]-best)
        regrets['expected'].append(cost[expected]-best)
        regrets['expected_in_sample'].append(cost[in_sample]-best)
        regrets['random'].append(float(np.mean(cost-best)))
        if cost[rule]-best > 100:
            disasters += 1
            means = r['samples'].mean(0)
            persistent += int(means[rule]-means.min() > 100)
    report = dict(moments=len(rows), **{f'{k}_regret_s': round(float(np.mean(v)), 2) for k, v in regrets.items()})
    report['expected_over_rule'] = round(report['expected_regret_s']/max(report['rule_regret_s'], 1e-9), 3)
    report['rule_disasters'] = disasters
    report['disasters_that_recur'] = round(persistent/max(disasters, 1), 3)
    report['sample_spread_s'] = round(float(np.median([np.median(r['samples'].std(0)) for r in rows])), 2)
    (Path(out)/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--branches', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--samples', type=int, default=3)
    r.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('out', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.branches, a.out, a.samples, a.workers) if a.command == 'run' else summary(a.out)))


if __name__ == '__main__':
    main()
