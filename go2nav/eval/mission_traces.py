"""P22 item 12: what the robot and the walkers were doing when a mission ended in a contact or a timeout.

    python -m go2nav.eval.mission_traces run --model SPEC --from DIR --out DIR [--exits contact_walker timeout]
        [--workers N]
    python -m go2nav.eval.mission_traces summary DIR [DIR ...]

`run` reruns the missions of a finished suite (DIR/<maze>_<episode>_<walkers>.json, `eval.kinematic_missions`) whose
exit is one of `exits`, with the same settings and a probe that changes nothing (missions repeat exactly; the row is
checked against the suite's). Per plan after the first: the frame, the robot's pose, the chosen tape's name, its cost
and terms, the share of its cost not in time, clearance or progress (the contact price for a model of several
futures), its mean forward request over the frames run; per walker: position, whether it is held up by the robot
(`sim.kinematic.waiting_on`), paused, polite. DIR/<maze>_<episode>_<walkers>.json: the mission's row and `plans`.

**summary,** per exit: for contacts, over the last plan before each: whether the robot was moving (a forward request
over `MOVING_M_S`), the chosen tape's kind, the contact price it carried, whether the walker touched was held up by the
robot, and the seconds the robot had held in the last 10 s; for timeouts, over the last 60 s: the share of plans that
held, the path walked, the walkers within 1 m and those held up by the robot, and whether the robot ended in a dead end.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

MOVING_M_S = .05


class Recorder:
    def __init__(self):
        self.plans, self.attached, self.last = [], False, None

    def _attach(self, planner):
        plan = planner.plan

        def recorded(snapshot, target):
            tape, info = plan(snapshot, target)
            self.last = (tape.copy(), info)
            return tape, info
        planner.plan, self.attached = recorded, True

    def __call__(self, planner, world, snapshot, target):
        from go2nav.sim import kinematic as kin
        if not self.attached:
            self._attach(planner)
            return
        tape, info = self.last
        price = info['cost']-info['time']-info['clearance']-info['progress']
        self.plans.append(dict(
            frame=int(world.frame), xy=np.round(world.xy, 3).tolist(), yaw=round(float(world.yaw), 3),
            chosen=info['chosen'], cost=round(info['cost'], 2), time=round(info['time'], 2),
            progress=round(info['progress'], 2), sealed=info['sealed'], price=round(price, 2),
            forward=round(float(tape[:, 0].mean()), 3),
            walkers=[dict(xy=np.round(w.xy, 3).tolist(), held=bool(kin.waiting_on(w, world.xy)),
                          paused=bool(w.pause_left > 0), polite=bool(getattr(w, 'polite', True))) for w in world.walkers]))


def _job(args):
    out, source, kwargs = args
    from go2nav.eval.kinematic_missions import run_mission
    row = json.loads(Path(source).read_text())
    target = Path(out)/Path(source).name
    if target.exists():
        return json.loads(target.read_text())
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    recorder = Recorder()
    traced = run_mission(row['set'], row['maze'], row['episode'], walkers=row['walkers'], kind=row['kind'],
                         horizon_s=row['horizon_s'], terminal=row['terminal'], model=kwargs['model'],
                         seed=kwargs['seed'], max_speed=row['max_speed'], cells=row['cells'], probe=recorder)
    traced['repeats'] = traced['exit'] == row['exit'] and traced['simulated_s'] == row['simulated_s']
    traced['plans_traced'] = recorder.plans
    partial = target.with_suffix('.tmp')
    partial.write_text(json.dumps(traced)+'\n')
    partial.rename(target)
    return traced


def run(out, *, model, source, exits, seed, workers=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [f for f in sorted(Path(source).glob('[0-9]*_*.json')) if json.loads(f.read_text())['exit'] in exits]
    jobs = [(str(out), str(f), dict(model=model, seed=seed)) for f in rows]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        traced = list(pool.map(_job, jobs))
    report = summary(traced)
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def _dead_end(row, xy):
    from go2nav.eval.mission import episode_spec
    from go2nav.sim import obstacles as moving
    spec, _ = episode_spec(row['set'], row['maze'], row['episode'])
    neighbours, pitch = moving.passages(spec), moving.pitch_of(spec)
    cell = moving.cell_of(xy, pitch)
    return len(neighbours.get(cell, ())) == 1


def summary(rows):
    out = dict(missions=len(rows), repeats=sum(r.get('repeats', False) for r in rows))
    contacts = [r for r in rows if r['exit'] == 'contact_walker' and r['plans_traced']]
    if contacts:
        last = [r['plans_traced'][-1] for r in contacts]
        touched = []
        for r, p in zip(contacts, last):
            gaps = [np.linalg.norm(np.subtract(w['xy'], p['xy'])) for w in p['walkers']]
            touched.append(p['walkers'][int(np.argmin(gaps))])
        held_10 = [sum(q['chosen'] == 'hold' for q in r['plans_traced'] if q['frame'] >= r['plans_traced'][-1]['frame']-100)
                   * .4 for r in contacts]
        kinds = {}
        for p in last:
            k = p['chosen'].split('@')[0].split('_wait')[0]
            k = 'cell' if k.startswith('cell') else ('gaussian' if k.startswith('gaussian') else k)
            kinds[k] = kinds.get(k, 0)+1
        out['contacts'] = dict(
            count=len(contacts), robot_moving=sum(p['forward'] > MOVING_M_S for p in last),
            robot_backing=sum(p['forward'] < -MOVING_M_S for p in last), chosen=kinds,
            price_median=float(np.median([p['price'] for p in last])),
            priced_over_20_s=sum(p['price'] > 20 for p in last),
            touched_held_by_robot=sum(w['held'] for w in touched), touched_paused=sum(w['paused'] for w in touched),
            held_s_in_last_10_s_median=float(np.median(held_10)))
    timeouts = [r for r in rows if r['exit'] == 'timeout' and r['plans_traced']]
    if timeouts:
        stats = []
        for r in timeouts:
            end = r['plans_traced'][-1]['frame']
            tail = [q for q in r['plans_traced'] if q['frame'] >= end-600]
            path = sum(np.linalg.norm(np.subtract(b['xy'], a['xy'])) for a, b in zip(tail, tail[1:]))
            near = [sum(np.linalg.norm(np.subtract(w['xy'], q['xy'])) < 1. for w in q['walkers']) for q in tail]
            held = [sum(w['held'] for w in q['walkers']) for q in tail]
            stats.append(dict(hold=np.mean([q['chosen'] == 'hold' for q in tail]), path=path, near=np.mean(near),
                              held=np.mean(held), dead_end=_dead_end(r, tail[-1]['xy'])))
        out['timeouts'] = dict(
            count=len(timeouts), hold_share_last_60_s=round(float(np.mean([s['hold'] for s in stats])), 3),
            path_m_last_60_s_median=round(float(np.median([s['path'] for s in stats])), 2),
            walkers_within_1_m_mean=round(float(np.mean([s['near'] for s in stats])), 2),
            walkers_held_by_robot_mean=round(float(np.mean([s['held'] for s in stats])), 2),
            ended_in_dead_end=sum(s['dead_end'] for s in stats))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--from', dest='source', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--exits', nargs='+', default=['contact_walker', 'timeout'])
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--workers', type=int)
    s = sub.add_parser('summary')
    s.add_argument('folders', type=Path, nargs='+')
    a = p.parse_args()
    if a.command == 'run':
        report = run(a.out, model=a.model, source=a.source, exits=a.exits, seed=a.seed, workers=a.workers)
    else:
        report = {str(f): summary([json.loads(x.read_text()) for x in sorted(f.glob('[0-9]*_*.json'))])
                  for f in a.folders}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
