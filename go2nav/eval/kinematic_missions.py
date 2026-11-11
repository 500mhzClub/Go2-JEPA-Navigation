"""Round-trip missions in the kinematic world under the sampling planner (docs/dynamic/PLAN.md P17): what a suite
allows a planner whose model is the truth.

    python -m go2nav.eval.kinematic_missions run --out DIR --kind rude|polite|reactive|none --walkers 1 2
        --horizon 6.4 --terminal static|walkers [--set validation] [--mazes 10-29] [--episodes 0 1] [--workers N]
    python -m go2nav.eval.kinematic_missions summary DIR [DIR ...]

A mission is the registered episode of a maze (`eval.mission.episode_spec`: home pose and beacon) with the walkers a
mission of the same set, maze, episode and count draws (`sim.obstacles.place_obstacles`; `rude`: walkers that do not
yield). The task is `planning.mission.RoundTripMission` on the true position: the beacon, then home, each held within
`ARRIVAL_RADIUS_M` for a second at a zero command, within `FRAMES` frames (480 s). The planner (`planning.sampling`)
replans every 0.4 s; while the mission asks for a hold the request is zero.

Per mission, DIR/<maze>_<episode>_<walkers>.json (a finished one is skipped on a rerun): `exit` (round_trip, timeout,
contact_wall, contact_walker), the beacon reached, simulated seconds, path, the lowest clearance kept, the seconds a
walker waited on the robot, and for the end of a mission that timed out: the seconds the way to the target had been
sealed by walkers (`sealed_end_s`) and the cells still reachable past them (`reachable_cells`; 2 or fewer: pinned).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

ARRIVAL_RADIUS_M, FRAMES = .04, 4800


def placement_for(set_name, maze, episode_index, spec, episode, walkers, kind):
    from go2nav.sim import obstacles as moving
    if not walkers or kind == 'none':
        return []
    seed = moving.placement_seed(set_name, maze, episode_index, walkers)
    return moving.place_obstacles(spec, episode, walkers, seed, polite=kind != 'rude', reactive=kind == 'reactive')


def run_mission(set_name, maze, episode_index, *, walkers, kind, horizon_s, terminal, seed=0, frames=FRAMES,
                model='true', max_speed=None, cells=True):
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.mission import RoundTripMission
    from go2nav.planning.sampling import REPLAN_FRAMES, Planner
    from go2nav.sim.kinematic import FRAME_S, World
    spec, episode = episode_spec(set_name, maze, episode_index)
    world = World(spec, episode['home_se2_world'], placement_for(set_name, maze, episode_index, spec, episode, walkers, kind))
    mission = RoundTripMission(episode['beacon_xy_world'], episode['home_se2_world'][:2],
                               arrival_radius_m=ARRIVAL_RADIUS_M, navigation_frames=frames)
    planner = Planner(world.geometry, horizon_s=horizon_s, terminal=terminal, seed=seed, model=model,
                      max_speed=max_speed, cells=cells)
    requested, queue, plans, spent, sealed_s, chosen = np.zeros(3, np.float32), [], 0, 0., 0., {}
    frame = 0
    observes, tracked = hasattr(planner.model, 'observe'), []
    while True:
        state = mission.advance(world.xy, frame=frame, requested=requested)
        if state['terminal'] is not None:
            break
        if observes:
            planner.model.observe(frame*FRAME_S, world.scan(), world.xy, world.yaw, world.geometry)
            tracked += tracking_errors(planner.model.tracker, world)
        if state['hold_required']:
            requested, queue = np.zeros(3, np.float32), []
        else:
            if not queue:
                began = time.time()
                tape, info = planner.plan(world.snapshot(), mission.target())
                spent, plans = spent+time.time()-began, plans+1
                queue = list(tape)
                kind_chosen = info['chosen'].split('@')[0].split('_wait')[0]
                kind_chosen = 'cell' if kind_chosen.startswith('cell') else ('gaussian' if kind_chosen.startswith('gaussian') else kind_chosen)
                chosen[kind_chosen] = chosen.get(kind_chosen, 0)+1
            requested = queue.pop(0)
        world.step(requested)
        if world.contact is not None:
            break
        if frame % 10 == 0 and world.walkers:      # how long the way has been sealed, for a mission that times out
            fields = planner.fields
            value = float(fields.value(fields.with_walkers(mission.target(), world.walkers_xy()), world.xy[None])[0])
            sealed_s = sealed_s+1. if not np.isfinite(value) else 0.
        frame += 1
    round_trip = state['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    exit_ = 'round_trip' if round_trip else (f'contact_{world.contact}' if world.contact else 'timeout')
    reachable = None
    if exit_ == 'timeout' and world.walkers:
        fields = planner.fields
        field = fields.with_walkers(world.xy, world.walkers_xy())
        reachable = int(sum(np.isfinite(float(fields.value(field, centre[None])[0])) for centre in planner.cells.values()))
    return dict(set=set_name, maze=int(maze), episode=int(episode_index), walkers=int(walkers), kind=kind,
                horizon_s=horizon_s, terminal=terminal, model=model, max_speed=max_speed, cells=cells, exit=exit_,
                round_trip=bool(round_trip),
                beacon=bool(any(a['phase'] == 'OUTBOUND' for a in state['arrivals'])),
                simulated_s=round(world.frame*FRAME_S, 1), path_m=round(world.path_m, 2),
                minimum_clearance_m=round(float(world.minimum_clearance), 3),
                waiting_s=round(world.waiting_frames*FRAME_S, 1), sealed_end_s=sealed_s, reachable_cells=reachable,
                plans=plans, plan_s=round(spent/max(plans, 1), 3), wall_s=round(spent, 1), chosen=chosen,
                replan_frames=REPLAN_FRAMES, **({'tracking': tracking_summary(tracked)} if observes else {}))


def tracking_errors(tracker, world):
    """For each walker in the sensor's line of sight this frame: (the nearest track's distance or None when no track
    is within 0.4 m, that track's velocity error, the walker's speed)."""
    from go2nav.perception.ring_walkers import GATE_M, mount_of
    origin, (at, velocity) = mount_of(world.xy, world.yaw), tracker.state()
    rows = []
    for xy, v in zip(world.walkers_xy(), world.walkers_velocity):
        if not world.geometry.visible(origin, xy):
            continue
        distance = np.linalg.norm(at-xy, axis=1) if len(at) else np.array([np.inf])
        k = int(distance.argmin())
        found = bool(distance[k] <= GATE_M)
        rows.append((float(distance[k]) if found else None, float(np.linalg.norm(velocity[k]-v)) if found else None,
                     float(np.linalg.norm(v))))
    return rows


def tracking_summary(rows):
    """Over a mission's walker-frames in sight: how many, how many with no track, and the tracks' mean position
    error (m) and median and root-mean-square velocity error (m/s; the walker's velocity is the world's, over the last
    frame)."""
    found = [r for r in rows if r[0] is not None]
    return dict(in_sight=len(rows), missed=len(rows)-len(found),
                position_m=round(float(np.mean([r[0] for r in found])), 4) if found else None,
                velocity_median_m_s=round(float(np.median([r[1] for r in found])), 4) if found else None,
                velocity_rms_m_s=round(float(np.sqrt(np.mean([r[1]**2 for r in found]))), 4) if found else None,
                speed_rms_m_s=round(float(np.sqrt(np.mean([r[2]**2 for r in found]))), 4) if found else None)


def _job(args):
    out, key, kwargs = args
    target = Path(out)/f'{key[1]:02d}_{key[2]}_{key[3]}.json'
    if target.exists():
        return json.loads(target.read_text())
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    row = run_mission(key[0], key[1], key[2], walkers=key[3], **kwargs)
    partial = target.with_suffix('.tmp')
    partial.write_text(json.dumps(row, indent=1)+'\n')
    partial.rename(target)
    return row


def summary(rows):
    """Counts per walker count and in all."""
    out = {}
    groups = {str(n): [r for r in rows if r['walkers'] == n] for n in sorted({r['walkers'] for r in rows})} | dict(all=rows)
    for name, group in groups.items():
        failed = [r for r in group if not r['round_trip']]
        out[name] = dict(
            missions=len(group), round_trips=sum(r['round_trip'] for r in group),
            exits={e: sum(r['exit'] == e for r in group) for e in sorted({r['exit'] for r in group})},
            pinned=sum(r['exit'] == 'timeout' and r['reachable_cells'] is not None and r['reachable_cells'] <= 2
                       and r['sealed_end_s'] >= 30 for r in group),
            sealed_at_end=sum(r['exit'] == 'timeout' and r['sealed_end_s'] >= 30 for r in group),
            median_round_trip_s=float(np.median([r['simulated_s'] for r in group if r['round_trip']] or [np.nan])),
            median_waiting_s_failed=float(np.median([r['waiting_s'] for r in failed] or [np.nan])),
            lowest_clearance_m=min((r['minimum_clearance_m'] for r in group), default=None),
            under_10_cm=sum(r['round_trip'] and r['minimum_clearance_m'] < .10 for r in group),
            mean_plan_s=round(float(np.mean([r['plan_s'] for r in group])), 3))
    return out


def run(out, *, set_name, mazes, episodes, walkers, kind, horizon_s, terminal, workers=None, seed=0, model='true',
        max_speed=None, cells=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(set=set_name, mazes=list(mazes), episodes=list(episodes), walkers=list(walkers), kind=kind,
                  horizon_s=horizon_s, terminal=terminal, seed=seed, model=model, max_speed=max_speed, cells=cells)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    kwargs = dict(kind=kind, horizon_s=horizon_s, terminal=terminal, seed=seed, model=model, max_speed=max_speed,
                  cells=cells)
    jobs = [(str(out), (set_name, m, e, n), kwargs) for n in walkers for m in mazes for e in episodes]
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(_job, jobs))
    report = dict(config=config, summary=summary(rows))
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def span(text):
    a, _, b = text.partition('-')
    return list(range(int(a), int(b or a)+1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--set', default='validation')
    r.add_argument('--mazes', type=span, default=span('10-29'))
    r.add_argument('--episodes', type=int, nargs='+', default=[0, 1])
    r.add_argument('--walkers', type=int, nargs='+', required=True)
    r.add_argument('--kind', choices=('rude', 'polite', 'reactive', 'none'), required=True)
    r.add_argument('--horizon', type=float, required=True)
    r.add_argument('--terminal', choices=('static', 'walkers'), required=True)
    r.add_argument('--workers', type=int)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--model', default='true', help='the walkers\' part of the planner\'s model (planning.sampling.MODELS)')
    r.add_argument('--max-speed', type=float, help='the fastest forward command the planner asks for (default: 0.4)')
    r.add_argument('--no-cells', action='store_true', help='without the walk-to-a-cell proposals')
    s = sub.add_parser('summary')
    s.add_argument('folders', type=Path, nargs='+')
    a = p.parse_args()
    if a.command == 'summary':
        for folder in a.folders:
            rows = [json.loads(f.read_text()) for f in sorted(folder.glob('[0-9]*_*.json'))]
            print(folder.name, json.dumps(summary(rows)['all']))
        return
    report = run(a.out, set_name=a.set, mazes=a.mazes, episodes=a.episodes, walkers=a.walkers, kind=a.kind,
                 horizon_s=a.horizon, terminal=a.terminal, workers=a.workers, seed=a.seed, model=a.model,
                 max_speed=a.max_speed, cells=not a.no_cells)
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
