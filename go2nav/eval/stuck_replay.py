"""Replay a cohort's deadlocks: the decisions where an arm stood still because every moving candidate was BLOCKED, and
what other forecasters (or the simulator itself) would have said there.

    python -m go2nav.eval.stuck_replay list --cohort DIR[:REFERENCE_DIR] [...] --out LIST.json
    python -m go2nav.eval.stuck_replay forecast --list LIST.json --out DIR --model NAME=WORLD.pt,FIT.pt [...]
                                                [--logged NAME] [--min-stuck 100] [--workers 4]
    python -m go2nav.eval.stuck_replay summary DIR
    python -m go2nav.eval.stuck_replay counterfactual --list LIST.json --out DIR [--min-stuck 100] [--workers 4]
    python -m go2nav.eval.stuck_replay evaluate --list LIST.json --dir DIR --model NAME=WORLD.pt,FIT.pt [...] --out FILE

Built for span8w C5's deadlocks (docs/lewm/STATUS.md, span8w benchmarks and the decoder fix), for any LeWM arm.

**List** (`list`): the failed episodes of each cohort (with `:REFERENCE_DIR`, only those the reference cohort passed on
the same episode) and their stuck decisions: hold, with every moving candidate BLOCKED after the turn re-checks.

**Forecast** (`forecast`): per episode with at least `--min-stuck` stuck decisions, the run's primary frames are
rerendered up to the last sampled decision (`go2nav.data.rerender` on the truncated request tape, physics checked
against the trace), and the first 10 stuck decisions and those about 25, 50, 100, 200 and 300 decisions later are
replayed through each model's C5 and C6. The wall is a half-plane fitted per decision to the logged segment clearances
of the logged forecasts (the residual is reported); each forecast is then scored by the planner's `clearance_check`
and `reserve_turns`. `--logged NAME`: the deployed model, whose C5 replay is compared with the logged forecast.
**Summary** (`summary`): escapes (a moving candidate not BLOCKED) per forecaster, at the first stuck decision and over
all sampled ones, and the turn forecasts' dip below the start clearance and largest translation.

**Counterfactual** (`counterfactual`): the ground truth at the first stuck decision and the first one 50 or more
decisions later. The run is replayed to the decision, the simulator snapshotted (`go2nav.sim.snapshot`, as the C0
oracle) and every candidate's tape executed for 800 ms, rendering its frames: DIR/<episode>/d<decision>.npz holds
`motion` (6, 8, 3) (true body-frame x, y, yaw at 100-800 ms), `frames` (6, 8, H, W, 3) and the history frames
(4, H, W, 3). **Evaluate** (`evaluate`) scores, against it and under the same planner rules, the truth (a perfect
forecaster), each model's C5 and C6, and C5's decoder on the true future frames' z (`C5_<name>_truez`: the predictor
replaced by the encoder of what happened): escapes, and per in-place turn the largest translation error against the
truth, the dip below the start clearance and the largest translation.

Each episode's work runs in its own process (Genesis). A rerun skips finished episodes and decisions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
TURNS = (4, 5)  # left_turn, right_turn
FIRST_STUCK, LATER_STUCK, COUNTERFACTUAL_LATER = 10, (25, 50, 100, 200, 300), 50
FRAME_NS = 100_000_000


# -- list --------------------------------------------------------------------------------------------------------------

def episode_key(assignment):
    return re.search(r'_(\d+)_ep(\d+)$', assignment).groups()


def stuck_decisions(planning):
    """Indices of the decisions that held because every moving candidate was BLOCKED."""
    return [i for i, d in enumerate(planning) if d.get('selection') and d['action'] == 'hold'
            and d['selection'].get('memory_forecast_candidates')
            and all(c['clearance_check_mode'] == 'BLOCKED' for c in d['selection']['memory_forecast_candidates']
                    if c['action'] != 'hold')]


def list_stuck(cohorts, out):
    """`cohorts`: 'DIR' or 'DIR:REFERENCE_DIR' strings (see the module docstring)."""
    entries = []
    for spec in cohorts:
        folder, _, reference = spec.partition(':')
        folder = Path(folder)
        rows = {episode_key(r['assignment']): r for r in json.loads((folder/'result.json').read_text())['rows']}
        passed = ({episode_key(r['assignment']): r['round_trip'] for r in
                   json.loads((Path(reference)/'result.json').read_text())['rows']} if reference else None)
        for key, row in rows.items():
            if row['round_trip'] or (passed is not None and not passed.get(key)):
                continue
            run = folder/'runs'/row['assignment']
            planning = json.loads((run/'planning.json').read_text())
            stuck = stuck_decisions(planning)
            entries.append(dict(name=f'{folder.name}_{key[0]}_ep{key[1]}', run=str(run), contacts=row['contacts'],
                                decisions=len(planning), stuck=len(stuck), stuck_idx=stuck))
            print(f"{entries[-1]['name']}: contacts {row['contacts']}, {len(planning)} decisions, {len(stuck)} stuck",
                  flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(entries)+'\n')
    return entries


def deadlocked(list_path, min_stuck):
    return [e for e in json.loads(Path(list_path).read_text()) if e['stuck'] >= min_stuck]


# -- the planner's rules on a forecast ---------------------------------------------------------------------------------

def fit_wall(decision):
    """Half-plane c - n.p (body frame) fitted to the logged segment clearances of the logged forecasts: (c, angle) and
    the largest residual (m)."""
    from scipy.optimize import least_squares
    forecast = np.asarray(decision['forecast_xy_m'])
    starts, ends, values = [], [], []
    for i, candidate in enumerate(decision['selection']['memory_forecast_candidates']):
        points = np.vstack([np.zeros(2), forecast[i]])
        for k, v in enumerate(candidate['segment_clearances_m']):
            if v is not None:
                starts.append(points[k])
                ends.append(points[k+1])
                values.append(v)
    a, b, values = np.asarray(starts), np.asarray(ends), np.asarray(values)

    def predicted(x):
        n = np.array([np.cos(x[1]), np.sin(x[1])])
        return x[0]-np.maximum(a@n, b@n)
    best = min((least_squares(lambda x: predicted(x)-values, [values.max(), angle]) for angle in np.linspace(-3, 3, 12)),
               key=lambda r: r.cost)
    return best.x, float(np.abs(predicted(best.x)-values).max())


class HalfPlane:
    """The fitted wall as the planner's clearance source."""

    def __init__(self, x):
        self.c, self.n = x[0], np.array([np.cos(x[1]), np.sin(x[1])])

    def minimum(self, a, b):
        return float(self.c-max(self.n@np.asarray(a, float), self.n@np.asarray(b, float)))


def score(decision, forecast, clearance):
    """The planner's clearance check, then the turn re-checks (with exits), on a (6, 8, 5) forecast: per action, its
    mode and segment clearances."""
    from go2nav.planning import selection
    base = {k: v for k, v in decision['selection'].items()
            if k in ('candidates', 'waypoint_body_xy_m', 'alignment_scale_m', 'selection_objective')}
    base = deepcopy(base) | dict(action=decision['selection']['before_memory_filter_action'])
    base['action_index'] = selection.ACTIONS.index(base['action'])
    result = selection.reserve_turns(selection.clearance_check(base, forecast, clearance, np.zeros(3), np.eye(3)))
    return {c['action']: dict(mode=c['clearance_check_mode'], seg=[round(v, 4) for v in c['segment_clearances_m']])
            for c in result['memory_forecast_candidates']}


def as_forecast(motion):
    """(6, 8, 3) x, y, yaw to the planner's (6, 8, 5) forecast (contact not predicted)."""
    m = np.asarray(motion, np.float32)
    return np.concatenate((m[..., :2], np.sin(m[..., 2:3]), np.cos(m[..., 2:3]), np.full((*m.shape[:2], 1), -1000.,
                                                                                            np.float32)), -1)


# -- forecasters -------------------------------------------------------------------------------------------------------

def parse_models(specs):
    """NAME=WORLD.pt,FIT.pt strings to {name: (world, fit)}."""
    models = {}
    for spec in specs:
        name, _, paths = spec.partition('=')
        world, _, fit = paths.partition(',')
        if not (name and world and fit):
            raise ValueError(f'expected NAME=WORLD.pt,FIT.pt, got {spec!r}')
        models[name] = (world, fit)
    return models


def forecasters(models, device='cuda:0'):
    """C5_<name> and C6_<name> for each model."""
    from go2nav.controllers.lewm import LeWMDirectForecaster, LeWMForecaster
    from go2nav.models.lewm import load_heads, load_world_model
    out = {}
    for name, (world, fit) in models.items():
        model, (c5, c6) = load_world_model(world), load_heads(fit)
        out[f'C5_{name}'] = LeWMForecaster(model, c5, device=device)
        out[f'C6_{name}'] = LeWMDirectForecaster(model, c6, device=device)
    return out


def forecast_context(histories, decision, frames):
    """The forecasters' input at a decision: four frames (t-300 .. t ms) and the applied-command history."""
    j = int(np.flatnonzero(histories['image_ns'] == decision['measured_ns'])[0])
    if not histories['applied_command_valid'][j].all():
        raise ValueError('incomplete applied history')
    applied = histories['applied_command_values'][j]
    return SimpleNamespace(history=[dict(image=dict(measured_ns=decision['measured_ns']-(3-k)*FRAME_NS, rgb=frames[k]),
                                         sensor_state=dict(control=dict(applied_command=dict(values=applied))))
                                    for k in range(4)])


# -- forecast ----------------------------------------------------------------------------------------------------------

def render(run, out, last_frame):
    """Rerender a run's primary frames up to `last_frame` (its own process: Genesis)."""
    import time
    from go2nav.data import rerender
    run, out = Path(run), Path(out)
    if (out/'.done').exists():
        return
    original = rerender.load

    def truncated(r, name):
        value = original(r, name)
        if name == 'requests.json':
            return value[:(last_frame+1)*5]
        if name == 'acquisitions.json':
            return value[:last_frame+1]
        return value
    rerender.load = truncated
    started = time.time()
    frames = rerender.replay(run, out)
    (out/'.done').write_text(json.dumps(dict(frames=frames, wall_s=round(time.time()-started))))


def sampled(stuck):
    """The first stuck decisions and those about LATER_STUCK decisions after the first."""
    later = [next(i for i in stuck if i >= stuck[0]+k) for k in LATER_STUCK if any(i >= stuck[0]+k for i in stuck)]
    return sorted(set(stuck[:FIRST_STUCK]+later))


def replay_episode(entry, frames_dir, out, models, logged=None):
    """Score the logged forecast and every forecaster at the sampled stuck decisions (see the module docstring)."""
    from PIL import Image
    run, frames_dir = Path(entry['run']), Path(frames_dir)
    planning = json.loads((run/'planning.json').read_text())
    histories = dict(np.load(run/'native/policy_histories.npz'))
    sources = forecasters(models)
    rows = []
    for i in sampled(entry['stuck_idx']):
        d = planning[i]
        x, residual = fit_wall(d)
        wall = HalfPlane(x)
        frames = [np.asarray(Image.open(frames_dir/f"ego_frames/{d['frame']-3+k:04d}.png")) for k in range(4)]
        context = forecast_context(histories, d, frames)
        logged_xy = np.asarray(d['forecast_xy_m'])
        row = dict(decision=i, frame=d['frame'], wall_offset_m=round(float(x[0]), 4), wall_resid_mm=round(1000*residual, 2),
                   logged=score(d, np.concatenate([logged_xy, np.zeros((6, 8, 3))], -1), wall))
        for name, source in sources.items():
            prediction = source.forecast(context, d['committed_prefix'], pulse=False)
            row[name] = score(d, prediction, wall)
            row[name+'_xy'] = np.round(prediction[:, :, :2], 4).tolist()
            if name == f'C5_{logged}':
                row['replay_vs_log_mm'] = round(1000*float(np.abs(prediction[:, :, :2]-logged_xy).max()), 2)
        rows.append(row)
        print(json.dumps(dict(decision=i, resid_mm=row['wall_resid_mm'], replay_vs_log_mm=row.get('replay_vs_log_mm'),
                              clear={k: [a for a, v in row[k].items() if a != 'hold' and v['mode'] != 'BLOCKED']
                                     for k in ('logged', *sources)})), flush=True)
    Path(out).write_text(json.dumps(rows)+'\n')


def forecast_all(list_path, out, model_specs, *, logged=None, min_stuck=100, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for entry in deadlocked(list_path, min_stuck):
        planning = json.loads((Path(entry['run'])/'planning.json').read_text())
        jobs.append((entry, max(planning[i]['frame'] for i in sampled(entry['stuck_idx']))))

    def render_job(job):
        entry, last = job
        with open(out/f"{entry['name']}.render.log", 'w') as log:
            subprocess.run([sys.executable, '-m', __spec__.name, '_render', entry['run'], str(out/entry['name']), str(last)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        print('rendered', entry['name'], flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(render_job, jobs))
    models = parse_models(model_specs)
    for entry, _ in jobs:
        target = out/f"{entry['name']}.json"
        if not target.exists():
            replay_episode(entry, out/entry['name'], target, models, logged)


def summary(folder):
    """Escapes per forecaster and the turn forecasts' dip and translation (see the module docstring)."""
    episodes = {f.stem: rows for f in sorted(Path(folder).glob('*.json'))
                for rows in [json.loads(f.read_text())] if isinstance(rows, list) and rows and 'decision' in rows[0]}
    if not episodes:
        raise FileNotFoundError(f'no replayed episodes in {folder}')
    first = next(iter(episodes.values()))[0]
    sources = ['logged']+[k for k in first if k.startswith(('C5_', 'C6_')) and not k.endswith('_xy')]
    totals = {s: [0, 0, 0] for s in sources}
    dips, translations, residuals = {s: [] for s in sources[1:]}, {s: [] for s in sources[1:]}, []
    for name, rows in episodes.items():
        line = f'{name:28s}'
        for s in sources:
            escapes = [any(v['mode'] != 'BLOCKED' for k, v in r[s].items() if k != 'hold') for r in rows]
            totals[s][0] += escapes[0]
            totals[s][1] += sum(escapes)
            totals[s][2] += len(escapes)
            line += f' {s}: {"Y" if escapes[0] else "n"} {sum(escapes):2d}/{len(escapes)}'
        for r in rows:
            residuals.append(r['wall_resid_mm'])
            for s in sources[1:]:
                for t in TURNS:
                    segments = r[s][ACTIONS[t]]['seg']
                    dips[s].append(1000*(segments[0]-min(segments)))
                    translations[s].append(1000*max(math.hypot(x, y) for x, y in r[s+'_xy'][t]))
        print(line)
    print('\ntotals (first stuck decision escapes / sampled decisions with an escape):')
    for s in sources:
        print(f'  {s:16s} first {totals[s][0]}/{len(episodes)}  all {totals[s][1]}/{totals[s][2]}')
    q = lambda v: f'median {statistics.median(v):.1f} p90 {statistics.quantiles(v, n=10)[8]:.1f}'
    print('\nturn forecasts: dip below start clearance (mm) | largest translation (mm)')
    for s in sources[1:]:
        print(f'  {s:16s} {q(dips[s])} | {q(translations[s])}')
    print(f'\nwall fit residual: median {statistics.median(residuals):.2f} mm, max {max(residuals):.2f} mm')
    return dict(totals=totals)


# -- counterfactual ----------------------------------------------------------------------------------------------------

def counterfactual(run, out, decisions):
    """The ground truth at `decisions` of a run (its own process: Genesis; see the module docstring)."""
    from go2nav.controllers.oracle import encoded_tapes
    from go2nav.data import rerender
    from go2nav.sim import scene, snapshot as snapshots
    from go2nav.sim.sensors import rotation_xyzw
    from go2nav.sim.session import MissionSession
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    planning = json.loads((run/'planning.json').read_text())
    todo = {planning[i]['frame']: i for i in decisions if not (out/f'd{i}.npz').exists()}
    if not todo:
        return
    spec, requests = rerender.load(run, 'specification.json'), rerender.load(run, 'requests.json')
    with np.load(run/'native/physics_trace.npz') as archive:
        trace = {k: archive[k].copy() for k in rerender.TRACE_FIELDS}
    scene.init_genesis(spec['procedural_seed'])
    session = MissionSession(spec, out/'visual_meshes', policy_checkpoint=rerender.run_policy(run),
                             dynamics=rerender.run_dynamics(run), demo_textures=rerender.run_demo_textures(run))
    session.settle()
    checked = rerender.check_samples(session, trace, 0)
    recent, last = [], max(todo)
    for tick, row in enumerate(requests):
        if tick % 5 == 0:
            frame = session.frames
            recent = (recent+[session.frame()[0]['image']['rgb']])[-4:]
            if frame in todo:
                i = todo[frame]
                if planning[i]['measured_ns'] != session.sim_ns:
                    raise ValueError('decision time mismatch')
                snap = snapshots.capture(session)
                records = {name: list(getattr(session, name)) for name in snapshots.RECORD_LISTS}
                initial = session.samples[-1]['base_pose_world'].copy()
                rotation = rotation_xyzw(initial[3:])
                motions, frames = [], []
                for tape in encoded_tapes(planning[i]['committed_prefix'], False):
                    snapshots.restore(session, snap)
                    for name, value in records.items():
                        setattr(session, name, list(value))
                    session.frames = frame+1
                    start, shots = len(session.samples), []
                    for k, command in enumerate(np.repeat(tape, 5, axis=0)):
                        session.step(command)
                        if k % 5 == 4:
                            shots.append(session.frame()[0]['image']['rgb'])
                    poses = np.stack([s['base_pose_world'] for s in session.samples[start:]])
                    horizon = poses[np.arange(49, 400, 50)]
                    xy = (horizon[:, :3]-initial[:3])@rotation
                    relative = [rotation.T@rotation_xyzw(p[3:]) for p in horizon]
                    motions.append(np.c_[xy[:, :2], [math.atan2(r[1, 0], r[0, 0]) for r in relative]])
                    frames.append(np.stack(shots))
                snapshots.restore(session, snap)
                for name, value in records.items():
                    setattr(session, name, value)
                session.frames = frame+1
                np.savez_compressed(out/f'd{i}.npz', motion=np.asarray(motions, np.float32), frames=np.asarray(frames),
                                    history=np.stack(recent), decision=i, frame=frame)
                print(json.dumps(dict(decision=i, frame=frame)), flush=True)
                if frame == last:
                    return
        session.phase = 2
        session.step(row['requested_command'])
        checked = rerender.check_samples(session, trace, checked)


def counterfactual_all(list_path, out, *, min_stuck=100, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for entry in deadlocked(list_path, min_stuck):
        stuck = entry['stuck_idx']
        later = next((i for i in stuck if i >= stuck[0]+COUNTERFACTUAL_LATER), None)
        jobs.append((entry, [stuck[0]]+([later] if later is not None else [])))

    def one(job):
        entry, decisions = job
        with open(out/f"{entry['name']}.log", 'w') as log:
            subprocess.run([sys.executable, '-m', __spec__.name, '_counterfactual', entry['run'], str(out/entry['name']),
                            ','.join(map(str, decisions))], check=True, stdout=log, stderr=subprocess.STDOUT)
        print('done', entry['name'], flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))


def evaluate(list_path, folder, model_specs, out, device='cuda:0'):
    """Score the counterfactual ground truth against the forecasters (see the module docstring)."""
    import torch
    from go2nav.models.lewm import normalise_frames, resize_frame
    runs = {e['name']: Path(e['run']) for e in json.loads(Path(list_path).read_text())}
    sources = forecasters(parse_models(model_specs), device)
    rows = []
    for episode in sorted(p for p in Path(folder).iterdir() if p.is_dir() and p.name in runs):
        run = runs[episode.name]
        planning = json.loads((run/'planning.json').read_text())
        histories = dict(np.load(run/'native/policy_histories.npz'))
        for path in sorted(episode.glob('d*.npz')):
            truth = np.load(path)
            d = planning[int(truth['decision'])]
            context = forecast_context(histories, d, truth['history'])
            x, residual = fit_wall(d)
            wall = HalfPlane(x)
            predictions = dict(truth=as_forecast(truth['motion']))
            for name, source in sources.items():
                predictions[name] = source.forecast(context, d['committed_prefix'], pulse=False)
                if name.startswith('C5_'):
                    zh = source.inputs(context, d['committed_prefix'], False)[0]
                    future = truth['frames'].reshape(-1, *truth['frames'].shape[2:])
                    with torch.inference_mode():
                        pixels = torch.from_numpy(np.stack([resize_frame(f) for f in future])).to(source.device)
                        with torch.autocast(source.device.type, dtype=torch.bfloat16):
                            z = source.model.encode(normalise_frames(pixels)[:, None])[:, 0].float()
                        motion = source.decoder(zh.expand(48, -1, -1), z).reshape(6, 8, 3).cpu().numpy()
                    predictions[name+'_truez'] = as_forecast(motion)
            row = dict(case=episode.name, decision=int(truth['decision']), resid_mm=round(1000*residual, 2))
            reference = predictions['truth']
            for name, p in predictions.items():
                modes = score(d, p, wall)
                row[name] = dict(escape=[a for a in ACTIONS[1:] if modes[a]['mode'] != 'BLOCKED'],
                                 turn_err_mm=[float(1000*np.linalg.norm(p[i, :, :2]-reference[i, :, :2], axis=1).max())
                                              for i in TURNS],
                                 turn_dip_mm=[float(1000*(modes[ACTIONS[i]]['seg'][0]-min(modes[ACTIONS[i]]['seg'])))
                                              for i in TURNS],
                                 turn_trans_mm=[float(1000*np.linalg.norm(p[i, :, :2], axis=1).max()) for i in TURNS],
                                 xy=np.round(p[:, :, :2], 4).tolist())
            rows.append(row)
            print(row['case'], row['decision'], {k: row[k]['escape'] for k in predictions}, flush=True)
    Path(out).write_text(json.dumps(rows)+'\n')
    q = lambda v: f'{statistics.median(v):5.1f} / {statistics.quantiles(v, n=10)[8]:5.1f}'
    print(f'\n{len(rows)} decisions; wall fit residual median {statistics.median(r["resid_mm"] for r in rows):.2f} mm')
    print(f'{"source":22s} escapes  turn error vs truth  turn dip  turn translation (median / p90 mm)')
    for name in (k for k in rows[0] if isinstance(rows[0][k], dict)):
        v = [r[name] for r in rows]
        print(f'{name:22s} {sum(bool(r["escape"]) for r in v):3d}/{len(v)}   '
              f'{q([x for r in v for x in r["turn_err_mm"]])}   {q([x for r in v for x in r["turn_dip_mm"]])}   '
              f'{q([x for r in v for x in r["turn_trans_mm"]])}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('list')
    s.add_argument('--cohort', action='append', required=True, metavar='DIR[:REFERENCE_DIR]')
    s.add_argument('--out', required=True, type=Path)
    for name in ('forecast', 'counterfactual'):
        s = sub.add_parser(name)
        s.add_argument('--list', required=True, type=Path)
        s.add_argument('--out', required=True, type=Path)
        s.add_argument('--min-stuck', type=int, default=100)
        s.add_argument('--workers', type=int, default=4)
        if name == 'forecast':
            s.add_argument('--model', action='append', required=True, metavar='NAME=WORLD.pt,FIT.pt')
            s.add_argument('--logged', help='the model whose C5 ran the cohort (its replay is checked against the log)')
    s = sub.add_parser('summary')
    s.add_argument('dir', type=Path)
    s = sub.add_parser('evaluate')
    s.add_argument('--list', required=True, type=Path)
    s.add_argument('--dir', required=True, type=Path)
    s.add_argument('--model', action='append', required=True, metavar='NAME=WORLD.pt,FIT.pt')
    s.add_argument('--out', required=True, type=Path)
    for name, args in (('_render', ('run', 'out', 'last_frame')), ('_counterfactual', ('run', 'out', 'decisions'))):
        s = sub.add_parser(name)  # internal: one episode in its own process
        for arg in args:
            s.add_argument(arg)
    a = p.parse_args()
    if a.command == 'list':
        list_stuck(a.cohort, a.out)
    elif a.command == 'forecast':
        forecast_all(a.list, a.out, a.model, logged=a.logged, min_stuck=a.min_stuck, workers=a.workers)
    elif a.command == 'summary':
        summary(a.dir)
    elif a.command == 'counterfactual':
        counterfactual_all(a.list, a.out, min_stuck=a.min_stuck, workers=a.workers)
    elif a.command == 'evaluate':
        evaluate(a.list, a.dir, a.model, a.out)
    elif a.command == '_render':
        render(a.run, a.out, int(a.last_frame))
    else:
        counterfactual(a.run, a.out, [int(v) for v in a.decisions.split(',')])


if __name__ == '__main__':
    main()
