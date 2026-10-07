"""The T7/T8 (C3 decoder and matched C4) contexts, derived from recordings and mission runs (data/training/decoder).

    python -m go2nav.data.samples decoder --recordings-root ROOT [...] --out contexts.jsonl

A context is a decision frame t of a recording or a mission run with: its three frames (1 s, 0.5 s and 0 s before),
the executed applied-command tape of the eight 100-ms steps after t, the applied-command history at t (15 x 3), the
true body-frame motion at 100-800 ms, its movement category and the horizons it is used at. Mission runs are read
from runs/<run> of the capability folder, their frames from the run's verified replay (`go2nav.data.rerender`). A
context needs t >= 10, frames up to t+8 and an executed tape that is complete for all eight steps; most sets also
need a complete causal command history at t (`causal_complete`). Decisions of a mission are its planning records with
a selection; missions with a disallowed contact contribute nothing to the on-policy and stage-2 sets.

Sets, in order (decoder fix, then stage 2):
- `train`: `old` = the T5b old contexts, `maze` = the T5b maze contexts and the fit-split rest/turn recording
  contexts, `onpolicy` = every context of the 16 C3-v3 fit missions (C1), subsampled per movement category; two
  random horizons each (seeded);
- `eval_onpolicy`: the decisions of the 6 held-out C3-v3 missions, at 500 and 800 ms;
- `eval_offline`: the held-out rest/turn recording contexts, at 500 and 800 ms;
- `eval_transfer`: the maze-view transfer collection's departures, motions at 500 and 700 ms only;
- `eval_fresh_c3`: the decisions of the 10 C3-v2 fresh-check missions (C3's own states), at 800 ms;
- stage 2, from the s2rec2 recordings (C1 on marked low-friction strips): decisions whose next 700 ms request motion
  and contain no obstacle veto, binned by the body's signed distance to the nearest strip edge at t and t + 700 ms
  (approach, entry, on_patch, exit, off_patch): fit mazes give `train` contexts in groups `patch` and `patch_off`
  (at most 4,000, a seeded subset, appended last), held-out mazes give `eval_patch` at 500 and 800 ms.
"""
from dataclasses import dataclass
import json

import numpy as np

from go2nav.data.samples import COLLECTIONS, HORIZON, body_motion, t5_maze_view

CAPABILITY = 'go2_navigation_capability_v1_attempt_001'
REST_TURN = f'{CAPABILITY}/c3v2_rest_turn_recordings_v1'
TRANSFER = 'go2_maze_view_transfer_v1_attempt_001'
REPLAYS = dict(onpolicy=f'{CAPABILITY}/c3v3_onpolicy_replays', fresh=f'{CAPABILITY}/c3v2_gap_diagnosis_v1/replays',
               stage2=f'{CAPABILITY}/stage2_recording_replays')
SEED, STAGE2_SEED = 2026093010, 2026100517
ONPOLICY_QUOTA = dict(hold=10**9, rest_start=10**9, turn=3000, cruise=10**9, arc_steady=2000, switch=4000)
ONPOLICY_FIT, ONPOLICY_HELDOUT = range(16), range(16, 22)
FRESH_CHECKS = range(10)
OFF_PATCH_CAP = 4000
STEP_NS, STAGE2_HORIZON = 100_000_000, 7
VETOES = {'CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'}


@dataclass
class Source:
    """A recording or a mission run: applied-command histories, poses and frame names."""
    key: str
    values: np.ndarray
    valid: np.ndarray
    measured: np.ndarray
    available: np.ndarray
    stamps: np.ndarray
    poses: np.ndarray
    frame_folder: str
    frame_name: str  # format of a frame file name, e.g. 'rgb_{:04d}.png'

    def frame(self, index):
        return f'{self.frame_folder}/{self.frame_name.format(index)}'

    def sample(self, frame):
        return int(self.stamps[frame])//2_000_000-1


def load_source(recordings, folder, *, replay=None):
    """A recording folder (frames rgb_NNNN.png beside it), or a run folder with its replay (ego_frames/NNNN.png)."""
    base = recordings.find(folder)
    native = base/'native' if replay else base
    with np.load(native/'policy_histories.npz') as h, np.load(native/'physics_trace.npz') as trace:
        return Source(folder, h['applied_command_values'].astype(np.float32), h['applied_command_valid'].copy(),
                      h['applied_command_measured_ns'].copy(), h['applied_command_available_ns'].copy(),
                      h['image_ns'].copy(), trace['base_pose_world'].copy(),
                      f'{replay}/ego_frames' if replay else folder, '{:04d}.png' if replay else 'rgb_{:04d}.png')


def causal_complete(source, frame):
    """The run-time causal-context checks: frames 1 s and 0.5 s back, and a complete, causal command history."""
    indices = (frame-10, frame-5, frame)
    if min(indices) < 0:
        return False
    expected = [int(source.stamps[i]) for i in indices]
    if np.diff(expected).tolist() != [500_000_000]*2:
        return False
    values, measured, available = source.values[frame], source.measured[frame], source.available[frame]
    return bool(source.valid[frame].all() and np.array_equal(measured[[4, 9, 14]], expected)
                and np.all(np.diff(measured) == 100_000_000) and not np.any(measured > expected[-1])
                and not np.any(available > expected[-1]) and not np.any(values[:, 1] != 0))


def executed_tape(source, frame):
    """The applied commands of the eight 100-ms steps after `frame`, as recorded at frame + 8 (ValueError if any
    step is missing or the records disagree)."""
    future = []
    for h in range(1, HORIZON+1):
        if not source.valid[frame+h, -h:].all():
            raise ValueError('complete executed tape required')
        commands = source.values[frame+h, -h:]
        if future and not np.array_equal(commands[:-1], np.asarray(future)):
            raise ValueError('executed tape records disagree')
        future = commands.tolist()
    return np.asarray(future, np.float32)


def motion_targets(source, frame):
    return np.asarray([body_motion(source.poses, source.sample(frame), source.sample(frame+h))
                       for h in range(1, HORIZON+1)], np.float32)


def category(tape, past):
    """Movement category of a decision: hold, rest_start, turn, cruise, arc_steady or switch."""
    tape, past = np.asarray(tape), np.asarray(past)
    forward, yaw = tape[:, 0] > 0, tape[:, 2] != 0
    if not tape.any():
        return 'hold'
    if not past[-10:].any() and forward.any():
        return 'rest_start'
    if not forward.any():
        return 'turn'
    if np.all(tape == tape[0]):
        return 'cruise' if not yaw.any() else 'arc_steady'
    return 'switch'


def item(set_name, source, frame, tape, targets, horizons, *, stage='decoder_fix', group=None, **extra):
    past = source.values[frame]
    return dict(set=set_name, group=group, category=category(tape, past), tape=np.asarray(tape, np.float32).tolist(),
                past=np.asarray(past, np.float32).tolist(), targets=np.asarray(targets, np.float32).tolist(),
                horizons=list(horizons), frames=[source.frame(f) for f in (frame-10, frame-5, frame)], stage=stage,
                extra=extra)


# -- inputs -----------------------------------------------------------------------------------------------------------

class Sources:
    def __init__(self, recordings):
        self.recordings, self.cache = recordings, {}

    def __call__(self, folder, replay=None):
        if folder not in self.cache:
            self.cache[folder] = load_source(self.recordings, folder, replay=replay)
        return self.cache[folder]


def read(recordings, name):
    return json.loads(recordings.find(name).read_text())


def run_folder(name):
    return f'{CAPABILITY}/runs/{name}'


def contact_free(recordings, run):
    failure = recordings.find(run_folder(run)).joinpath('failure.json')
    if failure.exists() and 'DISALLOWED_CONTACT' in json.loads(failure.read_text()).get('reason', ''):
        return False
    evaluation = recordings.find(run_folder(run)).joinpath('episode_evaluation.json')
    return not (evaluation.exists() and json.loads(evaluation.read_text()).get('disallowed_contact_samples'))


def decision_frames(recordings, run):
    """(frame, measured_ns) of every decision of a run (planning records with a selection)."""
    return [((p['measured_ns']-1_500_000_000)//STEP_NS, p['measured_ns'])
            for p in read(recordings, f'{run_folder(run)}/planning.json') if 'selection' in p]


def rest_turn_contexts(recordings, sources, split):
    """(source, frame, motions) of the rest/turn recordings of one split: frames 10..n-9 without contact by t + 8."""
    plan = json.loads((COLLECTIONS/'c3v2_rest_turn/plan.json').read_text())
    out = []
    for case, spec in enumerate(plan['specifications']):
        if spec['recording_split'] != split:
            continue
        source = sources(f'{REST_TURN}/case_{case:02d}')
        with np.load(recordings.find(source.key)/'physics_trace.npz') as trace:
            contact = trace['physics_contact'].copy()
        for frame in range(10, len(source.stamps)-8):
            if not contact[:source.sample(frame+8)+1].any():
                out.append((source, frame, [body_motion(source.poses, source.sample(frame), source.sample(frame+h))
                                            for h in range(1, HORIZON+1)]))
    return out


def transfer_contexts(recordings, sources):
    """(source, frame, targets, group, maze) of the maze-view transfer departures (motions at 500 and 700 ms)."""
    plan = json.loads((COLLECTIONS/'maze_view_transfer/plan.json').read_text())
    out = []
    for case, spec in enumerate(plan['specifications']):
        source = sources(f'{TRANSFER}/case_{case:02d}')
        with np.load(recordings.find(source.key)/'physics_trace.npz') as trace:
            contact = trace['physics_contact'].copy()
        for frame in sorted(plan['evaluation']['departures']):
            targets, group = np.full((HORIZON, 3), np.nan, np.float32), None
            for ms in plan['evaluation']['horizons_ms']:
                h = ms//100
                if frame+h >= len(source.stamps) or contact[:source.sample(frame+h)+1].any():
                    continue
                targets[h-1] = body_motion(source.poses, source.sample(frame), source.sample(frame+h))
                action = plan['phases'][frame].split('_', 2)[2]
                group = group or ('hold' if action == 'hold' else 'turn' if action.endswith('_turn') else 'translation')
            if group is not None:
                out.append((source, frame, targets, group, spec['evaluation_maze_index']))
    return out


def signed_edge_distance(xy, rects):
    """Signed distance from a point to the nearest strip boundary: negative inside a strip, positive outside."""
    best = np.inf
    for x0, x1, y0, y1 in rects:
        dx, dy = max(x0-xy[0], 0., xy[0]-x1), max(y0-xy[1], 0., xy[1]-y1)
        d = -min(xy[0]-x0, x1-xy[0], xy[1]-y0, y1-xy[1]) if dx == 0 and dy == 0 else float(np.hypot(dx, dy))
        best = min(best, d)
    return float(best)


def edge_bin(d0, d1):
    if d0 <= -.3:
        return 'on_patch'
    if d0 <= .3:
        return 'entry' if d1 <= d0 else 'exit'
    if d0 <= 1.5 and d1 < d0:
        return 'approach'
    return 'off_patch'


def stage2_decisions(recordings, run):
    """(frame, measured_ns, edge bin, edge distances) of a stage-2 run's decisions whose next 700 ms request motion
    and contain no obstacle veto."""
    base = recordings.find(run_folder(run))
    with np.load(base/'native/physics_trace.npz') as trace:
        stamps = np.rint(trace['timestamp_s']*1e9).astype(np.int64)
        pose, requested = trace['base_pose_world'].copy(), trace['requested_command'].copy()
    from go2nav.data.rerender import run_dynamics
    rects = run_dynamics(base).strips
    vetoed = np.array([r['simulator_ns'] for r in read(recordings, f'{run_folder(run)}/requests.json')
                       if r.get('reason') in VETOES], np.int64)
    index = {int(t): i for i, t in enumerate(stamps)}
    out = []
    for frame, t in decision_frames(recordings, run):
        steps = [index.get(t+k*STEP_NS+STEP_NS//2) for k in range(STAGE2_HORIZON)]
        start, end = index.get(t), index.get(t+STAGE2_HORIZON*STEP_NS)
        if None in steps or start is None or end is None or not np.any(np.abs(requested[steps]) > 1e-9):
            continue
        if np.any((vetoed >= t) & (vetoed < t+STAGE2_HORIZON*STEP_NS)):
            continue
        d0, d1 = signed_edge_distance(pose[start, :2], rects), signed_edge_distance(pose[end, :2], rects)
        out.append((frame, t, edge_bin(d0, d1), d0, d1))
    return out


# -- the contexts -----------------------------------------------------------------------------------------------------

def contexts(recordings):
    sources = Sources(recordings)
    rng = np.random.default_rng(SEED)
    # Training population: (group, source, frame, targets).
    rows = []
    old_and_maze = t5_maze_view(recordings)
    for group in ('old', 'maze'):
        for pairs, motions in zip(old_and_maze[group]['pairs'], old_and_maze[group]['motions']):
            folder, name = pairs[0][0].rsplit('/', 1)
            rows.append((group, sources(folder), int(name[4:8]), motions))
    rows += [('maze', source, frame, motions) for source, frame, motions in rest_turn_contexts(recordings, sources, 'fit')]
    for maze in ONPOLICY_FIT:
        run = f'c3v3_onpolicy_C1_m{maze:02d}_ep0_attempt001'
        if not contact_free(recordings, run):
            continue
        source = sources(run_folder(run), f"{REPLAYS['onpolicy']}/{run}")
        for frame in range(10, len(source.stamps)-8):
            if causal_complete(source, frame):
                try:
                    executed_tape(source, frame)
                except ValueError:
                    continue
                rows.append(('onpolicy', source, frame, motion_targets(source, frame)))
    by_category = {}
    for i, (group, source, frame, _) in enumerate(rows):
        if group == 'onpolicy':
            by_category.setdefault(category(executed_tape(source, frame), source.values[frame]), []).append(i)
    kept = {int(i) for name, indices in by_category.items() for i in rng.permutation(indices)[:ONPOLICY_QUOTA[name]]}
    items = []
    for i, (group, source, frame, targets) in enumerate(rows):
        if group == 'onpolicy' and i not in kept:
            continue
        horizons = sorted(int(x) for x in rng.choice(np.arange(1, 9), size=2, replace=False))
        items.append(item('train', source, frame, executed_tape(source, frame), targets, horizons, group=group, row=i))
    # Held-out on-policy decisions.
    for maze in ONPOLICY_HELDOUT:
        run = f'c3v3_onpolicy_C1_m{maze:02d}_ep0_attempt001'
        if not contact_free(recordings, run):
            continue
        source = sources(run_folder(run), f"{REPLAYS['onpolicy']}/{run}")
        for frame, _ in decision_frames(recordings, run):
            if frame < 10 or frame+8 >= len(source.stamps) or not causal_complete(source, frame):
                continue
            try:
                tape = executed_tape(source, frame)
            except ValueError:
                continue
            items.append(item('eval_onpolicy', source, frame, tape, motion_targets(source, frame), [5, 8], run=run,
                              from_rest=bool(not source.values[frame][-10:].any())))
    for source, frame, motions in rest_turn_contexts(recordings, sources, 'heldout'):
        items.append(item('eval_offline', source, frame, executed_tape(source, frame), motions, [5, 8],
                          directory=source.key))
    for source, frame, targets, group, maze in transfer_contexts(recordings, sources):
        items.append(item('eval_transfer', source, frame, executed_tape(source, frame), targets, [5, 7],
                          transfer_group=group, maze=maze))
    for check in FRESH_CHECKS:
        run = f'c3v2_check_C3_chk{check:02d}_ep0_attempt001'
        source = sources(run_folder(run), f"{REPLAYS['fresh']}/{run}")
        for frame, _ in decision_frames(recordings, run):
            if frame < 10 or frame+8 >= len(source.stamps) or not source.valid[frame].all():
                continue
            try:
                tape = executed_tape(source, frame)
            except ValueError:
                continue
            items.append(item('eval_fresh_c3', source, frame, tape, motion_targets(source, frame), [8], run=run))
    return items+stage2_contexts(recordings, sources)


def stage2_contexts(recordings, sources):
    rng = np.random.default_rng(STAGE2_SEED)
    plan = read(recordings, f'{CAPABILITY}/dev_cohorts/s2rec2/config.json')['plan']
    items, off_patch = [], []
    for *_, run in plan:
        if not recordings.find(run_folder(run)).joinpath('planning.json').exists() or not contact_free(recordings, run):
            continue
        episode = read(recordings, f'{run_folder(run)}/episode.json')
        role, maze = episode['role'], int(episode['maze_id'])
        source = sources(run_folder(run), f"{REPLAYS['stage2']}/{run}")
        for frame, t, edge, d0, d1 in stage2_decisions(recordings, run):
            if frame < 10 or frame+8 >= len(source.stamps) or source.stamps[frame] != t or not causal_complete(source, frame):
                continue
            try:
                tape = executed_tape(source, frame)
            except ValueError:
                continue
            fit = role == 'stage2_fit'
            horizons = sorted(int(x) for x in rng.choice(np.arange(1, 9), size=2, replace=False)) if fit else [5, 8]
            row = item('train' if fit else 'eval_patch', source, frame, tape, motion_targets(source, frame), horizons,
                       stage='stage2', group=('patch_off' if edge == 'off_patch' else 'patch') if fit else None,
                       run=run, maze=maze, role=role, edge_bin=edge, edge_distance_start_m=d0, edge_distance_end_m=d1)
            (off_patch if fit and edge == 'off_patch' else items).append(row)
    keep = rng.permutation(len(off_patch))[:OFF_PATCH_CAP]
    return items+[off_patch[k] for k in sorted(keep)]
