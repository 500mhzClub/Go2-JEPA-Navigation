"""Training samples derived from the recordings: regenerates the files of data/training/ from recorded trials.

    python -m go2nav.data.samples DATASET --recordings-root ROOT [--recordings-root ROOT ...] --out FILE.jsonl
    python -m go2nav.data.samples all --recordings-root ROOT [...] --out data/training

`all` writes every file of data/training (except T1's, which come from the May corpus) and each folder's frames.txt
(the sorted frames its files use); see LAYOUT.

A recordings root holds one folder per collection, named as in FOLDERS, with one folder per trial (as written by
`go2nav.data.corridor` and `go2nav.data.collect`, or the original recordings). A trial folder needs
physics_trace.npz and policy_histories.npz; samples name frames as <folder>/<trial>/rgb_NNNN.png.

**Context windows.** A window is a decision at camera frame f of a training trial, with the commands known from
f on (up to 8 ticks of the trial's tape) and, per tick h = 1..8, the true motion over h x 100 ms (from the physics
trace, in the body frame at f) and whether frame f+h exists. A window is unavailable before frame 3, past the
recording, or after any disallowed contact. Windows of a trial:
- geometry family: frames 3..42 (the 40 ticks of the candidate plan, from tick 3);
- switch family: frames 6..12 (before the switch at 13, with the prefix tape), and 13..52 (the suffix plan);
- short pulse: frames 8..17.

**Datasets.**
- `t2`: native one-step (500-ms) predictor samples: windows at frames >= 10 with at least 5 known commands and a
  future frame at +500 ms; contexts 1 s, 0.5 s and 0 s before, target at +0.5 s; the 5 commands after the safety
  limits; the normalised applied-command history at f.
- `t3`: the 48 balanced starts T3 adds to T2: frames 0, 5, 10 and 15 of each balanced-start case (the action starts
  at frame 10), with the recorded applied commands.
- `t4`: horizon samples: every window at frame >= 10 and horizon h = 1..8 with h known commands and a frame at
  +h x 100 ms (commands after h padded with zeros), then each balanced-start-horizon case at every horizon.
- `t5_original`: the T2 contexts as (current frame, frame 500 ms later, true body-frame motion) examples.
- `t5_full_heading`: the full-heading collection's frames 10..n-6 at +500 ms, unless a contact occurred by then.
- `t5_maze_view`: contexts with all 8 horizons: `old` = the T2 contexts whose 8 targets are valid and contact-free,
  then every full-heading frame 10..n-9; `maze` = the maze-view collection's frames 10..n-9 with no contact by +800 ms.
- `c1`: C1's contexts (every available window: the family and switch trials' plan windows, the pre-switch windows,
  the short-pulse windows) with the known commands, the 420-value applied-command history of the four frames up to
  the decision (per tick: command / (0.3, 1, 0.5), validity, age in s), the true motions, and the weight: how often
  the seeded 7,200-draw training schedule draws the context (`c1_schedule`).
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from go2nav.data.corridor import TAPES, candidate_plan
from go2nav.locomotion.safety import apply_limits
from go2nav.sim.sensors import rotation_xyzw

REPO = Path(__file__).resolve().parents[2]
COLLECTIONS = REPO/'data/collections'
CONTROL_STATS = REPO/'data/training/control_norm_stats.json'
FOLDERS = dict(family='go2_geometry_progress_family_v1_attempt_001',
               switch='go2_moving_action_switch_family_v1_attempt_001',
               short_pulse='go2_short_pulse_learning_v1_attempt_001',
               balanced_start='go2_balanced_start_actions_v1_attempt_001',
               balanced_start_horizon='go2_balanced_start_horizon_actions_v1_attempt_001',
               full_heading='go2_full_heading_training_v1_attempt_001',
               maze_view='go2_maze_view_training_v1_attempt_001')
PLANS = dict(family='geometry_progress_family', switch='switch_family', short_pulse='short_pulse',
             balanced_start='balanced_start', balanced_start_horizon='balanced_start_horizon')
FIRST_FRAME_SAMPLE, SAMPLES_PER_FRAME, HORIZON = 749, 50, 8


@dataclass
class Window:
    sample_id: str
    source: str
    trial: str
    frame: int
    commands: list
    targets: list | None  # 8 per-tick targets, or None if the window is unavailable


class Recordings:
    """Trial folders under one or more recordings roots."""

    def __init__(self, roots):
        self.roots = [Path(r) for r in roots]

    def find(self, name):
        """The path of `name` (relative to a recordings root) under the first root that has it."""
        for root in self.roots:
            if (root/name).exists():
                return root/name
        raise FileNotFoundError(name)

    def trial(self, source, trial):
        return self.find(f'{FOLDERS[source]}/{trial}')

    def trace(self, source, trial):
        with np.load(self.trial(source, trial)/'physics_trace.npz') as archive:
            return {k: archive[k].copy() for k in ('timestamp_s', 'base_pose_world', 'physics_contact', 'requested_command')}

    def frame_samples(self, source, trial):
        """The physics-sample index of each camera frame (frames are taken at a physics sample's time)."""
        return [int(t)//2_000_000-1 for t in self.histories(source, trial)['image_ns']]

    def histories(self, source, trial):
        with np.load(self.trial(source, trial)/'policy_histories.npz') as archive:
            return {k: archive[k].copy() for k in archive.files}


def plan_trials(source):
    """{trial: plan entry} of a collection's training trials (balanced starts: every case)."""
    plan = json.loads((COLLECTIONS/PLANS[source]/'plan.json').read_text())
    return {name: trial for name, trial in plan['trials'].items()
            if trial.get('data_role') == 'train' or source.startswith('balanced_start')}


def tape_commands(source, trial):
    return [command for command, _, _ in TAPES[PLANS[source]](trial)]


# -- windows ----------------------------------------------------------------------------------------------------------

def targets(trace, frame_count, frame, commands):
    """The window's per-tick targets (see the module docstring), or None if it is unavailable."""
    pose, contact, requested = trace['base_pose_world'], trace['physics_contact'], trace['requested_command']
    n = len(pose)
    commands = np.asarray(commands, float)
    start = FIRST_FRAME_SAMPLE+SAMPLES_PER_FRAME*frame
    if start >= n or frame >= frame_count or contact[:start+1].any():
        return None
    R, origin = rotation_xyzw(pose[start, 3:]), pose[start, :3]
    out = []
    for h in range(1, HORIZON+1):
        active = h <= len(commands)
        at = start+h*SAMPLES_PER_FRAME
        complete = active and at < n
        if active:
            a, b = start+(h-1)*SAMPLES_PER_FRAME+1, min(at+1, n)
            if a < b and not np.array_equal(requested[a:b], np.tile(commands[h-1], (b-a, 1))):
                raise ValueError(f'frame {frame}: the known commands differ from the recorded ones')
        event = bool(active and contact[start+1:min(at+1, n)].any())
        motion = None
        if complete and not event:
            delta = R.T@(pose[at, :3]-origin)
            relative = R.T@rotation_xyzw(pose[at, 3:])
            motion = [float(delta[0]), float(delta[1]), float(np.arctan2(relative[1, 0], relative[0, 0]))]
        future = bool(active and motion is not None and frame+h < frame_count)
        out.append(dict(offset_ns=h*100_000_000 if active else 0, in_plan=active, motion_valid=motion is not None,
                        motion=motion, contact_valid=complete or event,
                        contact=float(event) if complete or event else None, future_image_valid=future,
                        future_observation_index=frame+h if future else None))
    return out


def trial_windows(recordings, source, trial, entry):
    """Every window of one training trial, in frame order."""
    trace = recordings.trace(source, trial)
    frame_count = len(recordings.histories(source, trial)['image_ns'])
    if source == 'family':
        plan = candidate_plan(entry['action'])
        specs = [(f'all_phase_train/family/{trial}/offset_{k:02d}', 3+k, plan[k:k+HORIZON]) for k in range(40)]
    elif source == 'switch':
        tape, plan = tape_commands(source, entry), candidate_plan(entry['suffix_action'])
        specs = [(f'pre_switch_train/{trial}/frame_{f:02d}', f, tape[f:f+HORIZON]) for f in range(6, 13)]
        specs += [(f'all_phase_train/switch/{trial}/offset_{k:02d}', 13+k, plan[k:k+HORIZON]) for k in range(40)]
    else:
        tape = tape_commands(source, entry)
        specs = [(f'short_pulse/{trial}/frame_{f:02d}', f, tape[f:f+HORIZON]) for f in range(8, 18)]
    return [Window(sample_id, source, trial, frame, [list(map(float, c)) for c in commands],
                   targets(trace, frame_count, frame, commands)) for sample_id, frame, commands in specs]


def windows(recordings, sources=('family', 'short_pulse', 'switch')):
    """{(source, trial): windows} of every training trial."""
    return {(source, trial): trial_windows(recordings, source, trial, entry)
            for source in sources for trial, entry in sorted(plan_trials(source).items())}


# -- datasets ---------------------------------------------------------------------------------------------------------

def control_stats():
    stats = json.loads(CONTROL_STATS.read_text())
    return tuple(np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std'))


def frame_path(source, trial, index):
    return f'{FOLDERS[source]}/{trial}/rgb_{index:04d}.png'


def applied_commands(recordings, source, trial):
    return recordings.histories(source, trial)['applied_command_values'].astype(np.float32)


def normalised_control(history, frame):
    mean, std = control_stats()
    return ((history[frame][:, [0, 2]].reshape(3, 5, 2)-mean)/std).tolist()


def limited(commands, history, frame):
    """The commands after the safety limits, from the applied command at `frame`."""
    return apply_limits(np.asarray(commands, np.float32)[None], history[frame, -1][None])[0]


def t2(recordings):
    out = []
    for (source, trial), group in sorted(windows(recordings).items()):
        selected = [w for w in group if w.frame >= 10 and len(w.commands) >= 5 and w.targets
                    and w.targets[4]['future_image_valid']]
        if not selected:
            continue
        history = applied_commands(recordings, source, trial)
        for w in sorted(selected, key=lambda w: w.frame):
            f = w.frame
            out.append(dict(sample_id=w.sample_id, source=source, trial=trial, frame=f,
                            action=limited(w.commands[:5], history, f)[:, [0, 2]].reshape(10).tolist(),
                            control=normalised_control(history, f),
                            frames=[frame_path(source, trial, i) for i in (f-10, f-5, f, f+5)]))
    return out


def t3(recordings):
    out = []
    for case, entry in sorted(plan_trials('balanced_start').items()):
        history = applied_commands(recordings, 'balanced_start', case)
        out.append(dict(sample_id=f'balanced_start/{case}/frame_10', source='balanced_start',
                        trial=entry['geometry_trial'], frame=10,
                        action=history[15][-5:][:, [0, 2]].reshape(10).tolist(), control=normalised_control(history, 10),
                        frames=[frame_path('balanced_start', case, i) for i in (0, 5, 10, 15)]))
    return out


def t4(recordings):
    out = []

    def append(source, trial, folder, history, frame, h, commands):
        action = np.zeros((HORIZON, 2), np.float32)
        action[:h] = limited(commands, history, frame)[:, [0, 2]]
        out.append(dict(source=source, trial=trial, frame=frame, horizon=h, action=action.tolist(),
                        control=normalised_control(history, frame),
                        frames=[frame_path(source if folder is None else 'balanced_start_horizon', folder or trial, i)
                                for i in (frame-10, frame-5, frame, frame+h)]))
    for (source, trial), group in sorted(windows(recordings).items()):
        selected = sorted((w for w in group if w.frame >= 10 and w.targets), key=lambda w: w.frame)
        if not selected:
            continue
        history = applied_commands(recordings, source, trial)
        for w in selected:
            for h in range(1, HORIZON+1):
                if len(w.commands) >= h and w.targets[h-1]['future_image_valid']:
                    append(source, trial, None, history, w.frame, h, w.commands[:h])
    for case, entry in sorted(plan_trials('balanced_start_horizon').items()):
        history = applied_commands(recordings, 'balanced_start_horizon', case)
        for h in range(1, HORIZON+1):
            append('balanced_start', entry['geometry_trial'], case, history, 10, h, [candidate_plan(entry['action'])[0]]*h)
    return out


def body_motion(poses, start, end):
    """Forward, lateral and yaw motion from physics sample `start` to `end`, in the body frame at `start`."""
    R = rotation_xyzw(poses[start, 3:])
    relative = R.T@rotation_xyzw(poses[end, 3:])
    delta = (poses[end, :3]-poses[start, :3])@R
    return [float(delta[0]), float(delta[1]), float(np.arctan2(relative[1, 0], relative[0, 0]))]


def cases(source):
    plan = json.loads((COLLECTIONS/{'full_heading': 'full_heading', 'maze_view': 'maze_view_training'}[source]/'plan.json').read_text())
    return [f'case_{i:02d}' for i in range(len(plan['specifications']))]


def t5_original(recordings):
    by_id = {w.sample_id: w for group in windows(recordings).values() for w in group}
    return [dict(current=s['frames'][2], future=s['frames'][3],
                 motion=np.asarray(by_id[s['sample_id']].targets[4]['motion'], np.float32).tolist())
            for s in t2(recordings)]


def t5_full_heading(recordings):
    out = []
    for case in cases('full_heading'):
        samples, trace = recordings.frame_samples('full_heading', case), recordings.trace('full_heading', case)
        for frame in range(10, len(samples)-5):
            if not trace['physics_contact'][:samples[frame+5]+1].any():
                out.append(dict(current=frame_path('full_heading', case, frame),
                                future=frame_path('full_heading', case, frame+5),
                                motion=body_motion(trace['base_pose_world'], samples[frame], samples[frame+5])))
    return out


def t5_maze_view(recordings):
    def contexts(source, frames_of, motion_of):
        pairs, motions = [], []
        for case in cases(source):
            samples, trace = recordings.frame_samples(source, case), recordings.trace(source, case)
            for frame in frames_of(samples, trace['physics_contact']):
                pairs.append([[frame_path(source, case, frame), frame_path(source, case, frame+h)]
                              for h in range(1, HORIZON+1)])
                motions.append([motion_of(body_motion(trace['base_pose_world'], samples[frame], samples[frame+h]))
                                for h in range(1, HORIZON+1)])
        return pairs, motions
    by_id = {w.sample_id: w for group in windows(recordings).values() for w in group}
    pairs, motions = [], []
    for s in t2(recordings):
        targets = by_id[s['sample_id']].targets
        if all(t['motion_valid'] and t['future_image_valid'] and t['contact_valid'] and t['contact'] == 0 for t in targets):
            source, trial, f = s['source'], s['trial'], s['frame']
            pairs.append([[frame_path(source, trial, f), frame_path(source, trial, f+h)] for h in range(1, HORIZON+1)])
            motions.append([t['motion'] for t in targets])
    heading_pairs, heading_motions = contexts('full_heading', lambda samples, contact: range(10, len(samples)-8),
                                              lambda m: np.asarray(m, np.float32).tolist())
    maze_pairs, maze_motions = contexts('maze_view', lambda samples, contact: [
        f for f in range(10, len(samples)-8) if not contact[:samples[f+8]+1].any()], lambda m: m)
    return dict(old=dict(pairs=pairs+heading_pairs, motions=motions+heading_motions),
                maze=dict(pairs=maze_pairs, motions=maze_motions))


def c1_rows(recordings):
    """C1's available windows in the original order: plan windows (family, then switch trials), pre-switch, pulse."""
    groups = sorted(windows(recordings).items())
    picked = [lambda w: w.sample_id.startswith('all_phase'), lambda w: w.sample_id.startswith('pre_switch'),
              lambda w: w.source == 'short_pulse']
    return [w for pick in picked for _, group in groups for w in group if w.targets and pick(w)]


def c1_schedule(rows):
    """The draw count of every context in the seeded 1,200 x 6 training schedule C1 is weighted by, built in three
    steps: (1) 75 draws per family trial and 50 per switch trial, each a truncated sequence of shuffled cycles over the
    trial's plan windows, shuffled into source-homogeneous batches; (2) the switch batches redrawn the same way over the
    switch trials' plan and pre-switch windows; (3) 900 repeated draws replaced by 5 draws of each short-pulse window."""
    seed = 2026091001
    by_trial = {}
    for w in rows:
        by_trial.setdefault((w.source, w.trial), []).append(w)

    def pool(source, draws, generator, selected):
        out = []
        for (s, _), group in sorted(by_trial.items()):
            if s != source:
                continue
            ids = [w.sample_id for w in sorted(group, key=lambda w: w.frame) if selected(w)]
            drawn = []
            while len(drawn) < draws:
                drawn.extend(str(i) for i in generator.permutation(ids))
            out.extend(drawn[:draws])
        return [out[int(i)] for i in generator.permutation(len(out))]
    plan_only = lambda w: w.sample_id.startswith('all_phase')
    family = pool('family', 75, np.random.default_rng(seed), plan_only)  # step 1's switch batches are redrawn in step 2
    order = np.random.default_rng(seed+1_000_000_000)
    kinds = [kind for _ in range(600) for kind in np.array(['family', 'switch'])[order.permutation(2)]]
    switch = pool('switch', 50, np.random.default_rng(seed+500_000_000), lambda w: True)
    batches, f, s = [], 0, 0
    for kind in kinds:
        if kind == 'family':
            batches.append(family[f:f+6]); f += 6
        else:
            batches.append(switch[s:s+6]); s += 6
    counts = Counter(i for b in batches for i in b)
    generator = np.random.default_rng(2026091602)
    positions = [(i, j) for i, b in enumerate(batches) for j in range(len(b))]
    positions = [positions[i] for i in generator.permutation(len(positions))]
    replace = []
    for i, j in positions:
        if counts[batches[i][j]] > 1:
            replace.append((i, j))
            counts[batches[i][j]] -= 1
        if len(replace) == 900:
            break
    pulse = [w.sample_id for w in rows if w.source == 'short_pulse']
    for (i, j), identifier in zip(replace, [str(x) for x in generator.permutation(pulse*5)], strict=True):
        batches[i][j] = identifier
    return Counter(i for b in batches for i in b)


def c1(recordings):
    rows = c1_rows(recordings)
    weights = c1_schedule(rows)
    out, cache = [], {}
    for w in rows:
        if (w.source, w.trial) not in cache:
            cache[w.source, w.trial] = recordings.histories(w.source, w.trial)
        h = cache[w.source, w.trial]
        history = []
        for f in range(w.frame-3, w.frame+1):
            measured = h['applied_command_measured_ns'][f]
            age = np.where(measured >= 0, (h['decision_ns'][f]-measured)/1e9, 1.5)[:, None]
            history.append(np.concatenate((h['applied_command_values'][f]/[.3, 1., .5],
                                           h['applied_command_valid'][f].astype(float), age), axis=1).ravel())
        out.append(dict(sample_id=w.sample_id, commands=w.commands, history=np.concatenate(history).tolist(),
                        motions=[t['motion'] if t['motion_valid'] else None for t in w.targets],
                        weight=weights[w.sample_id]))
    return out


def decoder(recordings):
    from go2nav.data.decoder_contexts import contexts
    return contexts(recordings)


DATASETS = dict(t2=t2, t3=t3, t4=t4, t5_original=t5_original, t5_full_heading=t5_full_heading,
                t5_maze_view=t5_maze_view, c1=c1, decoder=decoder)


LAYOUT = {'native_predictor': dict(t2_samples='t2', t3_balanced_start_samples='t3', t4_samples='t4'),
          'readouts': dict(t5_original='t5_original', t5_full_heading='t5_full_heading', t5_maze_view='t5_maze_view'),
          'c1_command_model': dict(contexts='c1'),
          'decoder': dict(contexts='decoder')}
FRAME_LISTS = ('native_predictor', 'readouts', 'decoder')


def write(data, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        out.write_text(''.join(json.dumps(r)+'\n' for r in data))
    else:
        out.write_text(json.dumps(data)+'\n')


def frames_in(value):
    """Every frame name (a string ending in .png) inside a dataset."""
    if isinstance(value, str):
        return {value} if value.endswith('.png') else set()
    items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    return set().union(*(frames_in(v) for v in items))


def write_all(recordings, out):
    for folder, files in LAYOUT.items():
        frames = set()
        for name, dataset in files.items():
            data = DATASETS[dataset](recordings)
            write(data, out/folder/(name+('.jsonl' if isinstance(data, list) else '.json')))
            frames |= frames_in(data)
            print(json.dumps(dict(file=f'{folder}/{name}', rows=len(data))), flush=True)
        if folder in FRAME_LISTS:
            (out/folder/'frames.txt').write_text('\n'.join(sorted(frames))+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', choices=sorted(DATASETS)+['all'])
    p.add_argument('--recordings-root', action='append', required=True)
    p.add_argument('--out', type=Path, required=True,
                   help='a .jsonl file (one sample per line; .json for t5_maze_view), or the data/training folder for all')
    a = p.parse_args()
    recordings = Recordings(a.recordings_root)
    if a.dataset == 'all':
        write_all(recordings, a.out)
        return
    data = DATASETS[a.dataset](recordings)
    write(data, a.out)
    print(json.dumps(dict(dataset=a.dataset, rows=len(data))))


if __name__ == '__main__':
    main()
