"""The command-range collection: random command-lattice tapes on open floor, for refitting C1's command model across
the whole command range (docs/dynamic/PLAN.md, the continuous baseline).

    python -m go2nav.data.command_range plan [--trials 240]      # writes data/collections/command_range/plan.json
    python -m go2nav.data.recordings ROOT --collection command_range --policy CHECKPOINT
    python -m go2nav.data.command_range contexts ROOT --role train|eval --out FILE.jsonl

**Plan.** Each trial is a corridor scene (the geometry family's first specification) with its walls replaced by a
12-m square enclosure (`HALF_SIDE_M`), so the robot can move any way for the whole tape (at most 3.6 m in 12 s). The tape: 3 quiet ticks, then `SEGMENTS` segments of `SEGMENT_TICKS`
100-ms ticks (a plan's commitment), each one command of the collection's lattice (`AXES`, the planner's first lattice:
yaw to ±0.6 rad/s; hold with probability `HOLD_SHARE`, else uniform over the others), then 10 zero ticks. A fifth of the trials are held out (`data_role` eval).

**Contexts** (C1's format, `go2nav.training.command_model`): a window at every frame from 15 (1.5 s of history) to the
end of the tape, with the tape's next eight commands, the 420-value applied-command history and the true motions
(`go2nav.data.samples.targets`); weight 1.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from go2nav.planning.actions import command, lattice

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO/'data/collections/command_range/plan.json'
SEGMENTS, SEGMENT_TICKS, HOLD_SHARE, QUIET_TICKS, DRAIN_TICKS = 30, 4, .15, 3, 10
EVAL_SHARE, SEED, FIRST_FRAME = .2, 2026102401, 15
HALF_SIDE_M = 6.
AXES = dict(forward=(-.10, 0., .15, .30, .45), lateral=(-.15, 0., .15), yaw=(-.6, -.3, 0., .3, .6))


def enclosure():
    """Four walls round the spawn, `HALF_SIDE_M` away."""
    side, thick, height = 2*HALF_SIDE_M, .08, .82
    return [dict(wall_id=f'enclosure_{k}', centre_xyz=[x, y, height/2], size_xyz=[sx, sy, height], yaw_rad=0.,
                 material_id='NEUTRAL_WALL')
            for k, (x, y, sx, sy) in enumerate(((HALF_SIDE_M, 0., thick, side), (-HALF_SIDE_M, 0., thick, side),
                                                (0., HALF_SIDE_M, side, thick), (0., -HALF_SIDE_M, side, thick)))]


def make_plan(trials):
    source = json.loads((REPO/'data/collections/geometry_progress_family/plan.json').read_text())
    base = next(iter(source['trials'].values()))['specification']
    rng = np.random.default_rng(SEED)
    moving = [a for a in lattice(**AXES) if a != 'hold']
    plan = dict(name='command_range', description=__doc__.split('\n\n')[0], trials={})
    for k in range(trials):
        name = f'range_episode_{k:03d}'
        spec = json.loads(json.dumps(base))
        spec['geometry']['wall_boxes'] = enclosure()
        spec.update(scene_id=f'command-range-v1-{name}', trial=name, data_role='command_range')
        segments = ['hold' if rng.random() < HOLD_SHARE else moving[rng.integers(len(moving))] for _ in range(SEGMENTS)]
        tape = [[0., 0., 0.]]*QUIET_TICKS+[list(command(a)) for a in segments for _ in range(SEGMENT_TICKS)]
        tape += [[0., 0., 0.]]*DRAIN_TICKS
        role = 'eval' if rng.random() < EVAL_SHARE else 'train'
        plan['trials'][name] = dict(specification=spec, data_role=role, segments=segments, tape=tape)
    return plan


def contexts(root, role):
    from go2nav.data.samples import FOLDERS, Recordings, targets
    recordings = Recordings([root])
    plan = json.loads(PLAN.read_text())
    out = []
    for name, trial in sorted(plan['trials'].items()):
        if trial['data_role'] != role:
            continue
        folder = recordings.find(f"{FOLDERS['command_range']}/{name}")
        if not (folder/'result.json').exists():
            continue
        trace = recordings.trace('command_range', name)
        h = recordings.histories('command_range', name)
        frames = len(h['image_ns'])
        for f in range(FIRST_FRAME, len(trial['tape'])):
            commands = trial['tape'][f:f+8]
            window = targets(trace, frames, f, commands)
            if window is None:
                continue
            history = []
            for g in range(f-3, f+1):
                measured = h['applied_command_measured_ns'][g]
                age = np.where(measured >= 0, (h['decision_ns'][g]-measured)/1e9, 1.5)[:, None]
                history.append(np.concatenate((h['applied_command_values'][g]/[.3, 1., .5],
                                               h['applied_command_valid'][g].astype(float), age), axis=1).ravel())
            out.append(dict(sample_id=f'command_range/{name}/frame_{f:03d}', commands=commands,
                            history=np.concatenate(history).tolist(),
                            motions=[t['motion'] if t['motion_valid'] else None for t in window], weight=1.))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--trials', type=int, default=240)
    b = sub.add_parser('contexts')
    b.add_argument('root', type=Path)
    b.add_argument('--role', choices=('train', 'eval'), required=True)
    b.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.step == 'plan':
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(make_plan(args.trials)))
        print(f'{args.trials} trials -> {PLAN.relative_to(REPO)}')
    else:
        rows = contexts(args.root, args.role)
        args.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(f'{len(rows)} contexts -> {args.out}')


if __name__ == '__main__':
    main()
