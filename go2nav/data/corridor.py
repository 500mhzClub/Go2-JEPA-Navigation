"""The older corridor collections (geometry-progress family scenes): scripted command tapes, robot not rendered,
primary camera only. Their frames trained the native predictor (T2-T4).

    python -m go2nav.data.corridor COLLECTION TRIAL OUT_DIR

A collection is data/collections/<name>/plan.json: per trial, the scene specification and what the tape is built from.
Each trial settles (no renderer warm-up), then at every 100-ms tick captures a frame (primary RGB and native depth)
and executes the tick's command for five policy steps; a final frame follows the last tick. Physical stops end a trial
early and are kept. Tapes:
- geometry_progress_family: 3 quiet ticks, then the action's 40-tick candidate plan (30 ticks of motion then 10 of
  zero; arcs straighten for their last 10 motion ticks);
- switch_family: 3 quiet ticks, 10 ticks of the prefix action, the suffix action's candidate plan, 10 zero ticks;
- short_pulse: 3 quiet ticks, 10 ticks of the preceding action, 3 zero ticks, one tick of the pulse action, 8 zero;
- balanced_start (balanced_start_horizon): 10 zero ticks, then 5 (8) ticks of the action. These also capture one
  frame at the moment of a physical stop.

A trial folder holds rgb_NNNN.png and native_depth_NNNN.npz per frame, frames.json, command_tape.json,
physics_trace.npz, policy_histories.npz, contact_events.json and result.json.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from go2nav import artifacts
from go2nav.planning.actions import ACTIONS

REPO = Path(__file__).resolve().parents[2]
COLLECTIONS = REPO/'data/collections'
QUIET_TICKS, PREFIX_TICKS, DRAIN_TICKS = 3, 10, 10


def candidate_plan(action):
    """The 40-tick candidate plan of an action (the collections' fixed six-action bank)."""
    if action not in ACTIONS:
        raise ValueError(f'unknown action {action}')
    zero = [0., 0., 0.]
    if action == 'hold':
        return [zero]*40
    if action == 'forward':
        return [[.20, 0., 0.]]*30+[zero]*10
    if action in ('left_arc', 'right_arc'):
        yaw = .45 if action == 'left_arc' else -.45
        return [[.16, 0., yaw]]*20+[[.16, 0., 0.]]*10+[zero]*10
    yaw = .45 if action == 'left_turn' else -.45
    return [[0., 0., yaw]]*30+[zero]*10


def switch_tape(trial):
    """[(requested command, phase, role)] of a switch-family trial."""
    rows = [([0., 0., 0.], 1, 'common_quiet_history')]*QUIET_TICKS
    rows += [(candidate_plan(trial['prefix_action'])[0], 2, 'moving_prefix')]*PREFIX_TICKS
    rows += [(c, 2 if i < 30 else 3, 'candidate_suffix' if i < 30 else 'candidate_brake')
             for i, c in enumerate(candidate_plan(trial['suffix_action']))]
    rows += [([0., 0., 0.], 3, 'terminal_zero_drain')]*DRAIN_TICKS
    return rows


def family_tape(trial):
    return [([0., 0., 0.], 1, 'common_quiet_history')]*QUIET_TICKS+[
        (c, 2 if i < 30 else 3, 'candidate_prefix' if i < 30 else 'candidate_brake')
        for i, c in enumerate(candidate_plan(trial['action']))]


def short_pulse_tape(trial):
    blocks = [(3, 'quiet', None), (10, 'preceding_action', trial['prefix_action']), (3, 'committed_zero', None),
              (1, 'translation_pulse', trial['pulse_action']), (8, 'zero_drain', None)]
    return [(candidate_plan(action)[0] if action else [0., 0., 0.], 2 if action else 3, role)
            for count, role, action in blocks for _ in range(count)]


def balanced_tape(ticks):
    def tape(trial):
        return [([0., 0., 0.], 1, 'quiet')]*10+[(candidate_plan(trial['action'])[0], 2, 'action')]*ticks
    return tape


TAPES = dict(geometry_progress_family=family_tape, switch_family=switch_tape, short_pulse=short_pulse_tape,
             balanced_start=balanced_tape(5), balanced_start_horizon=balanced_tape(8))
CAPTURE_AT_STOP = ('balanced_start', 'balanced_start_horizon')


def collect(name, trial_name, out):
    import cv2
    import torch
    from go2nav.eval.mission import persist, save
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession, PhysicalStop
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    plan = json.loads((COLLECTIONS/name/'plan.json').read_text())
    trial = plan['trials'][trial_name]
    spec, tape = trial['specification'], TAPES[name](trial)
    out = Path(out)
    out.mkdir(parents=True)
    save(out/'specification.json', spec)
    started, stop, frames, executed = time.monotonic(), None, [], []
    scene.init_genesis(spec['procedural_seed'])
    session = MissionSession(spec, out/'visual_meshes', policy_checkpoint=artifacts.path('go2_locomotion_policy'),
                             noise_sigma_m=0., noise_layout=0, robot_visible=False)

    def capture(aligned=True):
        index, rgb, depth, _ = session.capture(aligned=aligned)
        Image.fromarray(rgb).save(out/f'rgb_{index:04d}.png')
        np.savez_compressed(out/f'native_depth_{index:04d}.npz', optical_depth_m=depth)
        frames.append(dict(frame=index, measured_ns=session.sim_ns, physical_sample_index=len(session.samples)-1))
    try:
        session.settle(warm_renderer=False)
        for tick, (command, phase, role) in enumerate(tape):
            capture()
            session.phase = phase
            item = dict(tick=tick, requested_command=command, phase=phase, role=role, pre_sample_index=len(session.samples)-1)
            executed.append(item)
            session.command_tick(command)
            item['post_sample_index'] = len(session.samples)-1
        capture()
    except PhysicalStop as error:
        stop = str(error)
        if name in CAPTURE_AT_STOP:
            capture(aligned=False)
    finally:
        persist(session, out)
        save(out/'frames.json', frames)
        save(out/'command_tape.json', executed)
    result = dict(collection=name, trial=trial_name, physical_stop=stop, command_ticks=len(executed), frames=len(frames),
                  physics_samples=len(session.samples), wall_s=time.monotonic()-started)
    save(out/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('collection', choices=sorted(TAPES))
    p.add_argument('trial')
    p.add_argument('out', type=Path)
    a = p.parse_args()
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('EGL_DEVICE_ID', '0')
    print(json.dumps(collect(a.collection, a.trial, a.out)))


if __name__ == '__main__':
    main()
