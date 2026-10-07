"""Run one navigation mission: an arm, a registered maze episode, the simulator and the controller in closed loop.

    python -m go2nav.eval.mission --arm C1 --set prelim_test_v1 --maze 30 --out runs/C1_prelim30 [--recovery on]

The loop is the original harness's: 50 Hz policy steps, at most 24,000 (480 s simulated). Every fifth step the session
renders a sensor frame and the controller processes it (perception, mission, planning, all synchronous); at every
step the controller's request is executed. The mission ends at the mission's terminal state, at the step budget, or
when the robot stops physically (contact, fall or guard; `PhysicalStop`), which is kept as a failed attempt.

Settings that the original fixed for reproducibility are fixed here too: Genesis on CPU seeded with the episode's
simulation seed, 4 torch threads, single-threaded OpenCV.

C0 (the simulator oracle) also writes oracle_prefix_check.json; a failed check is recorded as the run's error.

The run folder: config.json, specification.json, episode.json, public_task_cues.json, requests.json,
acquisitions.json, planning.json, mission.json, poses.json, stage_timings.json, progress.jsonl, result.json (and
failure.json); native/
holds the physics trace, the contact records and the visual meshes.
"""
import argparse
from collections import deque
import dataclasses
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from go2nav import artifacts
from go2nav.mazes.sets import MazeSet

STEP_BUDGET = 24_000
FRAME_STEPS = 5


def plain(value):
    """A JSON-ready copy (numpy values, tuples, sets and dataclasses converted)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), allow_nan=False)+'\n')


def episode_spec(set_name, maze, episode_index):
    """The maze spec for one registered episode (its simulation seed and home pose) and the episode record."""
    maze_set = MazeSet(set_name)
    spec, episode = maze_set.maze(maze), maze_set.episode(maze, episode_index)
    spec = spec | dict(procedural_seed=episode['simulation_seed'],
                       geometry=spec['geometry'] | dict(spawn_se2_world=episode['home_se2_world']))
    return spec, episode


def mission_dynamics(set_name, maze, episode_index, spec, episode, *, friction=None, strips=None, stall_stop_s=None):
    """The `Dynamics` of a mission and its record. `strips` is (mu, marked); strip placement is seeded by the set's
    original name (the role, or `prelim_test`), the maze and the episode."""
    from go2nav.sim.dynamics import Dynamics, place_strips, placement_seed
    if friction is None and strips is None and not stall_stop_s:
        return None, None
    dynamics, record = Dynamics(friction_mu=friction, stall_stop_s=stall_stop_s), {}
    if friction is not None:
        record.update(perturbation='uniform_floor_friction', mu=friction)
    if strips is not None:
        mu, marked = strips
        seed = placement_seed('prelim_test' if set_name == 'prelim_test_v1' else episode['role'], maze, episode_index)
        placement = place_strips(spec, episode, seed)
        dynamics.strips, dynamics.strip_mu, dynamics.marked = [s['rect'] for s in placement['strips']], mu, marked
        record.update(perturbation='low_friction_strips', mu=mu, marked=marked, placement=placement)
    if stall_stop_s:
        record.update(recording_stall_stop_s=stall_stop_s)
    return dynamics, record


def run(arm, set_name, maze, out, *, episode_index=0, recovery=False, decoder_fit=None, friction=None, strips=None,
        stall_stop_s=None, degrade=None, margin=None, pessimistic=None, unknown_level=None):
    """Run one mission into the folder `out` (created; must not exist). Returns the result record. Dynamics
    perturbations: `friction` (uniform mu), `strips` ((mu, marked)), `stall_stop_s` (recording missions). Experiment
    hooks: `degrade` (a forecast degradation, `controllers.degraded`), `margin` (p95/p99 calibrated clearance margin),
    `pessimistic` (never-observed cells block: revised, seeded or lookaround; bound at `unknown_level`, else the
    margin level, else p99; `planning.caution`)."""
    import cv2
    import torch
    from go2nav.controllers import arms
    from go2nav.controllers.controller import NavigationController, SensorFrame
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession, settled_task_cues
    out = Path(out)
    out.mkdir(parents=True)
    spec, episode = episode_spec(set_name, maze, episode_index)
    dynamics, dynamics_record = mission_dynamics(set_name, maze, episode_index, spec, episode, friction=friction,
                                                 strips=strips, stall_stop_s=stall_stop_s)
    from go2nav.planning import caution
    margin_m = caution.calibrated_bound(arm, margin) if margin else 0.
    unknown_bound = caution.calibrated_bound(arm, unknown_level or margin or 'p99') if pessimistic else None
    save(out/'config.json', dict(arm=arm, set=set_name, maze=maze, episode=episode_index, recovery=recovery,
                                 decoder_fit=str(decoder_fit) if decoder_fit else None, step_budget=STEP_BUDGET,
                                 physics_backend='cpu', dynamics=dynamics_record, forecast_degradation=degrade,
                                 clearance_margin=dict(level=margin, margin_m=margin_m) if margin else None,
                                 pessimistic_unknown=dict(variant=pessimistic, bound_m=unknown_bound) if pessimistic else None))
    save(out/'specification.json', spec)
    save(out/'episode.json', episode)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    requests, acquisitions, timings = [], [], []
    controller = session = error = None
    started = time.monotonic()
    try:
        forecaster = None if arm == 'C0' else arms.forecaster(arm, decoder_fit=decoder_fit)
        scene.init_genesis(spec['procedural_seed'])
        session = MissionSession(spec, out/'native', policy_checkpoint=artifacts.path('go2_locomotion_policy'),
                                 dynamics=dynamics)
        cues = settled_task_cues(episode, session.settle())
        if arm == 'C0':
            from go2nav.controllers.oracle import OracleForecaster
            forecaster = OracleForecaster(session)
        if degrade:
            from go2nav.controllers.degraded import DegradedForecaster
            forecaster = DegradedForecaster(forecaster, degrade)
        save(out/'public_task_cues.json', cues)
        controller = NavigationController(forecaster, goal_xy=cues['goal_initial_body_xy_m'],
                                          home_xy=cues['return_initial_body_xy_m'], arrival_radius_m=.02,
                                          navigation_frames=4800, recovery=recovery, margin_m=margin_m,
                                          unknown=caution.UnknownCells(unknown_bound, pessimistic) if pessimistic else None)
        history = deque(maxlen=4)
        for tick in range(STEP_BUDGET):
            now = session.sim_ns
            if tick % FRAME_STEPS == 0:
                policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.frame()
                history.append(policy)
                planned = len(controller.planning)
                begun = time.perf_counter_ns()
                controller.on_frame(SensorFrame(tick//FRAME_STEPS, measured, policy['image']['rgb'],
                                                (depth['depth_m'], depth['valid']), auxiliary_rgb['rgb'],
                                                (auxiliary_depth['depth_m'], auxiliary_depth['valid']), fast['values'],
                                                policy['sensor_state']['sensed']['specific_force']['values'], policy,
                                                tuple(history)))
                if len(controller.planning) > planned and 'action' in controller.planning[-1]:
                    timings.append(dict(stage='frame_with_planning', frame=tick//FRAME_STEPS,
                                        wall_ns=time.perf_counter_ns()-begun))
                acquisitions.append(dict(frame=tick//FRAME_STEPS, measured_ns=measured))
            request = controller.request(now)
            row = dict(request, simulator_ns=now, pre_sample_index=len(session.samples)-1)
            requests.append(row)
            session.phase = 2
            row['applied_command'] = session.step(request['requested_command'])
            row['post_sample_index'] = len(session.samples)-1
            if tick % 500 == 0:
                with open(out/'progress.jsonl', 'a') as progress:
                    progress.write(json.dumps(dict(policy_steps=tick+1, simulated_s=(tick+1)*.02,
                                                   wall_s=time.monotonic()-started, plans=len(controller.planning)))+'\n')
            if controller.terminal is not None:
                break
    except BaseException as exc:
        error = exc
        save(out/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), policy_steps=len(requests),
                                      frames=len(acquisitions)))
    finally:
        if controller is not None:
            save(out/'planning.json', controller.planning)
            save(out/'mission.json', controller.mission_rows)
            save(out/'poses.json', [dict(frame=p['frame'], visual_pose=p['visual'], registered_pose=p['registered'])
                                    for p in controller.poses])
        save(out/'requests.json', requests)
        save(out/'acquisitions.json', acquisitions)
        save(out/'stage_timings.json', timings)
        if session is not None:
            persist(session, out/'native')
        if arm == 'C0' and requests and controller is not None:
            check = controller.forecaster.verify_executed(requests, session.samples)
            save(out/'oracle_prefix_check.json', check)
            if not check['passed'] and error is None:
                error = RuntimeError('oracle executed-prefix fidelity check failed')
        result = dict(status='EPISODE_RECORDED' if error is None else 'FAILED_ATTEMPT_PRESERVED',
                      error=None if error is None else repr(error), frames=len(acquisitions), policy_steps=len(requests),
                      simulated_s=len(requests)*.02, wall_s=time.monotonic()-started)
        save(out/'result.json', result)
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error
    return result


def persist(session, folder):
    """The evaluator's records: the physics trace (one row per physics step) and the native contacts."""
    from go2nav.sim.contacts import FIELDS
    folder.mkdir(parents=True, exist_ok=True)
    if session.samples:
        np.savez_compressed(folder/'physics_trace.npz',
                            **{key: np.stack([row[key] for row in session.samples]) for key in session.samples[0]})
    if session.contact_packets:
        sizes = [packet['link_a'].shape[1] for packet in session.contact_packets]
        native = {key: np.concatenate([packet[key][0] for packet in session.contact_packets], axis=0) for key in FIELDS}
        np.savez_compressed(folder/'native_contacts.npz', **native, frame_offsets=np.cumsum([0, *sizes], dtype=np.int64),
                            frame_timestamp_s=np.asarray(session.contact_times))
    save(folder/'contact_events.json', session.contact_events)
    save(folder/'contact_topology.json', dict(link_names=session.link_names, environment_object_ids=session.object_ids,
                                              robot_link_ids=sorted(session.world.links['robot']),
                                              support_link_ids=sorted(session.world.links['support']),
                                              ground_link_ids=sorted(session.world.links['ground'])))


def main():
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # headless rendering (read when Genesis is first imported)
    os.environ.setdefault('EGL_DEVICE_ID', '0')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, choices=('C0', 'C1', 'C1A', 'C1R', 'C2', 'C3', 'C4'))
    p.add_argument('--set', required=True, help='registered maze set (data/mazes), e.g. prelim_test_v1')
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--recovery', choices=('on', 'off'), default='on')
    p.add_argument('--decoder-fit', type=Path, help='C3-decoder/C4 fit file (default: the deployed T7 fit)')
    p.add_argument('--friction', type=float, help='uniform floor and robot friction (stage 1)')
    p.add_argument('--strips', help='low-friction strips MU:marked or MU:unmarked (stage 2)')
    p.add_argument('--stall-stop-s', type=float, help='stop a recording mission stalled this long (60)')
    p.add_argument('--degrade', help='forecast degradation: scale:S, fwdscale:S, turnscale:S or noise:E_mm')
    p.add_argument('--margin', choices=('p95', 'p99'), help='calibrated clearance margin')
    p.add_argument('--pessimistic', choices=('revised', 'seeded', 'lookaround'), help='never-observed cells block')
    p.add_argument('--unknown-level', choices=('p95', 'p99'), help='bound for never-observed cells (default: the margin level, else p99)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    strips = None
    if a.strips:
        mu, _, mark = a.strips.partition(':')
        if mark not in ('marked', 'unmarked'):
            p.error('--strips needs MU:marked or MU:unmarked')
        strips = (float(mu), mark == 'marked')
    result = run(a.arm, a.set, a.maze, a.out, episode_index=a.episode, recovery=a.recovery == 'on', decoder_fit=a.decoder_fit,
                 friction=a.friction, strips=strips, stall_stop_s=a.stall_stop_s, degrade=a.degrade, margin=a.margin,
                 pessimistic=a.pessimistic, unknown_level=a.unknown_level)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
