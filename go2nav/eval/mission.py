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
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from go2nav import artifacts
from go2nav.controllers.registry import ARMS
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_spec(set_name, maze, episode_index, root=None):
    """The maze spec for one registered episode (its simulation seed and home pose) and the episode record. `root`:
    the folder holding the set (default: the shipped sets, data/mazes)."""
    maze_set = MazeSet(set_name) if root is None else MazeSet(set_name, root=root)
    spec, episode = maze_set.maze(maze), maze_set.episode(maze, episode_index)
    spec = spec | dict(procedural_seed=episode['simulation_seed'],
                       geometry=spec['geometry'] | dict(spawn_se2_world=episode['home_se2_world']))
    return spec, episode


def mission_dynamics(set_name, maze, episode_index, spec, episode, *, friction=None, strips=None, stall_stop_s=None,
                     obstacles=None, obstacle_series='v1', obstacle_speed=None):
    """The `Dynamics` of a mission and its record. `strips` is (mu, marked); strip placement is seeded by the set's
    original name (the role, or `prelim_test`), the maze and the episode. `obstacles`: how many moving obstacles
    (`go2nav.sim.obstacles`), placed with a seed of the set, maze, episode and count in `obstacle_series` (v1: the
    development series), at speeds drawn from `obstacle_speed` (low, high; default `SPEED_M_S`)."""
    from go2nav.sim.dynamics import Dynamics, place_strips, placement_seed
    from go2nav.sim import obstacles as moving
    if friction is None and strips is None and not stall_stop_s and not obstacles:
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
    if obstacles:
        seed = moving.placement_seed(set_name, maze, episode_index, obstacles, obstacle_series)
        dynamics.obstacles = moving.place_obstacles(spec, episode, obstacles, seed, obstacle_speed or moving.SPEED_M_S)
        record.update(moving_obstacles=dict(count=obstacles, placement=dynamics.obstacles,
                                            **(dict(series=obstacle_series) if obstacle_series != 'v1' else {}),
                                            **(dict(speed_m_s=list(obstacle_speed)) if obstacle_speed else {})))
    return dynamics, record


def run(arm, set_name, maze, out, *, episode_index=0, recovery=False, decoder_fit=None, friction=None, strips=None,
        stall_stop_s=None, obstacles=None, degrade=None, margin=None, pessimistic=None, unknown_level=None, demo_textures=False,
        models=None, harness='reserve_exit_v2', coverage_fix=True, map_decay=False, deadlock_escape=False,
        terminal_burst=False, obstacle_prediction=False, footprint_clearing=False, obstacle_series='v1', obstacle_speed=None,
        candidates='primitives', safety='primitive', pose_oracle=False, map_oracle=None, prediction_oracle=False,
        mover_trails=False):
    """Run one mission into the folder `out` (created; must not exist). Returns the result record. Dynamics
    perturbations: `friction` (uniform mu), `strips` ((mu, marked)), `stall_stop_s` (recording missions), `obstacles`
    (how many moving obstacles; `obstacle_series` their placement series, `obstacle_speed` their speed range). Experiment
    hooks: `degrade` (a forecast degradation, `controllers.degraded`), `margin` (p95/p99 calibrated clearance margin),
    `pessimistic` (never-observed cells block: revised, seeded or lookaround; bound at `unknown_level`, else the
    margin level, else p99; `planning.caution`). `demo_textures`: demonstration surfaces (`go2nav.sim.demo_appearance`;
    presentation only, never seen in training). `models`: artefact name -> checkpoint, replacing shipped models (a
    retrained chain, or the locomotion policy `go2_locomotion_policy`); recorded with each file's SHA-256. `harness`: 'reserve_exit_v2' (the final harness) or 'v4' (the
    earlier one, for the experiments recorded under it; `go2nav.planning.planner`); `coverage_fix=False`: the
    translation coverage rule as before 2 October; `map_decay`: occupied cells seen free again are cleared
    (`go2nav.perception.mapping`, for moving obstacles), and with no route left stale cells are forgotten ('floor':
    cells once seen as floor first); `deadlock_escape`: the latch timeout and deadlock escape alone; `terminal_burst`:
    the terminal burst alone (`go2nav.planning.planner`); `obstacle_prediction`: constant-velocity tracks of moving
    obstacles in the forecast clearance check (`go2nav.perception.tracks`, `go2nav.planning.prediction`);
    `footprint_clearing`: map cells inside the body's outline are dropped (`go2nav.perception.mapping`);
    `mover_trails`: a confirmed obstacle track's cells stay in the map only while seen occupied
    (`go2nav.perception.mapping`); `candidates`: 'primitives' (the six actions) or 'lattice' (the command lattice, C1
    only; `go2nav.planning.actions`); `safety`: the command safety block, 'primitive' or 'continuous'
    (`go2nav.locomotion.safety`). Module oracles (the failure analysis; `go2nav.controllers.truth`): `pose_oracle` (the
    true pose), `map_oracle` ('obstacles': perceived walls with the moving obstacles where they truly are; 'full': the
    true map; 'static': the true map, obstacles kept out of the route grid), `prediction_oracle` (true obstacle futures
    in the clearance check)."""
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
                                                 strips=strips, stall_stop_s=stall_stop_s, obstacles=obstacles,
                                                 obstacle_series=obstacle_series, obstacle_speed=obstacle_speed)
    from go2nav.planning import actions, caution
    margin_m = caution.calibrated_bound(arm, margin) if margin else 0.
    unknown_bound = caution.calibrated_bound(arm, unknown_level or margin or 'p99') if pessimistic else None
    save(out/'config.json', dict(arm=arm, set=set_name, maze=maze, episode=episode_index, recovery=recovery,
                                 decoder_fit=str(decoder_fit) if decoder_fit else None, step_budget=STEP_BUDGET,
                                 physics_backend='cpu', dynamics=dynamics_record, forecast_degradation=degrade,
                                 clearance_margin=dict(level=margin, margin_m=margin_m) if margin else None,
                                 pessimistic_unknown=dict(variant=pessimistic, bound_m=unknown_bound) if pessimistic else None,
                                 **(dict(demo_textures=True) if demo_textures else {}),
                                 **(dict(models={name: dict(path=str(path), sha256=sha256(path)) for name, path in models.items()})
                                    if models else {}),
                                 **(dict(harness=harness) if harness != 'reserve_exit_v2' else {}),
                                 **(dict(coverage_fix=False) if not coverage_fix else {}),
                                 **(dict(map_decay=True if map_decay is True else map_decay) if map_decay else {}),
                                 **(dict(deadlock_escape=True) if deadlock_escape else {}),
                                 **(dict(terminal_burst=True) if terminal_burst else {}),
                                 **(dict(obstacle_prediction=True) if obstacle_prediction else {}),
                                 **(dict(footprint_clearing=True) if footprint_clearing else {}),
                                 **(dict(mover_trails=True) if mover_trails else {}),
                                 **(dict(candidates=candidates, lattice_axes=actions.LATTICE_AXES)
                                    if candidates == 'lattice' else {}),
                                 **(dict(safety=safety) if safety != 'primitive' else {}),
                                 **(dict(oracles=dict(pose=pose_oracle, map=map_oracle, prediction=prediction_oracle))
                                    if pose_oracle or map_oracle or prediction_oracle else {})))
    save(out/'specification.json', spec)
    save(out/'episode.json', episode)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    requests, acquisitions, timings = [], [], []
    controller = session = error = None
    started = time.monotonic()
    try:
        forecaster = None if arm == 'C0' else arms.forecaster(arm, decoder_fit=decoder_fit, models=models)
        scene.init_genesis(spec['procedural_seed'])
        policy = Path(models['go2_locomotion_policy']) if models and 'go2_locomotion_policy' in models \
            else artifacts.path('go2_locomotion_policy')
        session = MissionSession(spec, out/'native', policy_checkpoint=policy,
                                 dynamics=dynamics, demo_textures=demo_textures, safety=safety)
        cues = settled_task_cues(episode, session.settle())
        if arm == 'C0':
            from go2nav.controllers.oracle import OracleForecaster
            forecaster = OracleForecaster(session)
        if degrade:
            from go2nav.controllers.degraded import DegradedForecaster
            forecaster = DegradedForecaster(forecaster, degrade)
        save(out/'public_task_cues.json', cues)
        truth = None
        if pose_oracle or map_oracle or prediction_oracle:
            from go2nav.controllers.truth import GroundTruth
            truth = GroundTruth(session)
        controller = NavigationController(forecaster, goal_xy=cues['goal_initial_body_xy_m'],
                                          home_xy=cues['return_initial_body_xy_m'], arrival_radius_m=.02,
                                          navigation_frames=4800, recovery=recovery, margin_m=margin_m,
                                          unknown=caution.UnknownCells(unknown_bound, pessimistic) if pessimistic else None,
                                          harness=harness, coverage_fix=coverage_fix, map_decay=map_decay,
                                          deadlock_escape=deadlock_escape, terminal_burst=terminal_burst,
                                          obstacle_prediction=obstacle_prediction, footprint_clearing=footprint_clearing,
                                          candidates=actions.LATTICE if candidates == 'lattice' else actions.ACTIONS,
                                          truth=truth, pose_oracle=pose_oracle, map_oracle=map_oracle,
                                          prediction_oracle=prediction_oracle, mover_trails=mover_trails)
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
    if session.packet_rows:
        np.savez_compressed(folder/'policy_histories.npz', **{key: np.stack([row[key] for row in session.packet_rows])
                                                              for key in session.packet_rows[0]})
    if getattr(session, 'obstacles', None) is not None:
        np.savez_compressed(folder/'obstacle_trace.npz', **session.obstacles.arrays())
    save(folder/'contact_topology.json', dict(link_names=session.link_names, environment_object_ids=session.object_ids,
                                              robot_link_ids=sorted(session.world.links['robot']),
                                              support_link_ids=sorted(session.world.links['support']),
                                              ground_link_ids=sorted(session.world.links['ground'])))


def main():
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # headless rendering (read when Genesis is first imported)
    os.environ.setdefault('EGL_DEVICE_ID', '0')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--set', required=True, help='registered maze set (data/mazes), e.g. prelim_test_v1')
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--recovery', choices=('on', 'off'), default='on')
    p.add_argument('--decoder-fit', type=Path, help='C3-decoder/C4 fit file (default: the deployed T7 fit)')
    p.add_argument('--friction', type=float, help='uniform floor and robot friction (stage 1)')
    p.add_argument('--strips', help='low-friction strips MU:marked or MU:unmarked (stage 2)')
    p.add_argument('--stall-stop-s', type=float, help='stop a recording mission stalled this long (60)')
    p.add_argument('--obstacles', type=int, help='moving obstacles (go2nav.sim.obstacles)')
    p.add_argument('--obstacle-series', default='v1', help='obstacle placement series (v1: development)')
    p.add_argument('--obstacle-speed', help='obstacle speed range LOW:HIGH in m/s (default 0.15:0.35)')
    p.add_argument('--degrade', help='forecast degradation: scale:S, fwdscale:S, turnscale:S or noise:E_mm')
    p.add_argument('--margin', choices=('p95', 'p99'), help='calibrated clearance margin')
    p.add_argument('--pessimistic', choices=('revised', 'seeded', 'lookaround'), help='never-observed cells block')
    p.add_argument('--unknown-level', choices=('p95', 'p99'), help='bound for never-observed cells (default: the margin level, else p99)')
    p.add_argument('--demo-textures', action='store_true', help='brick and parquet demonstration surfaces (presentation only)')
    p.add_argument('--models', type=Path, help='YAML mapping artefact names to checkpoints that replace the shipped ones')
    p.add_argument('--harness', choices=('reserve_exit_v2', 'v4'), default='reserve_exit_v2')
    p.add_argument('--coverage-fix', choices=('on', 'off'), default='on')
    p.add_argument('--map-decay', choices=('on', 'floor', 'off'), default='off',
                   help='clear occupied cells seen free again; forget stale cells with no route (floor: seen-floor cells first)')
    p.add_argument('--deadlock-escape', choices=('on', 'off'), default='off', help='latch timeout and deadlock escape only')
    p.add_argument('--terminal-burst', choices=('on', 'off'), default='off', help='terminal burst only')
    p.add_argument('--obstacle-prediction', choices=('on', 'off'), default='off',
                   help='constant-velocity obstacle tracks in the clearance check')
    p.add_argument('--safety', choices=('primitive', 'continuous'), default='primitive', help='command safety block')
    p.add_argument('--candidates', choices=('primitives', 'lattice'), default='primitives',
                   help='forecast candidates: the six primitives or the command lattice (C1)')
    p.add_argument('--footprint-clearing', choices=('on', 'off'), default='off',
                   help="drop map cells inside the body's outline")
    p.add_argument('--mover-trails', choices=('on', 'off'), default='off',
                   help="a confirmed obstacle track's cells stay in the map only while seen occupied")
    p.add_argument('--pose-oracle', choices=('on', 'off'), default='off', help='the true pose (module oracle)')
    p.add_argument('--map-oracle', choices=('obstacles', 'full', 'static', 'off'), default='off',
                   help='obstacles: perceived walls, true obstacles; full: the true map; static: the true map, '
                        'obstacles kept out of the route grid (module oracle)')
    p.add_argument('--prediction-oracle', choices=('on', 'off'), default='off',
                   help='true obstacle futures in the clearance check (module oracle)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    strips = None
    if a.strips:
        mu, _, mark = a.strips.partition(':')
        if mark not in ('marked', 'unmarked'):
            p.error('--strips needs MU:marked or MU:unmarked')
        strips = (float(mu), mark == 'marked')
    result = run(a.arm, a.set, a.maze, a.out, episode_index=a.episode, recovery=a.recovery == 'on', decoder_fit=a.decoder_fit,
                 friction=a.friction, strips=strips, stall_stop_s=a.stall_stop_s, obstacles=a.obstacles, degrade=a.degrade, margin=a.margin,
                 pessimistic=a.pessimistic, unknown_level=a.unknown_level, demo_textures=a.demo_textures,
                 models=yaml.safe_load(a.models.read_text()) if a.models else None, harness=a.harness,
                 coverage_fix=a.coverage_fix == 'on', map_decay=dict(on=True, floor='floor', off=False)[a.map_decay],
                 deadlock_escape=a.deadlock_escape == 'on', terminal_burst=a.terminal_burst == 'on',
                 obstacle_prediction=a.obstacle_prediction == 'on', footprint_clearing=a.footprint_clearing == 'on',
                 obstacle_series=a.obstacle_series, candidates=a.candidates, safety=a.safety,
                 obstacle_speed=tuple(map(float, a.obstacle_speed.split(':'))) if a.obstacle_speed else None,
                 pose_oracle=a.pose_oracle == 'on', map_oracle=None if a.map_oracle == 'off' else a.map_oracle,
                 prediction_oracle=a.prediction_oracle == 'on', mover_trails=a.mover_trails == 'on')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
