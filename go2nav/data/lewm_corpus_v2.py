"""Corpus v2 (docs/dynamic/PLAN.md, P3): the LeWM corpus with moving obstacles and continuous commands, on the
continuous-contract policy, collected on the lewm_sets_v1 mazes like corpus v1 (`go2nav.data.lewm_corpus`).

    python -m go2nav.data.lewm_corpus_v2 collect ROOT [--roles lewm_val lewm_train] [--mazes A-B] [--workers N]
                                                 [--egl-devices D [D ...]]
    python -m go2nav.data.lewm_corpus_v2 unit ROOT ROLE MAZE KIND
    python -m go2nav.data.lewm_corpus_v2 index ROOT [--length L]
    python -m go2nav.data.lewm_corpus_v2 audit ROOT                # the data gates (L0 and L0 dynamic)
    python -m go2nav.data.lewm_corpus_v2 stats ROOT                # the command statistics of `command_dim=3` models

**Units** are v1's (explore, mission, free; transfer on the validation mazes), with the same frame caps, resume and
pool, and seeds keyed by the version as well. What changes:
- **Policy and harness:** the continuous-contract policy, and C1 with its lattice refit (`MODELS`), driving the
  lattice baseline's harness (lv2, `HARNESS`: 25 commands, no lateral, the continuous safety block), recovery off.
- **Explore:** with probability 0.25 a plan's command is replaced by a random command of the corpus lattice (`AXES`:
  forward -0.10 to 0.45 m/s, lateral +-0.15 m/s, yaw +-0.6 rad/s), lateral with probability `LATERAL_SHARE`, unless
  the free driver's stand-off rule, on the true pose, walls and obstacles, rejects it (`FreeDriverV2.admissible`: a
  random reverse or sideways step goes where no camera looks). The harness and its dispatch checks are unchanged; a
  scripted backup is never replaced.
- **Free driver** (`FreeDriverV2`): v1's segments at the range's speeds and with lateral commands, plus sideways steps
  and **encounters** (when an obstacle is in sight: approach it, wait 1-5 s, then back off, step aside, turn away or
  wait on), under v1's stand-off rule for any direction of travel, with the obstacles as well as the walls.
- **Moving obstacles** in every unit (`unit_obstacles`): a count drawn from `COUNTS` (weighted to one and two), placed
  as in the missions (`go2nav.sim.obstacles`; the free units' start stands in for home and beacon), and non-polite
  (`polite=False`: they wait at the robot instead of yielding) in `IMPOLITE_SHARE` of the units that have obstacles.

**Records** add, per unit folder, dynamic.npz: `obstacles_xy` (n, k, 2) world positions at each frame, `waiting`
(n, k) whether each obstacle stands stopped for the robot, `obstacle_pixels` (n,) the primary depth image's pixels on
an obstacle (true geometry), and `plan_command` (n, 3) the command planned at the frame (NaN if none). steps.npz keeps
v1's schema (`plan_action` is -1: lattice plans have no primitive index), so v1's loaders read v2 units. Every record
of the obstacles is a label, never a model input.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from go2nav.data import lewm_corpus as v1

VERSION = 'v2'
REPO = v1.REPO
MODELS = REPO/'outputs/locomotion/continuous_policy_c1_lattice.yaml'
HARNESS = dict(map_decay='floor', deadlock_escape=True, terminal_burst=True, obstacle_prediction=True,
               footprint_clearing=True)
AXES = dict(forward=(-.10, 0., .15, .30, .45), lateral=(-.15, 0., .15), yaw=(-.6, -.3, 0., .3, .6))
LATERAL_SHARE = .75
COUNTS, COUNT_P = (0, 1, 2, 3, 4), (.15, .30, .30, .15, .10)
IMPOLITE_SHARE = .2
MIN_OBSTACLE_PIXELS = 200       # an obstacle counts as in view from this many depth pixels
NEAR_M, RELATIVE_SPEED_M_S = 1., .05
GATES = dict(in_view=.20, waiting=.02, lateral=.10, contact_stops=.05)


def load_models():
    """The artefact overrides (`MODELS`), resolved against the repository."""
    return {name: str(REPO/path) for name, path in yaml.safe_load(MODELS.read_text()).items()}


def unit_seed(maze, kind, *keys):
    return v1.keyed_seed(VERSION, v1.SET, maze, kind, *keys)


def random_command(rng):
    """A random command of the corpus lattice (`AXES`): lateral with probability `LATERAL_SHARE`."""
    lateral = [v for v in AXES['lateral'] if v] if rng.random() < LATERAL_SHARE else [0.]
    return tuple(float(rng.choice(values)) for values in (AXES['forward'], lateral, AXES['yaw']))


def unit_obstacles(spec, home_se2, beacon_xy, maze, kind):
    """(placement or None, record) of a unit's moving obstacles (see the module docstring). A layout with fewer free
    cells than the drawn count holds as many as it can."""
    from go2nav.sim import obstacles as moving
    rng = np.random.default_rng(unit_seed(maze, kind, 'obstacles'))
    count, polite = int(rng.choice(COUNTS, p=COUNT_P)), bool(rng.random() >= IMPOLITE_SHARE)
    episode = dict(home_se2_world=list(home_se2), beacon_xy_world=list(beacon_xy))
    while count:
        try:
            placement = moving.place_obstacles(spec, episode, count, unit_seed(maze, kind, 'placement'), polite=polite)
            return placement, dict(count=count, polite=polite)
        except ValueError:
            count -= 1
    return None, dict(count=0, polite=None)


# -- recording --------------------------------------------------------------------------------------------------------

class Recorder(v1.Recorder):
    """v1's records plus dynamic.npz (see the module docstring)."""

    def __init__(self):
        super().__init__()
        self.obstacles_xy, self.waiting, self.obstacle_pixels, self.plan_command = [], [], [], []

    def frame(self, policy, measured_ns, sample, *, walkers=(), depth=None, truth=None):
        super().frame(policy, measured_ns, sample)
        self.obstacles_xy.append(np.array([w.xy for w in walkers], np.float32).reshape(-1, 2))
        self.waiting.append(np.array([w.blocked_for > 0 for w in walkers], bool))
        pixels = 0
        if walkers and truth is not None:
            from go2nav.perception import cameras
            pixels = int(truth.obstacle_pixels(depth['depth_m'], depth['valid'], cameras.PRIMARY).sum())
        self.obstacle_pixels.append(pixels)
        self.plan_command.append(np.full(3, np.nan, np.float32))

    def save(self, folder):
        n, blocks = super().save(folder)
        k = self.obstacles_xy[0].shape[0] if n else 0
        np.savez(folder/'dynamic.npz',
                 obstacles_xy=np.stack(self.obstacles_xy) if n else np.zeros((0, k, 2), np.float32),
                 waiting=np.stack(self.waiting) if n else np.zeros((0, k), bool),
                 obstacle_pixels=np.asarray(self.obstacle_pixels, np.int32),
                 plan_command=np.stack(self.plan_command) if n else np.zeros((0, 3), np.float32))
        return n, blocks


def exploring(planner, rng, epsilon, admissible=lambda command: True):
    """Wrap `planner.select`: with probability `epsilon` a random corpus-lattice command (`random_command`) replaces
    the choice (never a scripted backup, nor a command `admissible` rejects), marked in the selection as
    `exploration`."""
    from go2nav.planning.actions import name
    original = planner.select

    def select(*args, **kwargs):
        result = original(*args, **kwargs)
        draw, substitute = rng.random(), random_command(rng)
        if not planner.backup_plan and draw < epsilon and admissible(substitute):
            result = dict(result, action=name(*substitute), action_index=-1, requested_command=substitute,
                          exploration=dict(planner_action=result['action']))
        return result
    planner.select = select


def run_mission(spec, episode, recorder, *, cap, epsilon, seed, meshes, dynamics):
    """C1 on the lattice harness drives a mission (optionally exploring); returns the stop reason (None if the mission
    or the cap ended it)."""
    from go2nav.controllers import arms
    from go2nav.controllers.controller import NavigationController, SensorFrame
    from go2nav.controllers.truth import GroundTruth
    from go2nav.perception.rigid import PoseLost
    from go2nav.planning.actions import LATTICE, command
    from go2nav.sim.session import MissionSession, PhysicalStop, settled_task_cues
    models = load_models()
    session = MissionSession(spec, meshes, policy_checkpoint=Path(models['go2_locomotion_policy']), dynamics=dynamics,
                             safety='continuous')
    cues = settled_task_cues(episode, session.settle())
    truth = GroundTruth(session)
    walkers = session.obstacles.walkers if session.obstacles is not None else []
    controller = NavigationController(arms.forecaster('C1', models=models), goal_xy=cues['goal_initial_body_xy_m'],
                                      home_xy=cues['return_initial_body_xy_m'], arrival_radius_m=.02,
                                      navigation_frames=4800, recovery=False, candidates=LATTICE, **HARNESS)
    if epsilon:
        geometry = FreeDriverV2(spec, np.random.default_rng(seed), movers=lambda: [w.xy for w in walkers])
        exploring(controller.planner, np.random.default_rng(seed), epsilon,
                  admissible=lambda c: geometry.admissible(c, session.samples[-1]['base_pose_world']))
    history = deque(maxlen=4)
    try:
        for tick in range(cap*v1.FRAME_STEPS):
            now = session.sim_ns
            if tick % v1.FRAME_STEPS == 0:
                policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.frame()
                history.append(policy)
                recorder.frame(policy, measured, session.samples[-1], walkers=walkers, depth=depth, truth=truth)
                planned = len(controller.planning)
                controller.on_frame(SensorFrame(tick//v1.FRAME_STEPS, measured, policy['image']['rgb'],
                                                (depth['depth_m'], depth['valid']), auxiliary_rgb['rgb'],
                                                (auxiliary_depth['depth_m'], auxiliary_depth['valid']), fast['values'],
                                                policy['sensor_state']['sensed']['specific_force']['values'], policy,
                                                tuple(history)))
                if len(controller.planning) > planned and 'action' in controller.planning[-1]:
                    row = controller.planning[-1]
                    recorder.plan_command[-1] = np.asarray(command(row['action']), np.float32)
                    recorder.explored[-1] = 'exploration' in row['selection']
            request = controller.request(now)['requested_command']
            session.phase = 2
            recorder.step(request, session.step(request))
            if controller.terminal is not None and tick % v1.FRAME_STEPS == v1.FRAME_STEPS-1:
                return None
    except PhysicalStop as error:
        return str(error)
    except PoseLost:
        return 'POSE_LOST'
    return None


# -- the free driver --------------------------------------------------------------------------------------------------

class FreeDriverV2(v1.FreeDriver):
    """v1's free driver (see `go2nav.data.lewm_corpus.FreeDriver`) over the continuous command range, with the moving
    obstacles (`movers`: a callable giving their current world xy; privileged, a collector only). Segments, drawn at
    random (encounters only while an obstacle is in sight, their weight otherwise shared out):
    - harness structure (15%): four-tick commitments of corpus-lattice commands (`random_command`), one-tick
      translation pulses, and starts after 1.2-2 s of rest;
    - cruise (15%): v1's, at 0.15, 0.30 or 0.45 m/s, or a steady arc (0.16-0.30 m/s, yaw 0.3-0.6 rad/s);
    - Ornstein-Uhlenbeck commands (15%): forward -0.10 to 0.45 m/s, yaw +-0.6 rad/s, and in three in four of them
      lateral +-0.15 m/s;
    - wall approaches (10%): v1's, at 0.12, 0.20 or 0.30 m/s;
    - sideways steps (20%): lateral 0.15 m/s for 0.5-2 s towards the side with more room, reversing where the
      stand-off stops it, alone or with yaw +-0.3 rad/s;
    - encounters (25%): face the nearest obstacle in sight, approach at 0.15 or 0.30 m/s, steering at it, until
      0.7-1.1 m from its centre, wait 1-5 s, then back off (-0.10 m/s, 1-3 s), step aside, turn away or wait on.
    The stand-off rule (`admissible`) is v1's for any direction of travel: a translation whose way along its direction
    of travel is shorter than 0.45 m (0.30 m sideways: the body is 0.24 m wide), or that would bring the body centre
    closer to a wall or an obstacle's surface and within 0.45 m of it in the next 0.6 s, becomes a turn in place.
    Below 0.43 m it steps out, forward, back or sideways, whichever leaves it clearest, until 0.47 m clear."""

    WEIGHTS = dict(harness=.15, cruise=.15, ou=.15, wall=.10, side=.20, encounter=.25)
    SIDE_AHEAD_M = .30

    def __init__(self, spec, rng, movers=lambda: ()):
        super().__init__(spec, rng)
        self.movers = movers

    def wall_clearance(self, xy):
        return super().clearance(xy)

    def clearance(self, xy):
        """Distance from xy to the nearest wall or obstacle surface."""
        walls = self.wall_clearance(xy)
        from go2nav.sim.obstacles import RADIUS_M
        movers = [float(np.linalg.norm(np.asarray(xy, float)-m))-RADIUS_M for m in self.movers()]
        return min([walls]+movers)

    def in_sight(self, xy):
        """The nearest obstacle whose centre the walls do not hide from xy (or None)."""
        best = None
        for m in self.movers():
            d = np.asarray(m, float)-np.asarray(xy, float)
            distance = float(np.linalg.norm(d))
            steps = np.arange(.05, max(distance-.2, .05), .05)
            if all(self.wall_clearance(np.asarray(xy, float)+s*d/distance) >= .05 for s in steps):
                if best is None or distance < best[0]:
                    best = (distance, np.asarray(m, float))
        return None if best is None else best[1]

    def next(self, pose):
        while True:
            if self.segment is None:
                kinds = [k for k in self.WEIGHTS if k != 'encounter' or self.in_sight(pose[:2]) is not None]
                weights = np.array([self.WEIGHTS[k] for k in kinds])
                kind = kinds[int(self.rng.choice(len(kinds), p=weights/weights.sum()))]
                self.segment = (list(self.WEIGHTS).index(kind), getattr(self, f'_{kind}')())
                next(self.segment[1])
            try:
                command = self.segment[1].send(pose)
                break
            except StopIteration:
                self.segment = None
        return self._safe(command, pose), self.segment[0]

    def _harness(self):
        yield
        for _ in range(int(self.rng.integers(30, 61))//4):
            draw = self.rng.random()
            command = random_command(self.rng)
            if draw < .2:
                for _ in range(int(self.rng.integers(12, 21))):
                    yield (0., 0., 0.)
                tape = [command]*4
            elif draw < .32 and any(command[:2]):
                tape = [command]+[(0., 0., 0.)]*3
            else:
                tape = [command]*4
            for c in tape:
                yield c

    def _cruise(self):
        pose = yield
        h0 = v1.yaw_of(pose)
        headings = [h0+k*math.pi/4 for k in range(8)]
        room = [self.free_distance(pose[:2], h) for h in headings]
        best = max(range(8), key=lambda k: (room[k], -min(k, 8-k)))
        pose = yield from self._face(pose, headings[best])
        if self.rng.random() < .25:
            speed, arc = float(self.rng.uniform(.16, .30)), float(self.rng.choice((-.6, -.45, -.3, .3, .45, .6)))
            for _ in range(int(self.rng.integers(6, 15))):
                pose = yield (speed, 0., arc)
        else:
            speed = float(self.rng.choice((.15, .30, .45)))
            for _ in range(int(self.rng.integers(15, 41))):
                if self.free_distance(pose[:2], v1.yaw_of(pose)) < .6+speed:
                    break
                pose = yield (speed, 0., 0.)
        for _ in range(int(self.rng.integers(0, 6))):
            pose = yield (0., 0., 0.)

    def _ou(self):
        yield
        sideways = self.rng.random() < .75
        v, s, w, dt = self.rng.uniform(0, .3), 0., self.rng.uniform(-.5, .5), .1
        for _ in range(int(self.rng.integers(20, 41))):
            v += 1.5*(.2-v)*dt+.18*math.sqrt(dt)*self.rng.normal()
            s += 1.5*(0.-s)*dt+.15*math.sqrt(dt)*self.rng.normal() if sideways else 0.
            w += 1.5*(0.-w)*dt+.6*math.sqrt(dt)*self.rng.normal()
            v, s, w = float(np.clip(v, -.10, .45)), float(np.clip(s, -.15, .15)), float(np.clip(w, -.6, .6))
            yield (v, s, w)

    def _wall(self):
        pose = yield
        target = self.nearest_wall_point(np.asarray(pose[:2]))
        pose = yield from self._face(pose, math.atan2(target[1]-pose[1], target[0]-pose[0]))
        speed, stand_off = float(self.rng.choice((.12, .20, .30))), self.rng.uniform(.48, .60)
        for _ in range(40):
            if self.wall_clearance(pose[:2]) < stand_off:
                break
            pose = yield (speed, 0., 0.)
        for _ in range(int(self.rng.integers(3, 11))):
            pose = yield (0., 0., 0.)
        turn = .45 if self.rng.random() < .5 else -.45
        for _ in range(int(self.rng.integers(8, 21))):
            pose = yield (0., 0., turn)

    def _side(self):
        pose = yield
        h = v1.yaw_of(pose)
        room = [self.free_distance(pose[:2], h+side*math.pi/2, limit=1.) for side in (1, -1)]
        lateral = .15 if room[0] >= room[1] else -.15
        yaw = float(self.rng.choice((0., 0., -.3, .3)))
        for _ in range(int(self.rng.integers(5, 21))):
            if not self.admissible((0., lateral, yaw), pose):
                lateral = -lateral
            pose = yield (0., lateral, yaw)

    def _encounter(self):
        pose = yield
        target = self.in_sight(pose[:2])
        if target is None:
            return
        pose = yield from self._face(pose, math.atan2(target[1]-pose[1], target[0]-pose[0]))
        speed, stop_at = float(self.rng.choice((.15, .30))), float(self.rng.uniform(.7, 1.1))
        for _ in range(60):
            target = self.in_sight(pose[:2])
            if target is None or np.linalg.norm(target-np.asarray(pose[:2])) < stop_at:
                break
            error = math.atan2(target[1]-pose[1], target[0]-pose[0])-v1.yaw_of(pose)
            error = math.atan2(math.sin(error), math.cos(error))
            pose = yield (speed, 0., float(np.clip(2*error, -.45, .45)))
        for _ in range(int(self.rng.integers(10, 51))):
            pose = yield (0., 0., 0.)
        then = int(self.rng.choice(4))
        if then == 0:
            for _ in range(int(self.rng.integers(10, 31))):
                pose = yield (-.10, 0., 0.)
        elif then == 1:
            lateral = .15 if self.rng.random() < .5 else -.15
            for _ in range(int(self.rng.integers(5, 16))):
                pose = yield (0., lateral, 0.)
        elif then == 2:
            turn = .45 if self.rng.random() < .5 else -.45
            for _ in range(int(self.rng.integers(8, 21))):
                pose = yield (0., 0., turn)
        else:
            for _ in range(int(self.rng.integers(10, 41))):
                pose = yield (0., 0., 0.)

    def admissible(self, command, pose):
        """Whether a command keeps the stand-off (see the class docstring): any command while the body is clear of the
        escape distance, a translation only if its way is open and it does not close on anything within 0.45 m."""
        xy, h = np.asarray(pose[:2], float), v1.yaw_of(pose)
        now = self.clearance(xy)
        vx, vy, wz = command
        if not (vx or vy):
            return now >= self.ESCAPE_M or not wz
        path, heading = [xy], h
        for _ in range(6):  # the next 0.6 s of the command, in 0.1-s steps
            heading += wz*.1
            c, s = math.cos(heading), math.sin(heading)
            path.append(path[-1]+.1*np.array([c*vx-s*vy, s*vx+c*vy]))
        closest = min(self.clearance(q) for q in path[1:])
        open_way = self.free_distance(xy, h+math.atan2(vy, vx), limit=1.) >= (self.SIDE_AHEAD_M if abs(vy) > abs(vx)
                                                                                else self.AHEAD_M)
        return open_way and not (closest < self.BODY_M and closest < now)

    def _escape(self, xy, h):
        """The step at 0.12 m/s, forward, back or sideways, whose position 0.3 m on is clearest."""
        c, s = math.cos(h), math.sin(h)
        steps = ((.12, 0.), (-.12, 0.), (0., .12), (0., -.12))
        world = lambda v: np.array([c*v[0]-s*v[1], s*v[0]+c*v[1]])/.12   # the step's unit direction in the world
        return (*max(steps, key=lambda v: self.clearance(xy+.3*world(v))), 0.)

    def _safe(self, command, pose):
        xy, h = np.asarray(pose[:2], float), v1.yaw_of(pose)
        now = self.clearance(xy)
        if now < self.ESCAPE_M or self.escaping:
            self.escaping = now < self.CLEAR_M
            if self.escaping:
                return self._escape(xy, h)
        vx, vy, wz = command
        if self.admissible(command, pose):
            self.blocked_turn = None
            return (vx, vy, wz)
        if self.blocked_turn is None:
            self.blocked_turn = .45 if self.rng.random() < .5 else -.45
        return (0., 0., wz if wz else self.blocked_turn)


def run_free(spec, recorder, *, cap, seed, meshes, random_look, dynamics):
    from go2nav.controllers.truth import GroundTruth
    from go2nav.sim.session import MissionSession, PhysicalStop
    session = MissionSession(spec, meshes, policy_checkpoint=Path(load_models()['go2_locomotion_policy']),
                             random_look=random_look, dynamics=dynamics, safety='continuous')
    session.settle()
    truth = GroundTruth(session)
    walkers = session.obstacles.walkers if session.obstacles is not None else []
    driver = FreeDriverV2(spec, np.random.default_rng(seed+1), movers=lambda: [w.xy for w in walkers])
    segments = []
    try:
        for tick in range(cap*v1.FRAME_STEPS):
            if tick % v1.FRAME_STEPS == 0:
                policy, depth, *_, measured = session.frame()
                recorder.frame(policy, measured, session.samples[-1], walkers=walkers, depth=depth, truth=truth)
                command, code = driver.next(session.samples[-1]['base_pose_world'])
                recorder.plan_command[-1] = np.asarray(command, np.float32)
                segments.append(code)
            session.phase = 2
            recorder.step(command, session.step(command))
    except PhysicalStop as error:
        return str(error), segments
    return None, segments


def collect_unit(root, role, maze, kind):
    """Run one unit into ROOT/units/<role>/<maze>_<kind> (atomically). Returns its meta record."""
    import cv2
    import torch
    from go2nav.eval.mission import episode_spec
    from go2nav.sim import scene
    from go2nav.sim.dynamics import Dynamics
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)
    folder = Path(root)/'units'/role/v1.unit_name(maze, kind)
    partial = folder.with_name(folder.name+'.tmp')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started, seed = time.monotonic(), unit_seed(maze, kind)
    recorder, segments, look = Recorder(), None, None
    if kind in ('free', 'transfer'):
        from go2nav.sim import random_appearance
        spec, _ = episode_spec(v1.SET, maze, 0, root=v1.mazes_root(root))
        spec = v1.free_spec(spec, seed)
        spawn = spec['geometry']['spawn_se2_world']
        placement, obstacles = unit_obstacles(spec, spawn, spawn[:2], maze, kind)
        look = random_appearance.sample(unit_seed(maze, kind, 'look'), held_out=kind == 'transfer')
        scene.init_genesis(spec['procedural_seed'])
        stop, segments = run_free(spec, recorder, cap=v1.FRAME_CAPS[kind], seed=seed, meshes=partial/'meshes',
                                  random_look=look, dynamics=Dynamics(obstacles=placement) if placement else None)
    else:
        spec, episode = episode_spec(v1.SET, maze, v1.EPISODES[kind], root=v1.mazes_root(root))
        placement, obstacles = unit_obstacles(spec, episode['home_se2_world'], episode['beacon_xy_world'], maze, kind)
        scene.init_genesis(spec['procedural_seed'])
        stop = run_mission(spec, episode, recorder, cap=v1.FRAME_CAPS[kind],
                           epsilon=v1.EPSILON if kind == 'explore' else 0., seed=seed, meshes=partial/'meshes',
                           dynamics=Dynamics(obstacles=placement) if placement else None)
    frames, blocks = recorder.save(partial)
    if segments is not None:
        np.save(partial/'segments.npy', np.asarray(segments, np.int8))
    shutil.rmtree(partial/'meshes', ignore_errors=True)
    meta = dict(schema='go2nav_lewm_corpus_unit.v2', set=v1.SET, role=role, maze=maze, kind=kind, seed=seed,
                episode=v1.EPISODES.get(kind), recovery=False, epsilon=v1.EPSILON if kind == 'explore' else 0.,
                frame_cap=v1.FRAME_CAPS[kind], frames=frames, valid_blocks=blocks, physical_stop=stop, look=look,
                obstacles=dict(obstacles, placement=placement), start_se2_world=spec['geometry']['spawn_se2_world'],
                wall_s=round(time.monotonic()-started, 1), code=v1.code_identity(),
                models={k: str(Path(v).relative_to(REPO)) for k, v in load_models().items()})
    (partial/'meta.json').write_text(json.dumps(meta, indent=1)+'\n')
    shutil.rmtree(folder, ignore_errors=True)
    partial.rename(folder)
    (folder/'.done').touch()
    return meta


# -- audit ------------------------------------------------------------------------------------------------------------

def _unit_dynamic(folder):
    meta = json.loads((folder/'meta.json').read_text())
    with np.load(folder/'steps.npz') as steps:
        executed, valid, pose, twist = steps['executed'], steps['block_valid'], steps['pose'], steps['twist']
    with np.load(folder/'dynamic.npz') as dynamic:
        xy, waiting, pixels = dynamic['obstacles_xy'], dynamic['waiting'], dynamic['obstacle_pixels']
    n, ticks = len(pose), executed[:, 0]
    row = dict(kind=meta['kind'], count=meta['obstacles']['count'], polite=meta['obstacles']['polite'], frames=n,
               blocks=int(valid.sum()), lateral_blocks=int((np.abs(executed[valid, :, 1]) > 1e-3).any(axis=1).sum()),
               forward=np.histogram(ticks[valid, 0], bins=11, range=(-.125, .475))[0],
               lateral=np.histogram(ticks[valid, 1], bins=3, range=(-.225, .225))[0],
               yaw=np.histogram(ticks[valid, 2], bins=13, range=(-.65, .65))[0],
               in_view=0, moving_in_view=0, waiting=0, near=0, stop=meta['physical_stop'] or 'none')
    if n and xy.shape[1]:
        seen = pixels >= MIN_OBSTACLE_PIXELS
        mover_v = np.gradient(xy, .1, axis=0) if n > 1 else np.zeros_like(xy)
        relative = np.linalg.norm(mover_v-twist[:, None, :2], axis=-1).max(axis=1)
        distance = np.linalg.norm(xy-pose[:, None, :2], axis=-1).min(axis=1)
        row.update(in_view=int(seen.sum()), moving_in_view=int((seen&(relative > RELATIVE_SPEED_M_S)).sum()),
                   waiting=int(waiting.any(axis=1).sum()), near=int((distance < NEAR_M).sum()))
    return row


def audit(root, *, workers=None):
    """v1's L0 report (`go2nav.data.lewm_corpus.audit`) plus the dynamic gate: command coverage (forward, lateral and
    yaw histograms of the executed ticks; lateral in at least `GATES['lateral']` of the blocks), and in the units
    with obstacles the share of frames with an obstacle in view (at least `GATES['in_view']`), in view and moving
    relative to the robot, within 1 m, and waiting for the robot (an interaction; at least `GATES['waiting']`).
    Contact stops stay under 5% of units, by kind too."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    root = Path(root)
    report = v1.audit(root, workers=workers)
    folders = v1.unit_folders(root)
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(_unit_dynamic, folders, chunksize=8))
    with_obstacles = [r for r in rows if r['count']]
    frames = sum(r['frames'] for r in with_obstacles) or 1
    share = lambda key: round(sum(r[key] for r in with_obstacles)/frames, 4)
    blocks = sum(r['blocks'] for r in rows) or 1
    kinds = sorted({r['kind'] for r in rows})
    contacts = {k: sum(r['stop'] == 'DISALLOWED_CONTACT' for r in rows if r['kind'] == k) for k in kinds}
    units = {k: sum(r['kind'] == k for r in rows) for k in kinds}
    dynamic = dict(
        units_by_obstacle_count={str(c): sum(r['count'] == c for r in rows) for c in COUNTS},
        non_polite_units=sum(r['polite'] is False for r in rows),
        lateral_block_share=round(sum(r['lateral_blocks'] for r in rows)/blocks, 4),
        forward_histogram=np.sum([r['forward'] for r in rows], axis=0).tolist(),
        lateral_histogram=np.sum([r['lateral'] for r in rows], axis=0).tolist(),
        yaw_histogram=np.sum([r['yaw'] for r in rows], axis=0).tolist(),
        in_view_share=share('in_view'), moving_in_view_share=share('moving_in_view'), near_share=share('near'),
        waiting_share=share('waiting'), contact_stops_by_kind={k: f'{contacts[k]}/{units[k]}' for k in kinds})
    dynamic['gate'] = dict(
        lateral_at_least=dynamic['lateral_block_share'] >= GATES['lateral'],
        obstacle_in_view_at_least=dynamic['in_view_share'] >= GATES['in_view'],
        waiting_at_least=dynamic['waiting_share'] >= GATES['waiting'],
        contact_stops_below_by_kind=all(contacts[k] < GATES['contact_stops']*units[k] for k in kinds))
    report['dynamic'] = dynamic
    (root/'audit.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def command_stats(root):
    """Mean and standard deviation of the executed (forward, lateral, yaw) commands of the training units' valid
    blocks, written where `command_dim=3` world models read them (`go2nav.models.lewm.CONTROL_STATS_3`)."""
    from go2nav.models.lewm import CONTROL_STATS_3
    total, square, count = np.zeros(3), np.zeros(3), 0
    for folder in v1.unit_folders(root, roles=['lewm_train']):
        with np.load(folder/'steps.npz') as steps:
            commands = steps['executed'][steps['block_valid']].reshape(-1, 3).astype(np.float64)
        total, square, count = total+commands.sum(0), square+(commands**2).sum(0), count+len(commands)
    mean = total/count
    stats = dict(description='Mean and standard deviation of executed commands [forward m/s, lateral m/s, yaw rad/s] '
                             'on the corpus v2 training units (valid blocks); normalise the action blocks of world '
                             'models with command_dim 3.',
                 control_mean=mean.tolist(), control_std=np.sqrt(square/count-mean**2).tolist(), commands=int(count),
                 source=f'go2nav.data.lewm_corpus_v2 stats ({Path(root).name})')
    CONTROL_STATS_3.write_text(json.dumps(stats, indent=1)+'\n')
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('root', type=Path)
    c.add_argument('--roles', nargs='+', default=['lewm_val', 'lewm_train'])
    c.add_argument('--mazes', help='maze range A-B (default: every maze of the roles)')
    c.add_argument('--workers', type=int)
    c.add_argument('--egl-devices', nargs='+',
                   help='EGL devices for rendering, used in turn (default: EGL_DEVICE_ID, else 0)')
    u = sub.add_parser('unit')
    u.add_argument('root', type=Path)
    u.add_argument('role')
    u.add_argument('maze', type=int)
    u.add_argument('kind', choices=v1.KINDS)
    i = sub.add_parser('index')
    i.add_argument('root', type=Path)
    i.add_argument('--length', type=int, default=12)
    a = sub.add_parser('audit')
    a.add_argument('root', type=Path)
    s = sub.add_parser('stats')
    s.add_argument('root', type=Path)
    args = p.parse_args()
    if args.command == 'unit':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('EGL_DEVICE_ID', '0')
        print(json.dumps(collect_unit(args.root, args.role, args.maze, args.kind)))
    elif args.command == 'collect':
        sys.exit(1 if v1.collect(args.root, args.roles, mazes=v1.parse_range(args.mazes), workers=args.workers,
                                 egl_devices=args.egl_devices, module=__spec__.name) else 0)
    elif args.command == 'index':
        print(json.dumps(v1.build_index(args.root, length=args.length), indent=1))
    elif args.command == 'audit':
        print(json.dumps(audit(args.root), indent=1))
    else:
        print(json.dumps(command_stats(args.root), indent=1))


if __name__ == '__main__':
    main()
