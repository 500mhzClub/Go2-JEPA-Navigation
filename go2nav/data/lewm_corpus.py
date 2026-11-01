"""The LeWM training corpus: Go2 camera video with the commands the robot executed, collected in the mission simulator
on the lewm_sets_v1 mazes (5 x 4 grid; configs/mazes/lewm_sets_v1.yaml).

    python -m go2nav.data.lewm_corpus collect ROOT [--roles lewm_val lewm_train] [--mazes A-B] [--workers N]
                                              [--egl-devices D [D ...]]
    python -m go2nav.data.lewm_corpus unit ROOT ROLE MAZE KIND     # one unit (the pool runs each in its own process)
    python -m go2nav.data.lewm_corpus index ROOT                   # the window index that training reads
    python -m go2nav.data.lewm_corpus audit ROOT                   # the data gate (L0)

ROOT/mazes/lewm_sets_v1 is generated from the config on first use and must equal the committed registry
(data/mazes/registries/lewm_sets_v1_registry.json).

**Units.** Each maze gives three units, each one simulator episode in its own process:
- `explore` (episode 0): C1 drives the mission (recovery on), but at each plan, with probability 0.25, a uniformly
  random one of the six actions replaces the planner's choice. Harness timing and the dispatch safety checks are
  unchanged, and a scripted backup is never replaced. This gives on-policy states with counterfactual actions, so
  the state alone cannot predict the action. At most 900 frames;
- `mission` (episode 1): C1 unaltered, recovery on (even mazes) or off (odd): the deployed state distribution; at most
  1,000 frames;
- `free`: from a random cell and heading, a scripted driver (no planner) with ground-truth wall distances
  (`FreeDriver`): the harness's action structure, straight cruises and steady arcs into open space,
  Ornstein-Uhlenbeck commands and wall approaches, under a privileged stand-off rule; 600 frames. It supplies the
  steady runs and starts from rest that the missions, which turn a lot in short corridors, rarely make. Its scene has
  a random look (`go2nav.sim.random_appearance`: surface families, CC0 textures, light direction, colour and
  intensity), where the missions keep the benchmark's grey cells and light (their visual tracker is tuned to them);
- `transfer` (validation mazes only): a free unit under the held-out look (textures and a dim, colour-cast light that
  no training unit draws), for the appearance-transfer test.
A unit ends early at a physical stop (kept), or, for missions, when the mission ends or the controller loses its
visual pose (`POSE_LOST`; the frames up to it are kept).

**Records** (ROOT/units/<role>/<maze>_<kind>/): frames.npy (n, 168, 224, 3) uint8 (`lewm.resize_frame` of the primary
RGB), steps.npz and meta.json. Per frame i of steps.npz:
- `executed` (n, 5, 3): the commands the locomotion layer executed in the five policy steps after frame i (frame
  i's action block); `block_valid` is false where the block is incomplete (the last frame, a physical stop);
- `requested` (n, 5, 3): the requests of those steps;
- `applied_history` (n, 15, 3): the applied-command history the policy packet carries (what a forecaster sees);
- `pose` (n, 7: x, y, z, qx, qy, qz, qw) and `twist` (n, 6) at the frame: labels, never model inputs;
- `plan_action`: the action planned at frame i (-1 if none); `explored`: it was the random substitute.

**Rendering** runs on the EGL devices given, in turn (default: `EGL_DEVICE_ID`, else 0). On a machine with an
integrated GPU as device 0, give the discrete GPU's device: the integrated one saturates. Devices render a unit
bit-identically, so the choice changes speed only.

**Resume.** A unit is written to <name>.tmp and renamed when complete, then marked .done; `collect` skips finished
units and clears partial ones, so a stopped collection resumes where it was. A unit that fails or exceeds its time
limit is listed and retried by the next `collect`. Seeds are keyed by (maze, kind), so a rerun unit reproduces its
data (CPU physics is deterministic).
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SET = 'lewm_sets_v1'
SET_CONFIG = REPO/'configs/mazes/lewm_sets_v1.yaml'
SET_REGISTRY = REPO/'data/mazes/registries/lewm_sets_v1_registry.json'
KINDS = ('explore', 'mission', 'free', 'transfer')
ROLE_KINDS = dict(lewm_train=('explore', 'mission', 'free'), lewm_val=('explore', 'mission', 'free', 'transfer'))
FRAME_CAPS = dict(explore=900, mission=1000, free=600, transfer=600)
EPISODES = dict(explore=0, mission=1)
EPSILON = .25
UNIT_TIMEOUT_S = 3*3600
FRAME_STEPS = 5
MEMORY_GIB, MEMORY_MARGIN_GIB = 3.5, 6
MIN_TYPE_WINDOWS = 20_000


def keyed_seed(*keys):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'big') % (2**31-1)


def unit_name(maze, kind):
    return f'{maze:04d}_{kind}'


def mazes_root(root):
    return Path(root)/'mazes'


def ensure_mazes(root):
    """Generate the maze set under ROOT/mazes once, and check it against the committed registry."""
    folder = mazes_root(root)/SET
    if not (folder/'registry.json').exists():
        partial = mazes_root(root)/f'{SET}.tmp'
        shutil.rmtree(partial, ignore_errors=True)
        subprocess.run([sys.executable, '-m', 'go2nav.mazes.cli', str(SET_CONFIG), '--out', str(partial), '--name', SET],
                       cwd=REPO, check=True)
        (partial/SET).rename(folder)
        partial.rmdir()
    if (folder/'registry.json').read_bytes() != SET_REGISTRY.read_bytes():
        raise ValueError(f'{folder} does not match the committed registry {SET_REGISTRY}')
    return folder


# -- recording --------------------------------------------------------------------------------------------------------

class Recorder:
    """Frames and per-frame records of one unit (see the module docstring)."""

    def __init__(self):
        from go2nav.models.lewm import resize_frame
        self.resize = resize_frame
        self.frames, self.ns, self.history, self.pose, self.twist = [], [], [], [], []
        self.executed, self.requested, self.plan_action, self.explored = [], [], [], []

    def frame(self, policy, measured_ns, sample):
        self.frames.append(self.resize(policy['image']['rgb']))
        self.ns.append(measured_ns)
        self.history.append(np.asarray(policy['sensor_state']['control']['applied_command']['values'], np.float32))
        self.pose.append(np.asarray(sample['base_pose_world'], np.float64))
        self.twist.append(np.asarray(sample['base_twist_world'], np.float64))
        self.executed.append([])
        self.requested.append([])
        self.plan_action.append(-1)
        self.explored.append(False)

    def step(self, requested, executed):
        self.requested[-1].append(np.asarray(requested, np.float32))
        self.executed[-1].append(np.asarray(executed, np.float32))

    def save(self, folder):
        n = len(self.frames)
        valid = np.array([len(e) == FRAME_STEPS for e in self.executed], bool)
        block = lambda rows: np.stack([np.stack(r) if len(r) == FRAME_STEPS else np.zeros((FRAME_STEPS, 3), np.float32)
                                       for r in rows]) if n else np.zeros((0, FRAME_STEPS, 3), np.float32)
        np.save(folder/'frames.npy', np.stack(self.frames) if n else np.zeros((0, 168, 224, 3), np.uint8))
        np.savez(folder/'steps.npz', frame_ns=np.asarray(self.ns, np.int64), executed=block(self.executed),
                 requested=block(self.requested), block_valid=valid,
                 applied_history=np.stack(self.history) if n else np.zeros((0, 15, 3), np.float32),
                 pose=np.stack(self.pose) if n else np.zeros((0, 7)), twist=np.stack(self.twist) if n else np.zeros((0, 6)),
                 plan_action=np.asarray(self.plan_action, np.int8), explored=np.asarray(self.explored, bool))
        return n, int(valid.sum())


def exploring(planner, rng, epsilon):
    """Wrap `planner.select`: with probability `epsilon` a uniformly random action replaces the choice (never a
    scripted backup). The substitute is marked in the selection as `exploration`."""
    from go2nav.planning.actions import ACTIONS, command
    original = planner.select

    def select(*args, **kwargs):
        result = original(*args, **kwargs)
        draw, index = rng.random(), int(rng.integers(len(ACTIONS)))
        if not planner.backup_plan and draw < epsilon:
            action = ACTIONS[index]
            result = dict(result, action=action, action_index=index, requested_command=command(action),
                          exploration=dict(planner_action=result['action']))
        return result
    planner.select = select


def run_mission(spec, episode, recorder, *, cap, recovery, epsilon, seed, meshes):
    """C1 drives a mission (optionally exploring); returns the stop reason (None if the mission or the cap ended it)."""
    from go2nav import artifacts
    from go2nav.controllers import arms
    from go2nav.controllers.controller import NavigationController, SensorFrame
    from go2nav.perception.rigid import PoseLost
    from go2nav.planning.actions import ACTIONS
    from go2nav.sim.session import MissionSession, PhysicalStop, settled_task_cues
    session = MissionSession(spec, meshes, policy_checkpoint=artifacts.path('go2_locomotion_policy'))
    cues = settled_task_cues(episode, session.settle())
    controller = NavigationController(arms.forecaster('C1'), goal_xy=cues['goal_initial_body_xy_m'],
                                      home_xy=cues['return_initial_body_xy_m'], arrival_radius_m=.02,
                                      navigation_frames=4800, recovery=recovery)
    if epsilon:
        exploring(controller.planner, np.random.default_rng(seed), epsilon)
    history = deque(maxlen=4)
    try:
        for tick in range(cap*FRAME_STEPS):
            now = session.sim_ns
            if tick % FRAME_STEPS == 0:
                policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.frame()
                history.append(policy)
                recorder.frame(policy, measured, session.samples[-1])
                planned = len(controller.planning)
                controller.on_frame(SensorFrame(tick//FRAME_STEPS, measured, policy['image']['rgb'],
                                                (depth['depth_m'], depth['valid']), auxiliary_rgb['rgb'],
                                                (auxiliary_depth['depth_m'], auxiliary_depth['valid']), fast['values'],
                                                policy['sensor_state']['sensed']['specific_force']['values'], policy,
                                                tuple(history)))
                if len(controller.planning) > planned and 'action' in controller.planning[-1]:
                    row = controller.planning[-1]
                    recorder.plan_action[-1] = ACTIONS.index(row['action'])
                    recorder.explored[-1] = 'exploration' in row['selection']
            request = controller.request(now)['requested_command']
            session.phase = 2
            recorder.step(request, session.step(request))
            if controller.terminal is not None and tick % FRAME_STEPS == FRAME_STEPS-1:
                return None
    except PhysicalStop as error:
        return str(error)
    except PoseLost:  # the controller's visual pose is lost (random actions can cause it): the episode ends here
        return 'POSE_LOST'
    return None


# -- the free driver --------------------------------------------------------------------------------------------------

def yaw_of(pose):
    qx, qy, qz, qw = pose[3:7]
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


def planar_yaw(quaternions):
    """Yaw (rad) of quaternions (..., 4) in the corpus's (x, y, z, w) order."""
    x, y, z, w = np.moveaxis(np.asarray(quaternions), -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def body_motion(pose, start, end):
    """The corpus's motion labels: the body-frame motion (forward, left, yaw change) from frame `start` to frame `end`
    of a unit's poses (n, 7); `start` and `end` are index arrays of one shape, the result (*shape, 3)."""
    yaw = planar_yaw(pose[:, 3:7])
    c, s = np.cos(yaw[start]), np.sin(yaw[start])
    d = pose[end, :2]-pose[start, :2]
    return np.stack([c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1], np.angle(np.exp(1j*(yaw[end]-yaw[start])))], -1)


class FreeDriver:
    """Commands for the free unit, one per 100-ms tick, from the ground-truth pose and the maze's wall boxes (a data
    collector only: privileged state never reaches a model). Segments, drawn at random:
    - harness structure (35%): four-tick commitments of the six actions, one-tick translation pulses, and starts after
      1.2-2 s of rest (the decisions the planner makes, without a planner);
    - cruise (30%): turn in place to the most open of eight headings, then drive straight (or, one time in four, a
      steady arc) until the way ahead closes: the long steady runs the missions rarely make in short corridors;
    - Ornstein-Uhlenbeck commands (20%): forward 0-0.22 m/s, yaw within the +-0.5 rad/s limit;
    - wall approaches (15%): turn to the nearest wall, drive in until 0.48-0.60 m from it, hold, turn away.
    The stand-off rule then removes any translation whose way ahead is shorter than 0.45 m, or that would bring the
    body centre closer to a wall and within 0.45 m of it in the next 0.6 s, turning in place instead; and below
    0.43 m (the body's half-diagonal is about 0.38 m, so a turn there could touch) it backs straight out along the
    body axis, forward or in reverse, until 0.47 m clear."""

    PROBE_M, AHEAD_M, BODY_M, ESCAPE_M, CLEAR_M = .20, .45, .45, .43, .47

    def __init__(self, spec, rng):
        boxes = spec['geometry']['wall_boxes']
        self.centres = np.array([b['centre_xyz'][:2] for b in boxes], float)
        self.halves = np.array([b['size_xyz'][:2] for b in boxes], float)/2
        self.rng, self.segment, self.blocked_turn, self.escaping = rng, None, None, False

    def clearance(self, xy):
        outside = np.maximum(np.abs(np.asarray(xy, float)[None]-self.centres)-self.halves, 0.)
        return float(np.sqrt((outside**2).sum(axis=1)).min())

    def free_distance(self, xy, heading, limit=3.):
        """How far a thin probe (radius 0.2 m) can travel from xy along heading before touching a wall."""
        direction = np.array([math.cos(heading), math.sin(heading)])
        for d in np.arange(0., limit, .05):
            if self.clearance(np.asarray(xy, float)+d*direction) < self.PROBE_M:
                return float(d)
        return limit

    def nearest_wall_point(self, xy):
        closest = np.clip(np.asarray(xy, float)[None], self.centres-self.halves, self.centres+self.halves)
        return closest[np.argmin(((closest-xy)**2).sum(axis=1))]

    def next(self, pose):
        """(command [vx, 0, wz], segment code) for the next tick."""
        while True:
            if self.segment is None:
                kind = int(self.rng.choice(4, p=[.35, .30, .20, .15]))
                self.segment = (kind, (self._harness, self._cruise, self._ou, self._wall)[kind]())
                next(self.segment[1])
            try:
                command = self.segment[1].send(pose)
                break
            except StopIteration:
                self.segment = None
        return self._safe(command, pose), self.segment[0]

    def _face(self, pose, heading):
        for _ in range(30):
            error = heading-yaw_of(pose)
            error = math.atan2(math.sin(error), math.cos(error))
            if abs(error) < .12:
                break
            pose = yield (0., 0., .45 if error > 0 else -.45)
        return pose

    def _harness(self):
        from go2nav.planning.actions import ACTIONS, TRANSLATIONS, command
        yield
        for _ in range(int(self.rng.integers(30, 61))//4):
            draw = self.rng.random()
            if draw < .2:
                for _ in range(int(self.rng.integers(12, 21))):
                    yield (0., 0., 0.)
                tape = [command(TRANSLATIONS[int(self.rng.integers(3))])]*4
            elif draw < .32:
                tape = [command(TRANSLATIONS[int(self.rng.integers(3))])]+[(0., 0., 0.)]*3
            else:
                tape = [command(ACTIONS[int(self.rng.choice(6, p=[.12, .34, .15, .15, .12, .12]))])]*4
            for c in tape:
                yield c

    def _cruise(self):
        pose = yield
        h0 = yaw_of(pose)
        headings = [h0+k*math.pi/4 for k in range(8)]
        room = [self.free_distance(pose[:2], h) for h in headings]
        best = max(range(8), key=lambda k: (room[k], -min(k, 8-k)))
        pose = yield from self._face(pose, headings[best])
        if self.rng.random() < .25:
            arc = .45 if self.rng.random() < .5 else -.45
            for _ in range(int(self.rng.integers(6, 15))):
                pose = yield (.16, 0., arc)
        else:
            for _ in range(int(self.rng.integers(15, 41))):
                if self.free_distance(pose[:2], yaw_of(pose)) < .6:
                    break
                pose = yield (.20, 0., 0.)
        for _ in range(int(self.rng.integers(0, 6))):
            pose = yield (0., 0., 0.)

    def _ou(self):
        yield
        v, w, dt = self.rng.uniform(0, .2), self.rng.uniform(-.4, .4), .1
        for _ in range(int(self.rng.integers(20, 41))):
            v += 1.5*(.12-v)*dt+.12*math.sqrt(dt)*self.rng.normal()
            w += 1.5*(0.-w)*dt+.5*math.sqrt(dt)*self.rng.normal()
            v, w = float(np.clip(v, 0., .22)), float(np.clip(w, -.5, .5))
            yield (v, 0., w)

    def _wall(self):
        pose = yield
        target = self.nearest_wall_point(np.asarray(pose[:2]))
        pose = yield from self._face(pose, math.atan2(target[1]-pose[1], target[0]-pose[0]))
        speed, stand_off = (.12, .2)[int(self.rng.integers(2))], self.rng.uniform(.48, .60)
        for _ in range(40):
            if self.clearance(pose[:2]) < stand_off:
                break
            pose = yield (speed, 0., 0.)
        for _ in range(int(self.rng.integers(3, 11))):
            pose = yield (0., 0., 0.)
        turn = .45 if self.rng.random() < .5 else -.45
        for _ in range(int(self.rng.integers(8, 21))):
            pose = yield (0., 0., turn)

    def _safe(self, command, pose):
        xy, h = np.asarray(pose[:2], float), yaw_of(pose)
        now = self.clearance(xy)
        if now < self.ESCAPE_M or self.escaping:
            # Too close to turn safely (a corner could touch): back straight out along the body axis, towards the
            # side with more room, until 0.47 m clear.
            self.escaping = now < self.CLEAR_M
            if self.escaping:
                ahead, behind = self.free_distance(xy, h, limit=1.), self.free_distance(xy, h+math.pi, limit=1.)
                return (.12 if ahead >= behind else -.12, 0., 0.)
        vx, _, wz = command
        if vx > 0:
            path, heading = [xy], h
            for _ in range(6):  # the next 0.6 s of the command, in 0.1-s steps
                heading += wz*.1
                path.append(path[-1]+vx*.1*np.array([math.cos(heading), math.sin(heading)]))
            closest = min(self.clearance(q) for q in path[1:])
            if self.free_distance(xy, h, limit=1.) < self.AHEAD_M or (closest < self.BODY_M and closest < now):
                if self.blocked_turn is None:
                    self.blocked_turn = .45 if self.rng.random() < .5 else -.45
                return (0., 0., wz if wz else self.blocked_turn)
        self.blocked_turn = None
        return (vx, 0., wz)


def run_free(spec, recorder, *, cap, seed, meshes, random_look):
    from go2nav import artifacts
    from go2nav.sim.session import MissionSession, PhysicalStop
    driver = FreeDriver(spec, np.random.default_rng(seed+1))
    session = MissionSession(spec, meshes, policy_checkpoint=artifacts.path('go2_locomotion_policy'), random_look=random_look)
    session.settle()
    segments = []
    try:
        for tick in range(cap*FRAME_STEPS):
            if tick % FRAME_STEPS == 0:
                policy, *_, measured = session.frame()
                recorder.frame(policy, measured, session.samples[-1])
                command, code = driver.next(session.samples[-1]['base_pose_world'])
                segments.append(code)
            session.phase = 2
            recorder.step(command, session.step(command))
    except PhysicalStop as error:
        return str(error), segments
    return None, segments


def free_spec(spec, seed):
    """The maze with a random start: a random cell centre (+-0.15 m) and a uniform heading; its own physics seed."""
    from go2nav.mazes import grid
    rng = np.random.default_rng(seed)
    shape = grid.Shape(*spec['grid_shape']) if 'grid_shape' in spec else grid.SQUARE
    cell = np.asarray(shape.cells[int(rng.integers(len(shape.cells)))], float)
    xy = cell*grid.PITCH_M+rng.uniform(-.15, .15, 2)
    return spec | dict(procedural_seed=seed,
                       geometry=spec['geometry'] | dict(spawn_se2_world=[*xy.tolist(), float(rng.uniform(-math.pi, math.pi))]))


def collect_unit(root, role, maze, kind):
    """Run one unit into ROOT/units/<role>/<maze>_<kind> (atomically). Returns its meta record."""
    import cv2
    import torch
    from go2nav.eval.mission import episode_spec
    from go2nav.sim import scene
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)
    folder = Path(root)/'units'/role/unit_name(maze, kind)
    partial = folder.with_name(folder.name+'.tmp')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started, seed = time.monotonic(), keyed_seed(SET, maze, kind)
    recorder, segments, look = Recorder(), None, None
    if kind in ('free', 'transfer'):
        from go2nav.sim import random_appearance
        spec, _ = episode_spec(SET, maze, 0, root=mazes_root(root))
        spec = free_spec(spec, seed)
        look = random_appearance.sample(keyed_seed(SET, maze, kind, 'look'), held_out=kind == 'transfer')
        scene.init_genesis(spec['procedural_seed'])
        stop, segments = run_free(spec, recorder, cap=FRAME_CAPS[kind], seed=seed, meshes=partial/'meshes', random_look=look)
        recovery = None
    else:
        spec, episode = episode_spec(SET, maze, EPISODES[kind], root=mazes_root(root))
        scene.init_genesis(spec['procedural_seed'])
        recovery = kind == 'explore' or maze % 2 == 0
        stop = run_mission(spec, episode, recorder, cap=FRAME_CAPS[kind], recovery=recovery,
                           epsilon=EPSILON if kind == 'explore' else 0., seed=seed, meshes=partial/'meshes')
    frames, blocks = recorder.save(partial)
    if segments is not None:
        np.save(partial/'segments.npy', np.asarray(segments, np.int8))
    shutil.rmtree(partial/'meshes', ignore_errors=True)
    meta = dict(schema='go2nav_lewm_corpus_unit.v1', set=SET, role=role, maze=maze, kind=kind, seed=seed,
                episode=EPISODES.get(kind), recovery=recovery, epsilon=EPSILON if kind == 'explore' else 0.,
                frame_cap=FRAME_CAPS[kind], frames=frames, valid_blocks=blocks, physical_stop=stop, look=look,
                start_se2_world=spec['geometry']['spawn_se2_world'], wall_s=round(time.monotonic()-started, 1),
                code=code_identity())
    (partial/'meta.json').write_text(json.dumps(meta, indent=1)+'\n')
    shutil.rmtree(folder, ignore_errors=True)
    partial.rename(folder)
    (folder/'.done').touch()
    return meta


def code_identity():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return dict(commit=commit or None)


# -- the collection pool ----------------------------------------------------------------------------------------------

def plan_units(root, roles, mazes=None, role_kinds=ROLE_KINDS):
    from go2nav.mazes.sets import MazeSet
    maze_set = MazeSet(SET, root=mazes_root(root))
    return [(role, maze, kind) for role in roles for maze in maze_set.maze_ids(role)
            if mazes is None or maze in mazes for kind in role_kinds[role]]


def available_memory():
    with open('/proc/meminfo') as stream:
        info = dict(line.split(':', 1) for line in stream)
    return int(info['MemAvailable'].split()[0])*1024


def collect(root, roles, *, mazes=None, workers=None, egl_devices=None, module='go2nav.data.lewm_corpus',
            role_kinds=ROLE_KINDS, memory_gib=MEMORY_GIB):
    """Run every unit not yet done, `workers` at a time (default: cores - 2), as memory allows. Resumable. `module`
    runs each unit (`python -m MODULE unit ...`; another corpus version reuses this pool)."""
    root = Path(root)
    egl_devices = tuple(egl_devices or (os.environ.get('EGL_DEVICE_ID', '0'),))
    ensure_mazes(root)
    units = plan_units(root, roles, mazes, role_kinds)
    pending = [u for u in units if not (root/'units'/u[0]/unit_name(*u[1:])/'.done').exists()]
    workers = workers or max(1, (os.cpu_count() or 2)-2)
    print(json.dumps(dict(units=len(units), pending=len(pending), workers=workers)), flush=True)
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    failures = []

    def run(unit):
        role, maze, kind = unit
        device = egl_devices[units.index(unit) % len(egl_devices)]
        with open(logs/f'{role}_{unit_name(maze, kind)}.log', 'w') as log:
            try:
                code = subprocess.run([sys.executable, '-m', module, 'unit', str(root), role, str(maze), kind],
                                      cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=UNIT_TIMEOUT_S,
                                      env=os.environ | dict(EGL_DEVICE_ID=device, PYOPENGL_PLATFORM='egl')).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        if code != 0 or not (root/'units'/role/unit_name(maze, kind)/'.done').exists():
            raise RuntimeError(f'exit {code}')
        return json.loads((root/'units'/role/unit_name(maze, kind)/'meta.json').read_text())

    done, started = 0, time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        futures, queue = {}, list(pending)
        while queue or futures:
            while queue and len(futures) < workers and available_memory() > (memory_gib+MEMORY_MARGIN_GIB)*1024**3:
                unit = queue.pop(0)
                futures[pool.submit(run, unit)] = unit
            if not futures:
                time.sleep(5)
                continue
            finished, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                unit = futures.pop(future)
                try:
                    meta = future.result()
                    done += 1
                    rate = done/(time.monotonic()-started)
                    print(json.dumps(dict(done=f'{unit[0]}/{unit_name(*unit[1:])}', frames=meta['frames'],
                                          stop=meta['physical_stop'], wall_s=meta['wall_s'], completed=done,
                                          remaining=len(queue)+len(futures), eta_h=round((len(queue)+len(futures))/rate/3600, 2))),
                          flush=True)
                except Exception as error:
                    failures.append(f'{unit[0]}/{unit_name(*unit[1:])}: {error}')
                    print(json.dumps(dict(failed=failures[-1])), flush=True)
    print(json.dumps(dict(completed=done, failures=failures)), flush=True)
    return failures


# -- index and audit --------------------------------------------------------------------------------------------------

def unit_folders(root, roles=None):
    base = Path(root)/'units'
    return sorted(f for role in (roles or sorted(p.name for p in base.iterdir() if p.is_dir()))
                  for f in (base/role).iterdir() if (f/'.done').exists())


def _unit_windows(args):
    folder, length = args
    with np.load(folder/'steps.npz') as steps:
        valid = steps['block_valid']
    n = len(valid)
    starts = [s for s in range(n-length+1) if valid[s:s+length-1].all()]
    meta = json.loads((folder/'meta.json').read_text())
    return dict(path=str(folder.relative_to(folder.parents[2])), role=meta['role'], maze=meta['maze'], kind=meta['kind'],
                frames=n, physical_stop=meta['physical_stop']), np.asarray(starts, np.int32)


def build_index(root, *, length=12, workers=None):
    """ROOT/index_<length>.npz (unit, start of every window of `length` frames with complete action blocks) and
    ROOT/index_<length>.json (the units). Windows never cross units, so never a reset or a physical stop."""
    root = Path(root)
    folders = unit_folders(root)
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        results = list(pool.map(_unit_windows, [(f, length) for f in folders], chunksize=16))
    units = [r[0] for r in results]
    unit = np.concatenate([np.full(len(s), i, np.int32) for i, (_, s) in enumerate(results)]) if results else np.zeros(0, np.int32)
    start = np.concatenate([s for _, s in results]) if results else np.zeros(0, np.int32)
    np.savez(root/f'index_{length}.npz', unit=unit, start=start)
    (root/f'index_{length}.json').write_text(json.dumps(dict(length=length, units=units), indent=0)+'\n')
    summary = Counter()
    for u, s in zip(units, (s for _, s in results)):
        summary[f"{u['role']}/{u['kind']}/frames"] += u['frames']
        summary[f"{u['role']}/{u['kind']}/windows"] += len(s)
    return dict(units=len(units), windows=int(len(start)), **dict(sorted(summary.items())))


def _unit_audit(folder):
    from go2nav.planning.actions import movement
    with np.load(folder/'steps.npz') as steps:
        executed, valid, pose = steps['executed'], steps['block_valid'], steps['pose']
    ticks = executed[:, 0]  # the command of each tick (constant within a tick unless a check intervened)
    moves = Counter()
    for t in range(15, len(ticks)-8, 2):
        if valid[t-15:t+8].all():
            moves[movement(ticks[t:t+8], ticks[t-15:t])] += 1
    mid_tick = int((np.abs(executed-executed[:, :1]).max(axis=(1, 2)) > 1e-6)[valid].sum())
    yaw = np.unwrap([yaw_of(p) for p in pose])
    rate = np.diff(yaw)/.1
    lag = {}
    for k in range(6):
        a, b = ticks[:len(rate)-k, 2], rate[k:]
        ok = valid[:len(rate)-k]
        lag[k] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0 else 0.
    return dict(moves=moves, mid_tick=mid_tick, blocks=int(valid.sum()), lag=lag,
                forward=np.histogram(ticks[valid, 0], bins=8, range=(-.3, .3))[0],
                yaw=np.histogram(ticks[valid, 2], bins=10, range=(-.5, .5))[0])


def audit(root, *, workers=None):
    """The L0 data gate: movement-type counts (at least 20,000 windows of each type; the shares follow the missions,
    which turn a lot) and shares, command histograms, mid-tick command changes, physical stops (contact stops under 5%
    of units), and the command-to-motion lag (correlation of the tick's yaw command with the measured yaw rate k ticks
    later)."""
    root = Path(root)
    folders = unit_folders(root)
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        rows = list(pool.map(_unit_audit, folders, chunksize=8))
    moves = sum((r['moves'] for r in rows), Counter())
    total = sum(moves.values())
    stops = Counter(json.loads((f/'meta.json').read_text())['physical_stop'] or 'none' for f in folders)
    lag = {k: float(np.mean([r['lag'][k] for r in rows])) for k in range(6)}
    blocks = sum(r['blocks'] for r in rows)
    report = dict(units=len(folders), blocks=blocks,
                  movement_shares={k: round(v/total, 4) for k, v in sorted(moves.items())} if total else {},
                  mid_tick_blocks=sum(r['mid_tick'] for r in rows), physical_stops=dict(stops),
                  yaw_command_to_rate_correlation_by_lag_ticks=lag,
                  forward_histogram=np.sum([r['forward'] for r in rows], axis=0).tolist(),
                  yaw_histogram=np.sum([r['yaw'] for r in rows], axis=0).tolist())
    counts = {k: 2*v for k, v in moves.items()}  # every second window was classified
    report['movement_windows'] = dict(sorted(counts.items()))
    report['gate'] = dict(each_movement_type_at_least_20k_windows=len(counts) == 6 and min(counts.values()) >= MIN_TYPE_WINDOWS,
                          contact_stops_below_5pct=stops.get('DISALLOWED_CONTACT', 0) < .05*len(folders))
    (root/'audit.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


def parse_range(text):
    if not text:
        return None
    lo, _, hi = text.partition('-')
    return set(range(int(lo), int(hi or lo)+1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('root', type=Path)
    c.add_argument('--roles', nargs='+', default=['lewm_val', 'lewm_train'])
    c.add_argument('--mazes', help='maze range A-B (default: every maze of the roles)')
    c.add_argument('--workers', type=int)
    c.add_argument('--egl-devices', nargs='+', help='EGL devices for rendering, used in turn (default: EGL_DEVICE_ID, else 0)')
    u = sub.add_parser('unit')
    u.add_argument('root', type=Path)
    u.add_argument('role')
    u.add_argument('maze', type=int)
    u.add_argument('kind', choices=KINDS)
    i = sub.add_parser('index')
    i.add_argument('root', type=Path)
    i.add_argument('--length', type=int, default=12)
    a = sub.add_parser('audit')
    a.add_argument('root', type=Path)
    args = p.parse_args()
    if args.command == 'unit':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('EGL_DEVICE_ID', '0')
        print(json.dumps(collect_unit(args.root, args.role, args.maze, args.kind)))
    elif args.command == 'collect':
        sys.exit(1 if collect(args.root, args.roles, mazes=parse_range(args.mazes), workers=args.workers,
                              egl_devices=args.egl_devices) else 0)
    elif args.command == 'index':
        print(json.dumps(build_index(args.root, length=args.length), indent=1))
    else:
        print(json.dumps(audit(args.root), indent=1))


if __name__ == '__main__':
    main()
