"""One simulated mission episode: settle, then 50 Hz command steps and 10 Hz sensor frames.

Timing: physics at 500 Hz, the locomotion policy at 50 Hz (10 physics steps each), frames at 10 Hz (every fifth
policy step). The session first **settles** for 1.5 s under a zero command (750 physics steps), then every
`step(requested)` executes one policy step and `frame()` renders the next sensor frame; frame k is at
1.5 s + k·100 ms. Optional dynamics perturbations (`go2nav.sim.dynamics`) change friction, mark the floor, raise the
speed guard on strip missions, or stop a recording mission that has stalled.

`step` applies the robot's safety limits, with the rate limit taken from the command executed at the start of the
current 100-ms window (so requests inside a window do not each get a fresh rate budget), then runs the policy and
10 physics steps. After every physics step the session records the robot state, the contacts and the sensors
(`go2nav.sim.sensors`), and stops the mission (`PhysicalStop`) on a disallowed contact, a body below 0.15 m or tilted
past 0.7 rad, or (after settling) the contact, speed and domain guard (`go2nav.sim.contacts`). With moving obstacles
(`go2nav.sim.obstacles`), every obstacle moves before each physics step after settling; they stand still while the
robot settles.
"""

import numpy as np

from go2nav.locomotion.policy import LocomotionPolicy
from go2nav.locomotion.safety import GUARD_SPEED_M_S, apply_limits
from go2nav.perception.cameras import AUXILIARY, PRIMARY
from go2nav.sim import contacts, dynamics as perturbations, scene as maze_scene, sensors
from go2nav.sim.obstacles import ObstacleField

SETTLE_TICKS = 15
FRAME_PERIOD_NS, FIRST_FRAME_NS = 100_000_000, 1_500_000_000
MIN_BODY_HEIGHT_M, MAX_TILT_RAD = .15, .70
NOISE_SIGMA_M = .002


class PhysicalStop(RuntimeError):
    """The mission ended physically (the reason is the message)."""


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class MissionSession:
    """The simulator side of one mission (see the module docstring). `samples` is the physics trace (one row per
    physics step); `contact_packets` and `contact_events` are the evaluator's contact records."""

    def __init__(self, spec, output_dir, *, policy_checkpoint, noise_sigma_m=NOISE_SIGMA_M, dynamics=None, noise_layout=None,
                 robot_visible=True, demo_textures=False, random_look=None, safety='primitive'):
        self.spec, self.output_dir, self.dynamics, self.safety = spec, output_dir, dynamics, safety
        self.world = maze_scene.build(spec, output_dir, dynamics, robot_visible=robot_visible, demo_textures=demo_textures,
                                      random_look=random_look)
        strips = dynamics is not None and dynamics.strips is not None
        self.friction_field = (perturbations.FrictionField(self.world.robot, self.world.floor, dynamics.strips, dynamics.strip_mu)
                               if strips else None)
        self.guard_speed_m_s = perturbations.STRIP_SPEED_LIMIT_M_S if strips else GUARD_SPEED_M_S[safety]
        self.stall = (perturbations.StallMonitor(dynamics.stall_stop_s)
                      if dynamics is not None and dynamics.stall_stop_s else None)
        self.policy = LocomotionPolicy(policy_checkpoint, device='cpu')
        self.policy.check_robot(self.world.robot)
        self.noise_layout = spec['layout_index'] % 4 if noise_layout is None else noise_layout
        self.noise_sigma_m = noise_sigma_m
        self.link_names = {int(l.idx): str(l.name) for e in self.world.scene.entities
                           if e not in self.world.surfaces for l in e.links}
        self.object_ids = {int(l.idx): 'ground_plane' for l in self.world.floor.links}
        for entity in self.world.walls:
            self.object_ids.update({int(l.idx): str(entity.name) for l in entity.links})
        self.obstacles = None
        if self.world.obstacles:
            self.obstacles = ObstacleField(spec, dynamics.obstacles, self.world.obstacles)
            for entity in self.world.obstacles:
                self.object_ids.update({int(l.idx): str(entity.name) for l in entity.links})
        self.sim_ns, self.frames, self.phase = 0, 0, 0
        self.last_executed = np.zeros((1, 3), dtype=np.float32)
        self.window, self.window_start_command = None, None
        self.samples, self.contact_packets, self.contact_times, self.contact_events = [], [], [], []
        self.base_xy = None  # the latest sampled base position (the obstacles yield to it; kept apart from `samples`)
        self.body_sensor = sensors.BodySensor()
        self.history = sensors.History(sensors.BODY_CHANNELS)
        self.fast_history = sensors.History((sensors.FAST_GYRO,))
        self.guard = None
        self._solver_fields = None
        self.packet_rows = []  # the policy histories at every frame (for training data)

    @property
    def solver_fields(self):
        """The mutable solver fields, for snapshots (`go2nav.sim.snapshot`)."""
        if self._solver_fields is None:
            from go2nav.sim.snapshot import solver_fields
            self._solver_fields = solver_fields(self.world.scene)
        return self._solver_fields

    # -- robot state ------------------------------------------------------------------------------------------------

    def _state(self):
        robot, dofs = self.world.robot, self.world.leg_dofs
        return dict(pos=array(robot.get_pos()), quat_wxyz=array(robot.get_quat()), vel=array(robot.get_vel()),
                    ang=array(robot.get_ang()), q=array(robot.get_dofs_position(dofs)), dq=array(robot.get_dofs_velocity(dofs)))

    def _act(self, command):
        """One policy evaluation for a (3,) float32 command and its joint targets applied."""
        s = self._state()
        quat_xyzw = np.stack([s['quat_wxyz'][..., i] for i in (1, 2, 3, 0)], axis=-1).astype(np.float32)
        targets = self.policy.act(quat_xyzw, s['ang'].astype(np.float32), s['q'].astype(np.float32),
                                  s['dq'].astype(np.float32), np.asarray(command[None, :], dtype=np.float32))
        self.world.robot.control_dofs_position(np.asarray(targets, dtype=np.float32), self.world.leg_dofs)

    # -- settle -----------------------------------------------------------------------------------------------------

    def settle(self, warm_renderer=True):
        """Reset to the spawn pose and stance, settle 1.5 s under a zero command, warm the renderer up (missions), and
        arm the guard. Returns the settled base pose (x, y, z, qx, qy, qz, qw)."""
        robot, dofs = self.world.robot, self.world.leg_dofs
        xyz, wxyz = maze_scene.spawn_pose(self.spec['geometry']['spawn_se2_world'])
        stance = self.policy.stance.reshape(1, 12).astype(np.float32)
        robot.set_pos(np.asarray(xyz, dtype=np.float32).reshape(1, 3), envs_idx=[0], zero_velocity=True)
        robot.set_quat(np.asarray(wxyz, dtype=np.float32).reshape(1, 4), envs_idx=[0], zero_velocity=False)
        robot.set_dofs_position(stance, dofs, envs_idx=[0])
        robot.set_dofs_velocity(np.zeros_like(stance), dofs, envs_idx=[0])
        self.policy.reset([0])
        self.last_executed.fill(0.)
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(SETTLE_TICKS):
            for _ in range(maze_scene.POLICY_STEPS_PER_COMMAND):
                self._act(zero)
                base_time = float(self.sim_ns)/1e9
                for k in range(maze_scene.PHYSICS_STEPS_PER_POLICY):
                    self.world.scene.step()
                    self._sample(zero, zero, base_time+(k+1)*maze_scene.PHYSICS_DT_S)
                self.sim_ns += int(maze_scene.POLICY_DT_S*1e9)
        if warm_renderer:
            self.render_pair()  # renderer warm-up; never an observation
        self.guard = self.world.guard_geoms
        return self.samples[-1]['base_pose_world']

    # -- 50 Hz step -------------------------------------------------------------------------------------------------

    def step(self, requested, record_as=None):
        """Execute one policy step for a requested command; returns the command actually executed. `record_as` is the
        request as the physics trace records it (default: the float32 command)."""
        if self.friction_field is not None:
            self.friction_field.update()
        recorded = requested if record_as is None else record_as
        requested = np.asarray(requested, dtype=np.float32)
        window = self.sim_ns//FRAME_PERIOD_NS
        if self.window != window:
            self.window, self.window_start_command = window, self.last_executed.copy()
        command = apply_limits(requested[None, None, :], self.window_start_command, self.safety)[0, 0]
        self._act(command)
        base_time, base_ns = float(self.sim_ns)/1e9, self.sim_ns
        for k in range(maze_scene.PHYSICS_STEPS_PER_POLICY):
            if self.obstacles is not None:
                self.obstacles.update(maze_scene.PHYSICS_DT_S, self.base_xy,
                                      base_time+(k+1)*maze_scene.PHYSICS_DT_S)
            self.world.scene.step()
            self.sim_ns = base_ns+(k+1)*2_000_000
            self.last_executed = command[None, :].copy()
            self._sample(requested if record_as is None else recorded, command, base_time+(k+1)*maze_scene.PHYSICS_DT_S)
        return command.tolist()

    def command_tick(self, requested):
        """One 100-ms command tick (five policy steps) as the older collections ran it; the request is recorded as
        given (float64)."""
        return [self.step(requested, record_as=np.asarray(requested, float)) for _ in range(maze_scene.POLICY_STEPS_PER_COMMAND)][0]

    def _sample(self, requested, applied, timestamp_s):
        """Record one physics step (state, contacts, sensors) and apply the stop rules."""
        s = self._state()
        pos, quat = np.asarray(s['pos'], np.float64).reshape(-1, 3)[0], np.asarray(s['quat_wxyz'], np.float64).reshape(-1, 4)[0]
        packet = contacts.native_contacts(self.world.robot)
        self.contact_packets.append(packet)
        self.contact_times.append(float(timestamp_s))
        disallowed = [r for r in contacts.attribute(packet, self.world.links, self.link_names, self.object_ids) if r['disallowed']]
        self.contact_events.append(dict(sample_index=len(self.samples), timestamp_s=float(timestamp_s), phase=self.phase,
                                        disallowed_contacts=disallowed))
        row = dict(timestamp_s=float(timestamp_s), base_pose_world=np.asarray([*pos, quat[1], quat[2], quat[3], quat[0]]),
                   base_twist_world=np.asarray([*np.asarray(s['vel'], np.float64).reshape(-1, 3)[0],
                                                *np.asarray(s['ang'], np.float64).reshape(-1, 3)[0]]),
                   joint_position=np.asarray(s['q'], np.float64).reshape(-1, 12)[0],
                   joint_velocity=np.asarray(s['dq'], np.float64).reshape(-1, 12)[0],
                   requested_command=np.asarray(requested, np.float64), post_slew_applied_command=np.asarray(applied, np.float64),
                   applied_command=np.asarray(applied, np.float64), physics_contact=np.uint8(bool(disallowed)),
                   phase=np.uint8(self.phase), edge_index=np.uint8(0))
        self.samples.append(row)
        self.base_xy = pos[:2].copy()
        try:
            if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):
                raise ValueError('nonfinite physical sample')
            rotation = sensors.rotation_xyzw(row['base_pose_world'][3:7])
            roll, pitch = np.arctan2(rotation[2, 1], rotation[2, 2]), np.arcsin(np.clip(-rotation[2, 0], -1, 1))
            if row['physics_contact']:
                raise PhysicalStop('DISALLOWED_CONTACT')
            if row['base_pose_world'][2] < MIN_BODY_HEIGHT_M or max(abs(roll), abs(pitch)) > MAX_TILT_RAD:
                raise PhysicalStop('BODY_STABILITY_LIMIT')
        finally:
            self._sense(row)
        if self.guard is not None:
            nonfoot = contacts.nonfoot_ground_contacts(packet, self.guard)
            speed = float(np.linalg.norm(row['base_twist_world'][:3]))
            inside = bool((np.abs(row['base_pose_world'][:2]) < contacts.GUARD_HALF_EXTENT_M).all())
            if nonfoot or speed > self.guard_speed_m_s or not inside:
                raise PhysicalStop('CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP')
        if self.stall is not None and self.stall.update(float(timestamp_s), row['base_pose_world'][:2]):
            raise PhysicalStop(perturbations.STALL_REASON)

    def _sense(self, row):
        ns = int(round(float(row['timestamp_s'])*1e9))
        if ns % 20_000_000 == 0:
            values = self.body_sensor.sample(ns, row['base_pose_world'][3:], row['base_twist_world'][:3],
                                             row['base_twist_world'][3:], row['joint_position'], row['joint_velocity'])
            for name, (value, valid) in values.items():
                self.history.append(name, value, valid, ns)
        if ns % FRAME_PERIOD_NS == 0:
            self.history.append('applied_command', row['applied_command'], np.ones(3, dtype=bool), ns)
        values, valid = sensors.gyro_sample(row['base_pose_world'][3:], row['base_twist_world'][3:])
        self.fast_history.append('gyro', values, valid, ns)

    # -- 10 Hz frames -----------------------------------------------------------------------------------------------

    def capture(self, aligned=True):
        """A primary-camera-only frame (the older corridor collections): (frame index, RGB, native depth, policy
        packet). Frames are at 1.5 s + k·100 ms, as `frame` (unless `aligned=False`: a frame at a physical stop)."""
        now, index = int(round(float(self.samples[-1]['timestamp_s'])*1e9)), self.frames
        if aligned and now != FIRST_FRAME_NS+index*FRAME_PERIOD_NS:
            raise ValueError('captures must be taken every fifth step after settling')
        rgb, depth = render_primary(self.world.robot, self.world.camera)
        policy = sensors.policy_packet(self.history, rgb, now)
        self._record_packet(policy, now)
        self.frames += 1
        return index, rgb, depth, policy

    def _record_packet(self, policy, now):
        row = dict(image_ns=np.int64(now), decision_ns=np.int64(now))
        for channel in sensors.BODY_CHANNELS:
            history = policy['sensor_state'][channel.role][channel.name]
            for field in ('values', 'valid', 'measured_ns', 'available_ns'):
                row[f'{channel.name}_{field}'] = history[field]
        self.packet_rows.append(row)

    def render_pair(self):
        """RGB and depth from the primary and auxiliary mounts at the current pose: [(rgb, depth), (rgb, depth)]."""
        return render_pair(self.world.robot, self.world.camera)

    def frame(self):
        """The next 10 Hz sensor frame: (policy, depth, fast gyro, auxiliary depth, auxiliary RGB, time) packets, with
        live depth noise."""
        now, index = self.sim_ns, self.frames
        if now != FIRST_FRAME_NS+index*FRAME_PERIOD_NS or len(self.samples) != 750+50*index:
            raise ValueError('frames must be taken every fifth step after settling')
        (rgb, depth), (auxiliary_rgb, auxiliary_depth) = self.render_pair()
        self.last_native_depth = (depth, auxiliary_depth)  # the renderer's depth before packaging and noise
        policy = sensors.policy_packet(self.history, rgb, now)
        self._record_packet(policy, now)
        depth = sensors.depth_packet(depth, policy, now)
        auxiliary_depth = sensors.depth_packet(auxiliary_depth, policy, now, auxiliary=True)
        fast = sensors.fast_gyro_packet(self.fast_history, now)
        if self.noise_sigma_m:
            depth = sensors.add_depth_noise(depth, layout=self.noise_layout, frame=index, camera=0, sigma_m=self.noise_sigma_m)
            auxiliary_depth = sensors.add_depth_noise(auxiliary_depth, layout=self.noise_layout, frame=index, camera=1,
                                                      sigma_m=self.noise_sigma_m)
        auxiliary_rgb = sensors.auxiliary_rgb_packet(auxiliary_rgb, auxiliary_depth, policy, now)
        self.frames += 1
        return policy, depth, fast, auxiliary_depth, auxiliary_rgb, now


def render_primary(robot, camera):
    """The primary camera's RGB and depth at the robot's current pose (one RGB and one depth render)."""
    p = array(robot.get_pos()).reshape(-1, 3)[0]
    q = array(robot.get_quat()).reshape(-1, 4)[0]
    R = sensors.rotation_xyzw(q[[1, 2, 3, 0]])
    position = p+R@PRIMARY[:3, 3]
    camera.set_pose(pos=position, lookat=position+R[:, 0], up=R[:, 2])
    rgb = np.asarray(camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0]).reshape(480, 640, 3).copy()
    depth = np.asarray(camera.render(rgb=False, depth=True, segmentation=False, normal=False)[1]).reshape(480, 640).copy()
    return rgb, depth


def render_pair(robot, camera):
    """Render RGB and depth from the primary and auxiliary mounts at the robot's current pose (the one camera is moved
    to each mount, then back to the primary): [(rgb uint8, depth float32), (rgb, depth)]."""
    p = array(robot.get_pos()).reshape(-1, 3)[0]
    q = array(robot.get_quat()).reshape(-1, 4)[0]
    R = sensors.rotation_xyzw(q[[1, 2, 3, 0]])
    images = []
    for E in (PRIMARY, AUXILIARY):
        H = np.eye(4)
        H[:3, :3], H[:3, 3] = R@E[:3, :3], p+R@E[:3, 3]
        camera.set_pose(pos=H[:3, 3], lookat=H[:3, 3]+H[:3, 2], up=-H[:3, 1])
        np.testing.assert_allclose(np.asarray(camera.transform, float)@np.diag([1., -1., -1., 1.]), H, atol=1e-6, rtol=0)
        rgb = np.asarray(camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0]).reshape(480, 640, 3).copy()
        depth = np.asarray(camera.render(rgb=False, depth=True, segmentation=False, normal=False)[1]).reshape(480, 640).copy()
        if rgb.dtype != np.uint8 or depth.dtype != np.float32:
            raise ValueError('uint8 RGB and float32 depth expected from the renderer')
        images.append((rgb, depth))
    position = p+R@PRIMARY[:3, 3]
    camera.set_pose(pos=position, lookat=position+R[:, 0], up=R[:, 2])
    return images


def rotation_from_pose(base_pose_world):
    return sensors.rotation_xyzw(np.asarray(base_pose_world)[3:])


def settled_task_cues(episode, base_pose_world):
    """The mission targets in the settled initial-body frame (xy): the goal beacon and the home position."""
    p, R = np.asarray(base_pose_world[:3], float), rotation_from_pose(base_pose_world)
    A = R[:2, :2].T
    return dict(goal_initial_body_xy_m=(A@(np.asarray(episode['beacon_xy_world'], float)-p[:2])).tolist(),
                return_initial_body_xy_m=(A@(np.asarray(episode['home_se2_world'][:2], float)-p[:2])).tolist(),
                require_return_after_goal=True)
