"""Vectorised Genesis environment for training the Go2 locomotion policy with PPO.

Ported from the Genesis Go2 locomotion example (examples/locomotion/go2_env.py, Genesis-Embodied-AI/Genesis, Apache
License 2.0), the environment the shipped policy was trained in. One change: commands are drawn from a fixed bank of
velocity primitives (configs/locomotion/contract_ppo.yaml) rather than uniformly from ranges, so the policy learns exactly
the commands the navigation stack issues.

Each step: the policy's 12 joint-position offsets (applied one control step late, as on the real robot) are scaled and
added to the default stance, tracked by joint PD control at 50 Hz with two physics substeps.

Observation (45 values, see `OBSERVATION_LAYOUT`): body angular velocity, gravity in the body frame, the scaled command,
joint positions relative to the stance, joint velocities, and the previous action.

Rewards: exponential tracking of the commanded planar velocity and yaw rate, minus penalties on vertical velocity, base
height error, action rate and distance from the default pose. Episodes end after 20 s or on roll/pitch above 10 degrees.
"""
import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

OBSERVATION_LAYOUT = (('base_ang_vel_body', 3), ('projected_gravity', 3), ('command', 3), ('joint_pos_offset', 12),
                      ('joint_vel', 12), ('last_action', 12))
CONTROL_DT_S = 0.02


class Go2LocomotionEnv:
    def __init__(self, num_envs, cfg, show_viewer=False, friction=None):
        """`cfg` is the loaded configs/locomotion/contract_ppo.yaml. `friction`, if given, is set on the floor and every
        robot geometry (Genesis uses the larger coefficient of a contact pair); training uses the default."""
        self.num_envs = num_envs
        self.env_cfg, self.obs_cfg, self.reward_cfg = cfg['env'], cfg['obs'], cfg['reward']
        self.num_actions = self.env_cfg['num_actions']
        self.num_commands = 3
        self.device = gs.device
        self.cfg = self.env_cfg  # read by rsl-rl
        self.dt = CONTROL_DT_S
        self.max_episode_length = math.ceil(self.env_cfg['episode_length_s']/self.dt)
        self.obs_scales = self.obs_cfg['obs_scales']
        self.reward_scales = {k: v*self.dt for k, v in self.reward_cfg['reward_scales'].items()}

        bank = torch.tensor(list(cfg['primitives'].values()), dtype=gs.tc_float, device=gs.device)
        self.command_bank, self.command_low, self.command_high = bank, bank.min(dim=0).values, bank.max(dim=0).values
        self.command_jitter_std = float(cfg.get('command_jitter_std', 0.0))

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5, max_collision_pairs=20),
            viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.5),
                                                    camera_fov=40, max_FPS=int(1.0/self.dt)),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer)
        self.floor = self.scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))
        self.robot = self.scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=self.env_cfg['base_init_pos'],
                                                          quat=self.env_cfg['base_init_quat']))
        self.scene.build(n_envs=num_envs)
        if friction is not None:
            self.robot.set_friction(float(friction))
            self.floor.set_friction(float(friction))

        self.motors_dof_idx = torch.tensor([self.robot.get_joint(n).dof_start for n in self.env_cfg['joint_names']],
                                           dtype=gs.tc_int, device=gs.device)
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self.robot.set_dofs_kp([self.env_cfg['kp']]*self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg['kd']]*self.num_actions, self.motors_dof_idx)

        f = dict(dtype=gs.tc_float, device=gs.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], **f)
        self.init_base_pos = torch.tensor(self.env_cfg['base_init_pos'], **f)
        self.init_base_quat = torch.tensor(self.env_cfg['base_init_quat'], **f)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor([self.env_cfg['default_joint_angles'][j.name] for j in self.robot.joints[1:]], **f)
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        n = self.num_envs
        self.base_lin_vel = torch.empty((n, 3), **f)
        self.base_ang_vel = torch.empty((n, 3), **f)
        self.projected_gravity = torch.empty((n, 3), **f)
        self.rew_buf = torch.empty((n,), **f)
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((n,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((n, self.num_commands), **f)
        self.commands_scale = torch.tensor([self.obs_scales['lin_vel'], self.obs_scales['lin_vel'], self.obs_scales['ang_vel']], **f)
        self.actions = torch.zeros((n, self.num_actions), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((n, 3), **f)
        self.base_quat = torch.empty((n, 4), **f)
        self.base_euler = torch.empty((n, 3), **f)
        self.default_dof_pos = torch.tensor([self.env_cfg['default_joint_angles'][j] for j in self.env_cfg['joint_names']], **f)
        self.extras = {}
        self.reward_functions = {name: getattr(self, '_reward_'+name) for name in self.reward_scales}
        self.episode_sums = {name: torch.zeros((n,), **f) for name in self.reward_scales}
        self.reset()

    # ---- commands -------------------------------------------------------------------------------------------------
    def _resample_commands(self, envs_idx):
        """Draw a primitive per environment (where `envs_idx` is True, or for all when None)."""
        commands = self.command_bank[torch.randint(0, self.command_bank.shape[0], (self.num_envs,), device=gs.device)].clone()
        if self.command_jitter_std > 0.0:
            commands += torch.randn_like(commands)*self.command_jitter_std
            commands = torch.max(torch.min(commands, self.command_high), self.command_low)
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    # ---- stepping -------------------------------------------------------------------------------------------------
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg['clip_actions'], self.env_cfg['clip_actions'])
        executed = self.last_actions if self.env_cfg['simulate_action_latency'] else self.actions
        target_dof_pos = executed*self.env_cfg['action_scale']+self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self.rew_buf.zero_()
        for name, reward in self.reward_functions.items():
            value = reward()*self.reward_scales[name]
            self.rew_buf += value
            self.episode_sums[name] += value

        self._resample_commands(self.episode_length_buf % int(self.env_cfg['resampling_time_s']/self.dt) == 0)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg['termination_if_pitch_greater_than']
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg['termination_if_roll_greater_than']
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras['time_outs'] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)
        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({'policy': self.obs_buf}, batch_size=[self.num_envs])

    def _update_observation(self):
        s = self.obs_scales
        self.obs_buf = torch.concatenate((
            self.base_ang_vel*s['ang_vel'], self.projected_gravity, self.commands*self.commands_scale,
            (self.dof_pos-self.default_dof_pos)*s['dof_pos'], self.dof_vel*s['dof_vel'], self.actions), dim=-1)

    # ---- resets ---------------------------------------------------------------------------------------------------
    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    def _reset_idx(self, envs_idx=None):
        """Reset all environments (None) or those where the boolean mask `envs_idx` is True."""
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            for buffer in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                           self.last_dof_vel, self.episode_length_buf):
                buffer.zero_()
            self.reset_buf.fill_(True)
        else:
            mask = envs_idx[:, None]
            torch.where(mask, self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(mask, self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(mask, self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(mask, self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            for buffer in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                           self.last_dof_vel):
                buffer.masked_fill_(mask, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        count = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras['episode'] = {}
        for key, value in self.episode_sums.items():
            mean = value.mean() if envs_idx is None else torch.where(count > 0, value[envs_idx].sum()/count, 0.0)
            self.extras['episode']['rew_'+key] = mean/self.env_cfg['episode_length_s']
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)
        self._resample_commands(envs_idx)

    # ---- rewards --------------------------------------------------------------------------------------------------
    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, :2]-self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error/self.reward_cfg['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2]-self.base_ang_vel[:, 2])
        return torch.exp(-error/self.reward_cfg['tracking_sigma'])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions-self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos-self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2]-self.reward_cfg['base_height_target'])
