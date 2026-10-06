"""The trained Go2 locomotion policy at run time: body-velocity command in, 12 joint-position targets out.

Missions call `LocomotionPolicy.act` once per 20-ms control step. The observation is built exactly as in training
(`env.py`), from the simulator's base orientation, angular velocity and joint state, and the policy's previous action.
Actions are applied one step late, as in training.

Joint orders: the policy uses the upstream order (FR, FL, RR, RL legs; hip, thigh, calf). The mission simulator applies
targets in its own order (all hips FL, FR, RL, RR, then thighs, then calves), i.e. Genesis DOF indices 6-17 sorted. The
two maps between them are fixed here and checked against the built robot.
"""
from pathlib import Path
import pickle

import numpy as np
import torch
import yaml
from tensordict import TensorDict

SIM_JOINT_ORDER = ('FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                   'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                   'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint')
SIM_DOF_INDICES = tuple(range(6, 18))
POLICY_DOF_INDICES = (7, 11, 15, 6, 10, 14, 9, 13, 17, 8, 12, 16)  # FR, FL, RR, RL legs (hip, thigh, calf)
OBSERVATION_SIZE = 45


def world_to_body(vectors_world, quat_xyzw):
    """Rotate (..., 3) world-frame vectors into the body frame (float64 maths, float32 result)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    q[..., :3] *= -1.0
    x, y, z, w = (q[..., i:i+1] for i in range(4))
    v = np.asarray(vectors_world, dtype=np.float64)
    vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    rows = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    return np.concatenate([r0*vx+r1*vy+r2*vz for r0, r1, r2 in rows], axis=-1).astype(np.float32)


def load_config(checkpoint):
    """The policy's training config: config.yaml beside a go2nav checkpoint, or cfgs.pkl beside an original one."""
    folder = Path(checkpoint).parent
    if (folder/'config.yaml').exists():
        return yaml.safe_load((folder/'config.yaml').read_text())
    env, obs, reward, _command, train = pickle.loads((folder/'cfgs.pkl').read_bytes())
    return dict(env=env, obs=obs, reward=reward, train=train)


def build_actor(checkpoint, cfg, device):
    """Reconstruct the trained actor through rsl-rl (the runner rebuilds the model from the training config)."""
    from rsl_rl.runners import OnPolicyRunner

    class SpecOnlyEnv:  # the runner reads only these attributes to build the networks
        num_envs, max_episode_length = 1, 1000

        def __init__(self):
            self.num_actions, self.device, self.cfg = int(cfg['env']['num_actions']), device, cfg['env']
            self.episode_length_buf = torch.zeros((1,), dtype=torch.int32, device=device)

        def get_observations(self):
            return TensorDict({'policy': torch.zeros((1, OBSERVATION_SIZE), device=device)}, batch_size=[1])

    runner = OnPolicyRunner(SpecOnlyEnv(), dict(cfg['train']), log_dir=None, device=device)
    runner.load(str(checkpoint), map_location=device)
    actor = runner.get_inference_policy(device=device)
    actor.eval()
    return actor


class LocomotionPolicy:
    def __init__(self, checkpoint, device=None):
        cfg = load_config(checkpoint)
        env, scales = cfg['env'], cfg['obs']['obs_scales']
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = build_actor(checkpoint, cfg, self.device)
        self.ang_vel_scale, self.dof_pos_scale, self.dof_vel_scale = (float(scales[k]) for k in ('ang_vel', 'dof_pos', 'dof_vel'))
        self.command_scale = np.array([scales['lin_vel'], scales['lin_vel'], scales['ang_vel']], dtype=np.float32)
        self.action_scale = float(env['action_scale'])
        self.latency = bool(env['simulate_action_latency'])
        self.joint_names = tuple(env['joint_names'])
        self.default_pose = np.array([env['default_joint_angles'][n] for n in self.joint_names], dtype=np.float32)
        self.sim_from_policy = np.array([POLICY_DOF_INDICES.index(i) for i in SIM_DOF_INDICES])
        self.policy_from_sim = np.array([SIM_DOF_INDICES.index(i) for i in POLICY_DOF_INDICES])
        self.stance = self.default_pose[self.sim_from_policy].copy()  # reset pose in simulator order
        self.last_actions = None

    def check_robot(self, robot):
        """The built robot's DOF indices must match the fixed policy joint map."""
        joints = {j.name: j for j in robot.joints}
        actual = tuple(int(np.asarray(joints[n].dofs_idx_local).reshape(-1)[0]) for n in self.joint_names)
        if actual != POLICY_DOF_INDICES:
            raise ValueError(f'robot joint DOFs {actual} differ from the policy map {POLICY_DOF_INDICES}')

    def reset(self, envs=None):
        """Clear the action history (all environments, or the listed ones)."""
        if self.last_actions is not None:
            if envs is None:
                self.last_actions.fill(0.0)
            else:
                self.last_actions[np.asarray(envs, dtype=np.int64)] = 0.0

    def observation(self, base_quat_xyzw, base_ang_vel_world, joint_pos, joint_vel, command):
        """The 45-value policy observation (all inputs batched; joints in simulator order)."""
        gravity = np.zeros((len(command), 3), dtype=np.float32)
        gravity[:, 2] = -1.0
        return np.concatenate((
            world_to_body(np.asarray(base_ang_vel_world, np.float32), base_quat_xyzw)*self.ang_vel_scale,
            world_to_body(gravity, base_quat_xyzw),
            np.asarray(command, np.float32)*self.command_scale,
            (np.asarray(joint_pos, np.float32)[:, self.policy_from_sim]-self.default_pose)*self.dof_pos_scale,
            np.asarray(joint_vel, np.float32)[:, self.policy_from_sim]*self.dof_vel_scale,
            self.last_actions), axis=-1).astype(np.float32, copy=False)

    def act(self, base_quat_xyzw, base_ang_vel_world, joint_pos, joint_vel, command):
        """Joint-position targets (simulator order) for one control step; `command` is [vx, vy, yaw rate] per robot."""
        n = len(command)
        if self.last_actions is None or self.last_actions.shape[0] != n:
            self.last_actions = np.zeros((n, len(self.joint_names)), dtype=np.float32)
        obs = self.observation(base_quat_xyzw, base_ang_vel_world, joint_pos, joint_vel, command)
        with torch.no_grad():
            batch = TensorDict({'policy': torch.as_tensor(obs, dtype=torch.float32, device=self.device)}, batch_size=[n])
            actions = self.actor(batch, stochastic_output=False).detach().cpu().numpy().astype(np.float32, copy=False)
        executed = self.last_actions if self.latency else actions
        targets = executed*self.action_scale+self.default_pose
        self.last_actions = actions.copy()
        return targets[:, self.sim_from_policy].astype(np.float32, copy=False)
