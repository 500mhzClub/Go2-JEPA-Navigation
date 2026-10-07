"""The bulk rollout's recorded messages (raw/<scene>/messages.jsonl), in the original ROS 2 message layouts but
without ROS.

The original rollouts wrote a rosbag2 MCAP and converted it to JSON lines; this module writes the same JSON lines
directly. Each line is a record {topic, canonical_topic, type, timestamp_ns, env_index (absent for /clock), payload},
dumped with sorted keys and compact separators. A payload has every field of its message type; float32 fields are
rounded to float32, as the bag's serialisation rounded them, and float64 fields keep full precision.

Per environment, every 100-ms command tick records episode_info, joint_states, base_state, odom, imu/data, mode and
foot_contacts; a command block records command_block at its start and executed_command_block at its end; resets
record reset_event. A global /clock follows every tick.
"""
import json
from pathlib import Path

import numpy as np

TOPICS = {
    'command_block': ('/lewm/go2/command_block', 'lewm_go2_control/msg/CommandBlock'),
    'executed_command_block': ('/lewm/go2/executed_command_block', 'lewm_go2_control/msg/ExecutedCommandBlock'),
    'base_state': ('/lewm/go2/base_state', 'lewm_go2_control/msg/BaseState'),
    'foot_contacts': ('/lewm/go2/foot_contacts', 'lewm_go2_control/msg/FootContacts'),
    'reset_event': ('/lewm/go2/reset_event', 'lewm_go2_control/msg/ResetEvent'),
    'episode_info': ('/lewm/episode_info', 'lewm_go2_control/msg/EpisodeInfo'),
    'mode': ('/lewm/go2/mode', 'lewm_go2_control/msg/Go2ModeState'),
    'imu': ('/imu/data', 'sensor_msgs/msg/Imu'),
    'joint_states': ('/joint_states', 'sensor_msgs/msg/JointState'),
    'odom': ('/odom', 'nav_msgs/msg/Odometry'),
}
CLOCK = ('/clock', 'rosgraph_msgs/msg/Clock')


def f32(value):
    return float(np.float32(value))


def stamp(stamp_ns):
    return {'sec': int(stamp_ns//1_000_000_000), 'nanosec': int(stamp_ns % 1_000_000_000)}


def header(stamp_ns, frame_id=''):
    return {'frame_id': str(frame_id), 'stamp': stamp(stamp_ns)}


def vector(v):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}


def pose(xyz, quat_xyzw):
    return {'position': vector(xyz), 'orientation': {k: float(q) for k, q in zip('xyzw', quat_xyzw)}}


def command_block(record):
    return {'header': header(record.stamp_ns, record.frame_id), 'sequence_id': int(record.sequence_id),
            'block_size': int(record.block_size), 'command_dt_s': f32(record.command_dt_s),
            'primitive_name': str(record.primitive_name), 'vx_body_mps': [f32(v) for v in record.vx_body_mps],
            'vy_body_mps': [f32(v) for v in record.vy_body_mps],
            'yaw_rate_radps': [f32(v) for v in record.yaw_rate_radps], 'event_name': str(record.event_name),
            'event_allowed_in_training': bool(record.event_allowed_in_training),
            'command_source': str(record.command_source), 'route_target_id': int(record.route_target_id),
            'next_waypoint_id': int(record.next_waypoint_id)}


def executed_command_block(record):
    out = {'header': header(record.stamp_ns, record.frame_id), 'sequence_id': int(record.sequence_id),
           'block_size': int(record.block_size), 'command_dt_s': f32(record.command_dt_s),
           'primitive_name': str(record.primitive_name), 'clipped': bool(record.clipped),
           'safety_overridden': bool(record.safety_overridden), 'controller_mode': str(record.controller_mode),
           'backend_id': str(record.backend_id)}
    for kind in ('requested', 'executed'):
        for axis in ('vx_body_mps', 'vy_body_mps', 'yaw_rate_radps'):
            out[f'{kind}_{axis}'] = [f32(v) for v in getattr(record, f'{kind}_{axis}')]
    return out


def base_state(record):
    return {'header': header(record.stamp_ns, record.frame_id),
            'pose_world': pose(record.pose_world_xyz, record.pose_world_quat_xyzw),
            'twist_world': {'linear': vector(record.twist_world_linear), 'angular': vector(record.twist_world_angular)},
            'twist_body': {'linear': vector(record.twist_body_linear), 'angular': vector(record.twist_body_angular)},
            'quat_world_xyzw': [f32(q) for q in record.pose_world_quat_xyzw], 'roll_rad': f32(record.roll_rad),
            'pitch_rad': f32(record.pitch_rad), 'yaw_rad': f32(record.yaw_rad)}


def foot_contacts(record):
    out = {'header': header(record.stamp_ns, record.frame_id)}
    for foot in ('fl', 'fr', 'rl', 'rr'):
        out[f'{foot}_contact'] = bool(getattr(record, f'{foot}_contact'))
        out[f'{foot}_force_n'] = f32(getattr(record, f'{foot}_force_n'))
    return out


def reset_event(record):
    return {'header': header(record.stamp_ns), 'scene_id': int(record.scene_id), 'episode_id': int(record.episode_id),
            'reset_count': int(record.reset_count), 'reason': str(record.reason), 'success': bool(record.success),
            'spawn_pose_world': pose(record.spawn_pose_xyz, record.spawn_pose_quat_xyzw)}


def episode_info(record):
    return {'header': header(record.stamp_ns), 'scene_id': int(record.scene_id), 'episode_id': int(record.episode_id),
            'episode_step': int(record.episode_step), 'reset_count': int(record.reset_count),
            'scene_family': str(record.scene_family), 'split': str(record.split),
            'manifest_sha256': str(record.manifest_sha256)}


def mode_state(*, stamp_ns, backend_id, controller_mode, gait_name, moving, status_text, standing=False,
               recovering=False, fallen=False, safety_stop=False):
    return {'header': header(stamp_ns), 'backend_id': str(backend_id), 'controller_mode': str(controller_mode),
            'gait_name': str(gait_name), 'standing': bool(standing), 'moving': bool(moving),
            'recovering': bool(recovering), 'fallen': bool(fallen), 'safety_stop': bool(safety_stop),
            'status_text': str(status_text)}


def imu(*, quat_xyzw, angular_vel_body, linear_accel_body, stamp_ns, frame_id):
    return {'header': header(stamp_ns, frame_id), 'orientation': {k: float(q) for k, q in zip('xyzw', quat_xyzw)},
            'orientation_covariance': [0.]*9, 'angular_velocity': vector(angular_vel_body),
            'angular_velocity_covariance': [0.]*9, 'linear_acceleration': vector(linear_accel_body),
            'linear_acceleration_covariance': [0.]*9}


def joint_state(*, joint_names, positions, velocities, stamp_ns, frame_id=''):
    return {'header': header(stamp_ns, frame_id), 'name': [str(n) for n in joint_names],
            'position': [float(p) for p in positions], 'velocity': [float(v) for v in velocities], 'effort': []}


def odometry(record, child_frame_id='base_link'):
    return {'header': header(record.stamp_ns, record.frame_id), 'child_frame_id': child_frame_id,
            'pose': {'pose': pose(record.pose_world_xyz, record.pose_world_quat_xyzw), 'covariance': [0.]*36},
            'twist': {'twist': {'linear': vector(record.twist_body_linear), 'angular': vector(record.twist_body_angular)},
                      'covariance': [0.]*36}}


class MessageWriter:
    """Writes one scene's messages.jsonl; counts messages per topic for the summary."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('x')
        self.counts = {}
        self.first_ns = self.last_ns = None

    def _write(self, topic, canonical, type_name, stamp_ns, payload, env_index=None):
        record = {'topic': topic, 'canonical_topic': canonical, 'type': type_name, 'timestamp_ns': int(stamp_ns),
                  'payload': payload}
        if env_index is not None:
            record['env_index'] = env_index
        self.stream.write(json.dumps(record, sort_keys=True, separators=(',', ':'))+'\n')
        self.counts[topic] = self.counts.get(topic, 0)+1
        self.first_ns = int(stamp_ns) if self.first_ns is None else min(self.first_ns, int(stamp_ns))
        self.last_ns = int(stamp_ns) if self.last_ns is None else max(self.last_ns, int(stamp_ns))

    def write_env(self, env_index, key, payload, stamp_ns):
        canonical, type_name = TOPICS[key]
        self._write(f'/env_{env_index:02d}{canonical}', canonical, type_name, stamp_ns, payload, env_index)

    def write_clock(self, stamp_ns):
        self._write(CLOCK[0], CLOCK[0], CLOCK[1], stamp_ns, {'clock': stamp(stamp_ns)})

    def close(self):
        self.stream.close()
