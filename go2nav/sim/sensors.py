"""Simulated sensing: ideal body sensors, causal histories, and the packets the controller receives.

Body sensors are ideal (no bias, noise, lever arm or delay):
- **body** (50 Hz): angular velocity and specific force in the body frame (specific force from the backward
  velocity difference over 20 ms; invalid at the first sample) and the 12 joint positions and velocities;
- **gyro** (500 Hz): angular velocity in the body frame, every physics step;
- **applied command** (10 Hz): the command the locomotion layer executed, at every 100-ms boundary.

A **history** is the last N samples measured within a maximum age before the decision time, left-padded with
invalid zeros: body channels 20 samples within 400 ms, applied commands 15 within 1.5 s, the fast gyro 51 within 100 ms.

At each 10 Hz frame the controller gets:
- the **policy packet**: the primary RGB image and the histories at the frame time (`policy_packet`);
- the primary and auxiliary **depth** (`depth_packet`): optical-axis depth, valid from 0.2 to 5 m, invalid pixels 0,
  perturbed with seeded Gaussian noise (sigma 2 mm; `add_depth_noise`);
- the auxiliary **RGB** and the **fast gyro** history.
Packets keep the original's schema names, calibration identifiers and digests, so recorded missions compare directly.
"""
from collections import deque
from dataclasses import dataclass
import hashlib

import numpy as np

IDENTITY = (0, 0, 0)
JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR'))
BODY_CALIBRATION = 'ideal-body-origin-50hz-v1'
CAMERA_CALIBRATION = 'go2-fixed-native-rgb-640x480-development-v1'
GYRO_CALIBRATION = 'ideal-body-gyro-500hz-development-v1'
GYRO_SCHEMA_ID = 'causal_fast_body_gyro_development.v1'
DEPTH_SCHEMA = 'causal_colocated_optical_depth_development.v1'
DEPTH_CALIBRATION = 'go2_colocated_640x480_hfov78.323_optical_depth_0.2_to_5m_v1'
AUXILIARY_DEPTH_SCHEMA = 'causal_auxiliary_tilted_optical_depth_development.v1'
AUXILIARY_DEPTH_CALIBRATION = 'go2_auxiliary_640x480_pitch45_mount035_000_008_depth02to5_v1'
AUXILIARY_RGB_SCHEMA = 'causal_auxiliary_downward45_rgb_development.v1'
AUXILIARY_RGB_CALIBRATION = 'go2_auxiliary_640x480_pitch45_mount035_000_008_rgb_v1'
MIN_DEPTH_M, MAX_DEPTH_M = .2, 5.
NOISE_SEED = 2026091414
GRAVITY = np.array([0., 0., -9.81])


@dataclass(frozen=True)
class Channel:
    name: str
    channels: tuple
    units: tuple
    history_length: int
    max_age_ns: int
    calibration_id: str
    role: str = 'sensed'


BODY_CHANNELS = (
    Channel('gyro', ('wx', 'wy', 'wz'), ('rad/s',)*3, 20, 400_000_000, BODY_CALIBRATION),
    Channel('specific_force', ('fx', 'fy', 'fz'), ('m/s^2',)*3, 20, 400_000_000, BODY_CALIBRATION),
    Channel('joints', tuple('q:'+n for n in JOINT_NAMES)+tuple('dq:'+n for n in JOINT_NAMES), ('rad',)*12+('rad/s',)*12,
            20, 400_000_000, BODY_CALIBRATION),
    Channel('applied_command', ('vx', 'vy', 'wz'), ('m/s', 'm/s', 'rad/s'), 15, 1_500_000_000,
            'applied-command-clock-v1', role='control'),
)
FAST_GYRO = Channel('gyro', ('wx', 'wy', 'wz'), ('rad/s',)*3, 51, 100_000_000, GYRO_CALIBRATION)


def rotation_xyzw(quaternion):
    """Rotation matrix of an xyzw quaternion (normalised)."""
    q = np.asarray(quaternion, dtype=float)
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class History:
    """Timestamped samples of some channels (newest last, at most 256 each) and their causal histories."""

    def __init__(self, channels):
        self.channels = {c.name: c for c in channels}
        self.samples = {c.name: deque(maxlen=256) for c in channels}

    def append(self, name, values, valid, measured_ns):
        data, mask = np.asarray(values, dtype=np.float64), np.asarray(valid)
        samples = self.samples[name]
        if samples and measured_ns <= samples[-1][0]:
            raise ValueError('sensor samples must arrive in time order')
        samples.append((int(measured_ns), int(measured_ns), np.where(mask, data, 0.).copy(), mask.copy()))

    def channel(self, name, decision_ns):
        """One channel's history at a decision time."""
        c = self.channels[name]
        shape = (c.history_length, len(c.channels))
        values, valid = np.zeros(shape), np.zeros(shape, dtype=bool)
        measured = np.full(c.history_length, -1, dtype=np.int64)
        available = np.full(c.history_length, -1, dtype=np.int64)
        eligible = [s for s in self.samples[name]
                    if s[0] <= decision_ns and s[1] <= decision_ns and decision_ns-s[0] <= c.max_age_ns][-c.history_length:]
        for i, (mt, at, data, mask) in enumerate(eligible, start=c.history_length-len(eligible)):
            measured[i], available[i], values[i], valid[i] = mt, at, data, mask
        return {'values': values, 'valid': valid, 'measured_ns': measured, 'available_ns': available,
                'channels': c.channels, 'units': c.units, 'calibration_id': c.calibration_id}

    def state(self, decision_ns):
        result = {'identity': IDENTITY, 'image_ns': decision_ns, 'decision_ns': decision_ns, 'sensor_anchor': 'decision',
                  'sensor_anchor_ns': decision_ns, 'sensed': {}, 'control': {}}
        for name, c in self.channels.items():
            result[c.role][name] = self.channel(name, decision_ns)
        return result


class BodySensor:
    """The ideal 50 Hz body sensor (see the module docstring)."""

    def __init__(self):
        self.previous_ns = self.previous_velocity = None

    def sample(self, measured_ns, quaternion_xyzw, velocity_world, angular_velocity_world, joint_position, joint_velocity):
        if self.previous_ns is not None and measured_ns-self.previous_ns != 20_000_000:
            raise ValueError('the body sensor needs an exact 20-ms cadence')
        rotation = rotation_xyzw(quaternion_xyzw)
        velocity, angular, q, dq = (np.asarray(v, dtype=float) for v in
                                    (velocity_world, angular_velocity_world, joint_position, joint_velocity))
        have_force = self.previous_ns is not None
        force = rotation.T@((velocity-self.previous_velocity)/.02-GRAVITY) if have_force else np.zeros(3)
        self.previous_ns, self.previous_velocity = measured_ns, velocity.copy()
        return {'gyro': (rotation.T@angular, np.ones(3, dtype=bool)),
                'specific_force': (force, np.full(3, have_force, dtype=bool)),
                'joints': (np.concatenate([q, dq]), np.ones(24, dtype=bool))}


def gyro_sample(quaternion_xyzw, angular_velocity_world):
    """The ideal 500 Hz gyro: body-frame angular velocity."""
    return rotation_xyzw(quaternion_xyzw).T@np.asarray(angular_velocity_world, dtype=float), np.ones(3, dtype=bool)


def rgb_digest(policy):
    return hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()


def depth_digest(depth):
    return hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()


def policy_packet(history, rgb, now_ns):
    """The policy observation at a frame: the primary RGB and the body histories."""
    return {'image': {'rgb': np.array(rgb, copy=True, order='C'), 'measured_ns': now_ns, 'available_ns': now_ns,
                      'calibration_id': CAMERA_CALIBRATION},
            'sensor_state': history.state(now_ns)}


def fast_gyro_packet(history, now_ns):
    return {'schema': GYRO_SCHEMA_ID, 'identity': IDENTITY, 'decision_ns': now_ns, **history.channel('gyro', now_ns)}


def depth_packet(native, policy, now_ns, *, auxiliary=False):
    """A camera's depth packet from the renderer's float32 depth (out-of-range or non-finite pixels: invalid, 0)."""
    native = np.asarray(native)
    if native.shape != (480, 640) or native.dtype != np.float32:
        raise ValueError('640 x 480 float32 depth required')
    valid = np.isfinite(native)&(native >= MIN_DEPTH_M)&(native <= MAX_DEPTH_M)
    packet = dict(schema=AUXILIARY_DEPTH_SCHEMA if auxiliary else DEPTH_SCHEMA,
                  calibration_id=AUXILIARY_DEPTH_CALIBRATION if auxiliary else DEPTH_CALIBRATION,
                  identity=IDENTITY, measured_ns=now_ns, available_ns=now_ns, decision_ns=now_ns)
    packet['primary_rgb_sha256' if auxiliary else 'rgb_sha256'] = rgb_digest(policy)
    packet.update(depth_m=np.where(valid, native, np.float32(0.)).copy(), valid=valid.copy(),
                  representation='optical_axis_depth_m', hardware_calibrated=False)
    return packet


def auxiliary_rgb_packet(rgb, auxiliary_depth, policy, now_ns):
    return dict(schema=AUXILIARY_RGB_SCHEMA, calibration_id=AUXILIARY_RGB_CALIBRATION,
                depth_calibration_id=AUXILIARY_DEPTH_CALIBRATION, identity=IDENTITY, measured_ns=now_ns,
                available_ns=now_ns, decision_ns=now_ns, primary_rgb_sha256=rgb_digest(policy),
                auxiliary_depth_sha256=depth_digest(auxiliary_depth), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                rgb=rgb.copy(), hardware_calibrated=False)


def add_depth_noise(depth, *, layout, frame, camera, sigma_m):
    """Seeded Gaussian noise on the valid depth (seed: NOISE_SEED, maze layout mod 4, frame, camera 0/1); pixels
    pushed outside 0.2-5 m become invalid."""
    rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, layout, frame, camera]))
    values, valid = depth['depth_m'].copy(), depth['valid'].copy()
    values[valid] += rng.normal(0., sigma_m, int(valid.sum())).astype(np.float32)
    valid &= (values >= MIN_DEPTH_M)&(values <= MAX_DEPTH_M)
    values[~valid] = 0.
    return depth | dict(depth_m=values, valid=valid)
