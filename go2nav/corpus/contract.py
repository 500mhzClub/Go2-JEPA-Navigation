"""The bulk rollout's command and state contract: safety limits, the primitive registry, command-block expansion
and limiting, the recorded record types, and per-environment episode bookkeeping.

The record dataclasses mirror the recorded message types field for field (`go2nav.corpus.messages` writes them).
Functions are vectorised over environments: numpy arrays with a leading env dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LEWM_FOOT_ORDER: tuple[str, ...] = ("fl", "fr", "rl", "rr")
"""LeWM ``FootContacts`` leg ordering: front-left, front-right, rear-left, rear-right."""


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyLimits:
    """Per-tick absolute and delta limits from the platform manifest."""

    min_vx_mps: float
    max_vx_mps: float
    min_vy_mps: float
    max_vy_mps: float
    max_yaw_rate_radps: float
    max_delta_vx_mps: float
    max_delta_vy_mps: float
    max_delta_yaw_rate_radps: float

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "SafetyLimits":
        safety = manifest.get("locomotion", {}).get("safety", {})
        delta = safety.get("max_command_delta_per_tick", {})
        return cls(
            min_vx_mps=float(safety.get("min_vx_mps", -np.inf)),
            max_vx_mps=float(safety.get("max_vx_mps", np.inf)),
            min_vy_mps=float(safety.get("min_vy_mps", -np.inf)),
            max_vy_mps=float(safety.get("max_vy_mps", np.inf)),
            max_yaw_rate_radps=float(safety.get("max_yaw_rate_radps", np.inf)),
            max_delta_vx_mps=float(delta.get("vx_mps", np.inf)),
            max_delta_vy_mps=float(delta.get("vy_mps", np.inf)),
            max_delta_yaw_rate_radps=float(delta.get("yaw_rate_radps", np.inf)),
        )


@dataclass(frozen=True)
class PrimitiveRegistry:
    """Loaded view of ``configs/corpus/go2_primitive_registry.yaml``."""

    block_size: int
    command_dt_s: float
    command_order: tuple[str, ...]
    primitives: dict[str, dict[str, Any]]
    defaults: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrimitiveRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            block_size=int(data.get("block_size", 5)),
            command_dt_s=float(data.get("command_dt_s", 0.10)),
            command_order=tuple(data.get("command_order", ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"))),
            primitives=dict(data.get("primitives", {})),
            defaults=dict(data.get("defaults", {})),
        )

    def trainable_velocity_names(self) -> list[str]:
        """Names of velocity-block primitives marked trainable."""

        names: list[str] = []
        default_train = bool(self.defaults.get("train", True))
        for name, spec in self.primitives.items():
            if spec.get("type") != "velocity_block":
                continue
            if not bool(spec.get("train", default_train)):
                continue
            names.append(name)
        return names

    def get(self, name: str) -> dict[str, Any]:
        if name not in self.primitives:
            raise KeyError(f"unknown primitive '{name}'")
        return self.primitives[name]




# ---------------------------------------------------------------------------
# Records (mirror the recorded message fields exactly)
# ---------------------------------------------------------------------------


@dataclass
class CommandBlockRecord:
    """Mirrors ``lewm_go2_control/msg/CommandBlock``."""

    sequence_id: int
    block_size: int
    command_dt_s: float
    primitive_name: str
    vx_body_mps: list[float]
    vy_body_mps: list[float]
    yaw_rate_radps: list[float]
    event_name: str = ""
    event_allowed_in_training: bool = False
    # Data spec §5.1 command_source tag. Defaults match the legacy random
    # sampler so existing tests keep passing without modification.
    command_source: str = "primitive_curriculum"
    route_target_id: int = -1
    next_waypoint_id: int = -1
    stamp_ns: int = 0
    frame_id: str = "base_link"


@dataclass
class ExecutedCommandBlockRecord:
    """Mirrors ``lewm_go2_control/msg/ExecutedCommandBlock``."""

    sequence_id: int
    block_size: int
    command_dt_s: float
    primitive_name: str
    requested_vx_body_mps: list[float]
    requested_vy_body_mps: list[float]
    requested_yaw_rate_radps: list[float]
    executed_vx_body_mps: list[float]
    executed_vy_body_mps: list[float]
    executed_yaw_rate_radps: list[float]
    clipped: bool
    safety_overridden: bool
    controller_mode: str
    backend_id: str = "genesis_tier_a"
    stamp_ns: int = 0
    frame_id: str = "base_link"


@dataclass
class BaseStateRecord:
    """Mirrors ``lewm_go2_control/msg/BaseState``."""

    pose_world_xyz: tuple[float, float, float]
    pose_world_quat_xyzw: tuple[float, float, float, float]
    twist_body_linear: tuple[float, float, float]
    twist_body_angular: tuple[float, float, float]
    twist_world_linear: tuple[float, float, float]
    twist_world_angular: tuple[float, float, float]
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    stamp_ns: int = 0
    frame_id: str = "odom"


@dataclass
class FootContactsRecord:
    """Mirrors ``lewm_go2_control/msg/FootContacts`` in LeWM fl/fr/rl/rr order."""

    fl_contact: bool
    fr_contact: bool
    rl_contact: bool
    rr_contact: bool
    fl_force_n: float = 0.0
    fr_force_n: float = 0.0
    rl_force_n: float = 0.0
    rr_force_n: float = 0.0
    stamp_ns: int = 0
    frame_id: str = "base_link"


@dataclass
class ResetEventRecord:
    """Mirrors ``lewm_go2_control/msg/ResetEvent``."""

    scene_id: int
    episode_id: int
    reset_count: int
    reason: str
    success: bool
    spawn_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_pose_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stamp_ns: int = 0


@dataclass
class EpisodeInfoRecord:
    """Mirrors ``lewm_go2_control/msg/EpisodeInfo``."""

    scene_id: int
    episode_id: int
    episode_step: int
    reset_count: int
    scene_family: str = ""
    split: str = ""
    manifest_sha256: str = ""
    stamp_ns: int = 0


# ---------------------------------------------------------------------------
# Command-block expansion (vectorized over envs and over a single block)
# ---------------------------------------------------------------------------


def expand_primitive_to_block(
    registry: PrimitiveRegistry,
    primitive_name: str,
    block_size: int | None = None,
) -> np.ndarray:
    """Return ``(T, 3)`` requested (vx, vy, yaw_rate) for a named velocity primitive.

    Raises ``KeyError`` if the primitive is missing and ``ValueError`` if it
    is not a velocity block.
    """

    primitive = registry.get(primitive_name)
    if primitive.get("type") != "velocity_block":
        raise ValueError(
            f"primitive '{primitive_name}' is not a velocity_block "
            f"(type={primitive.get('type')!r})"
        )
    command = primitive.get("command", {})
    T = int(block_size if block_size is not None else registry.block_size)
    block = np.full(
        (T, 3),
        [
            float(command.get("vx_body_mps", 0.0)),
            float(command.get("vy_body_mps", 0.0)),
            float(command.get("yaw_rate_radps", 0.0)),
        ],
        dtype=np.float32,
    )
    return block




# ---------------------------------------------------------------------------
# Safety clipping
# ---------------------------------------------------------------------------


def apply_safety_limits_batch(
    requested: np.ndarray,
    previous: np.ndarray,
    limits: SafetyLimits,
    enforce_rate_limits: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized safety clipping.

    Parameters
    ----------
    requested : np.ndarray
        Shape ``(n_envs, T, 3)`` requested (vx, vy, yaw_rate) per tick.
    previous : np.ndarray
        Shape ``(n_envs, 3)`` last executed command per env (used to seed
        the per-tick delta limiter).
    limits : SafetyLimits
        Absolute and per-tick delta bounds.
    enforce_rate_limits : bool
        Mirror the ``command_block_adapter`` parameter.

    Returns
    -------
    executed : np.ndarray
        Shape ``(n_envs, T, 3)``. Clipped commands.
    clipped_any : np.ndarray
        Shape ``(n_envs,)`` bool. ``True`` if any tick was clipped in that env.
    """

    if requested.ndim != 3 or requested.shape[-1] != 3:
        raise ValueError(f"requested must be (n_envs, T, 3); got {requested.shape}")
    if previous.shape != (requested.shape[0], 3):
        raise ValueError(
            f"previous must be (n_envs, 3); got {previous.shape} for "
            f"n_envs={requested.shape[0]}"
        )

    abs_lo = np.array(
        [limits.min_vx_mps, limits.min_vy_mps, -limits.max_yaw_rate_radps],
        dtype=np.float32,
    )
    abs_hi = np.array(
        [limits.max_vx_mps, limits.max_vy_mps, limits.max_yaw_rate_radps],
        dtype=np.float32,
    )
    delta = np.array(
        [
            limits.max_delta_vx_mps,
            limits.max_delta_vy_mps,
            limits.max_delta_yaw_rate_radps,
        ],
        dtype=np.float32,
    )

    n_envs, T, _ = requested.shape
    executed = np.empty_like(requested, dtype=np.float32)
    clipped_any = np.zeros((n_envs,), dtype=bool)

    prev = previous.astype(np.float32, copy=True)
    for t in range(T):
        step_requested = requested[:, t, :]
        bounded = np.clip(step_requested, abs_lo, abs_hi)
        clipped_step = np.any(bounded != step_requested, axis=-1)

        if enforce_rate_limits:
            lower = prev - delta
            upper = prev + delta
            after_rate = np.clip(bounded, lower, upper)
            clipped_step |= np.any(after_rate != bounded, axis=-1)
            bounded = after_rate

        executed[:, t, :] = bounded
        clipped_any |= clipped_step
        prev = bounded
    return executed, clipped_any




def reconstruct_executed_block(
    requested: list[tuple[float, float, float]],
    executed: list[tuple[float, float, float]],
    *,
    sequence_id: int,
    primitive_name: str,
    command_dt_s: float,
    clipped: bool,
    safety_overridden: bool = False,
    controller_mode: str = "cmd_vel",
    backend_id: str = "genesis_tier_a",
    stamp_ns: int = 0,
) -> ExecutedCommandBlockRecord:
    """Bundle requested+executed pairs into an ``ExecutedCommandBlockRecord``."""

    if len(requested) != len(executed):
        raise ValueError(
            f"requested ({len(requested)}) and executed ({len(executed)}) "
            "must be the same length"
        )
    return ExecutedCommandBlockRecord(
        sequence_id=int(sequence_id),
        block_size=len(executed),
        command_dt_s=float(command_dt_s),
        primitive_name=str(primitive_name),
        requested_vx_body_mps=[float(c[0]) for c in requested],
        requested_vy_body_mps=[float(c[1]) for c in requested],
        requested_yaw_rate_radps=[float(c[2]) for c in requested],
        executed_vx_body_mps=[float(c[0]) for c in executed],
        executed_vy_body_mps=[float(c[1]) for c in executed],
        executed_yaw_rate_radps=[float(c[2]) for c in executed],
        clipped=bool(clipped),
        safety_overridden=bool(safety_overridden),
        controller_mode=str(controller_mode),
        backend_id=str(backend_id),
        stamp_ns=int(stamp_ns),
    )


# ---------------------------------------------------------------------------
# Base-state computation (quat math, body↔world rotation)
# ---------------------------------------------------------------------------


def quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` xyzw quaternions to ``(..., 3)`` rpy in radians.

    Mirrors ``base_state_publisher._quat_to_rpy``: pitch is asin-clamped to
    avoid NaN at the gimbal singularities.
    """

    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat last dim must be 4 (xyzw); got {q.shape}")
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def rotate_body_to_world(vec_body: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate a ``(..., 3)`` body-frame vector into the world frame.

    Mirrors ``base_state_publisher._rotate_body_to_world`` and broadcasts.
    """

    v = np.asarray(vec_body, dtype=np.float64)
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"vec_body last dim must be 3; got {v.shape}")
    if q.shape[-1] != 4:
        raise ValueError(f"quat last dim must be 4 (xyzw); got {q.shape}")

    qx, qy, qz, qw = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    vx = v[..., 0:1]
    vy = v[..., 1:2]
    vz = v[..., 2:3]

    x = r00 * vx + r01 * vy + r02 * vz
    y = r10 * vx + r11 * vy + r12 * vz
    z = r20 * vx + r21 * vy + r22 * vz
    return np.concatenate([x, y, z], axis=-1).astype(np.float32)


def base_state_from_genesis(
    pos_world: np.ndarray,
    quat_xyzw: np.ndarray,
    linear_vel_body: np.ndarray,
    angular_vel_body: np.ndarray,
    *,
    stamp_ns: int = 0,
    frame_id: str = "odom",
) -> BaseStateRecord:
    """Build a single-env ``BaseStateRecord`` from Genesis state primitives."""

    rpy = quat_to_rpy(np.asarray(quat_xyzw))
    lin_world = rotate_body_to_world(linear_vel_body, quat_xyzw)
    ang_world = rotate_body_to_world(angular_vel_body, quat_xyzw)
    qx, qy, qz, qw = (float(quat_xyzw[i]) for i in range(4))
    return BaseStateRecord(
        pose_world_xyz=(float(pos_world[0]), float(pos_world[1]), float(pos_world[2])),
        pose_world_quat_xyzw=(qx, qy, qz, qw),
        twist_body_linear=tuple(float(v) for v in linear_vel_body),  # type: ignore[arg-type]
        twist_body_angular=tuple(float(v) for v in angular_vel_body),  # type: ignore[arg-type]
        twist_world_linear=tuple(float(v) for v in lin_world),  # type: ignore[arg-type]
        twist_world_angular=tuple(float(v) for v in ang_world),  # type: ignore[arg-type]
        roll_rad=float(rpy[0]),
        pitch_rad=float(rpy[1]),
        yaw_rad=float(rpy[2]),
        stamp_ns=int(stamp_ns),
        frame_id=str(frame_id),
    )


# ---------------------------------------------------------------------------
# Foot contacts
# ---------------------------------------------------------------------------


def foot_contacts_record(
    contact_bools_in_lewm_order: tuple[bool, bool, bool, bool],
    *,
    force_n_in_lewm_order: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stamp_ns: int = 0,
    frame_id: str = "base_link",
) -> FootContactsRecord:
    """Build a ``FootContactsRecord`` from booleans already in LeWM (fl/fr/rl/rr) order.

    The Genesis adapter is responsible for mapping the simulator's per-foot
    index to LeWM order before calling this. The mapping is platform-specific
    (depends on URDF link naming); ``LEWM_FOOT_ORDER`` is the LeWM canonical
    order and must not be reordered.
    """

    fl, fr, rl, rr = contact_bools_in_lewm_order
    f_fl, f_fr, f_rl, f_rr = force_n_in_lewm_order
    return FootContactsRecord(
        fl_contact=bool(fl),
        fr_contact=bool(fr),
        rl_contact=bool(rl),
        rr_contact=bool(rr),
        fl_force_n=float(f_fl),
        fr_force_n=float(f_fr),
        rl_force_n=float(f_rl),
        rr_force_n=float(f_rr),
        stamp_ns=int(stamp_ns),
        frame_id=str(frame_id),
    )


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class EpisodeState:
    """Per-env episode counters.

    Mirrors ``reset_manager`` semantics: monotonic ``episode_id`` and
    ``reset_count``, ``episode_step`` zeroed on reset and incremented per
    control tick. ``scene_id``, ``scene_family``, ``split``, and
    ``manifest_sha256`` are carried from the scene manifest.
    """

    scene_id: int = 0
    episode_id: int = 0
    reset_count: int = 0
    episode_step: int = 0
    scene_family: str = ""
    split: str = ""
    manifest_sha256: str = ""

    def step(self) -> None:
        """Advance ``episode_step`` by one control tick."""

        self.episode_step += 1

    def reset(
        self,
        *,
        scene_id: int | None = None,
        reason: str = "",
        success: bool = True,
        spawn_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spawn_pose_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        stamp_ns: int = 0,
    ) -> ResetEventRecord:
        """Advance episode counters and return the matching ``ResetEventRecord``."""

        self.episode_id += 1
        self.reset_count += 1
        self.episode_step = 0
        if scene_id is not None:
            self.scene_id = int(scene_id)
        return ResetEventRecord(
            scene_id=int(self.scene_id),
            episode_id=int(self.episode_id),
            reset_count=int(self.reset_count),
            reason=str(reason),
            success=bool(success),
            spawn_pose_xyz=spawn_pose_xyz,
            spawn_pose_quat_xyzw=spawn_pose_quat_xyzw,
            stamp_ns=int(stamp_ns),
        )

    def episode_info(self, *, stamp_ns: int = 0) -> EpisodeInfoRecord:
        return EpisodeInfoRecord(
            scene_id=int(self.scene_id),
            episode_id=int(self.episode_id),
            episode_step=int(self.episode_step),
            reset_count=int(self.reset_count),
            scene_family=str(self.scene_family),
            split=str(self.split),
            manifest_sha256=str(self.manifest_sha256),
            stamp_ns=int(stamp_ns),
        )


def make_episode_states(
    n_envs: int,
    *,
    scene_id: int = 0,
    scene_family: str = "",
    split: str = "",
    manifest_sha256: str = "",
) -> list[EpisodeState]:
    """Construct ``n_envs`` independent ``EpisodeState`` instances."""

    return [
        EpisodeState(
            scene_id=int(scene_id),
            scene_family=str(scene_family),
            split=str(split),
            manifest_sha256=str(manifest_sha256),
        )
        for _ in range(n_envs)
    ]
