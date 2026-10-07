"""Privileged per-step labels of a bulk rollout (place, heading, clearance, landmark distances and visibility).

    python -m go2nav.corpus.labels ROLLOUT_SCENE_DIR CORPUS OUT_DIR

Reads ROLLOUT_SCENE_DIR/messages.jsonl (`go2nav.corpus.bulk`) and the scene's manifest from CORPUS; writes
OUT_DIR/labels.jsonl (one label per environment and command tick) and OUT_DIR/summary.json.

The data-spec lists labels that depend on simulator-privileged state but are
not deployment inputs: ``cell_id``, ``yaw_bin``, ``local_graph_type``,
``nearest_cell_distance``, ``clearance``, ``bfs_distance_to_landmark`` per
landmark, ``landmark_visible/identity/bearing/range``, and
``integrated_body_motion`` over an action block and over H-JEPA history
windows.

This module computes them deterministically from:

- a :class:`SceneManifest` (regenerable from ``topology_seed`` — see
  ``go2nav.corpus.graph.SceneGraph``),
- a stream of per-step :class:`PoseStep` records produced by any rollout
  source (a bulk rollout's ``messages.jsonl``, or test fixtures),

and emits a stream of :class:`DerivedLabelStep` records. The computer is
pure Python so it can run inside an offline post-processor without any
simulator runtime.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from go2nav.corpus.topology import (
    local_graph_type_per_node,
)
from go2nav.corpus.manifest import SceneManifest
from go2nav.corpus.graph import SceneGraph, wrap_angle_pi


# Default discretization for the per-step yaw bin (data spec §5.3 yaw_bin).
DEFAULT_YAW_BINS: int = 8

BASE_STATE_TOPIC = "/lewm/go2/base_state"
COMMAND_BLOCK_TOPIC = "/lewm/go2/command_block"
EXECUTED_COMMAND_BLOCK_TOPIC = "/lewm/go2/executed_command_block"
EPISODE_INFO_TOPIC = "/lewm/episode_info"
ENV_TOPIC_RE = re.compile(r"^/env_(\d+)(/.*)$")


@dataclass(frozen=True)
class PoseStep:
    """One step of per-env state at command-tick cadence.

    All fields except ``last_command`` are required. ``last_command`` is the
    executed body-frame ``(vx, vy, yaw_rate)`` for the *previous* command
    tick, used to compute the ``stuck_label``; supply zeros if no command
    history is available.
    """

    timestamp_ns: int
    env_idx: int
    episode_id: int
    episode_step: int
    position_xy_world: tuple[float, float]
    yaw_world_rad: float
    last_command: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MessagePoseJoinSummary:
    """Audit summary for joining logged base poses to episode/command state."""

    source_record_count: int
    base_state_count: int
    command_block_count: int
    executed_command_block_count: int
    episode_info_count: int
    pose_step_count: int
    env_indices: tuple[int, ...]
    missing_episode_info_count: int
    missing_command_count: int


@dataclass(frozen=True)
class LandmarkObservation:
    object_id: str
    visible: bool
    bearing_body_rad: float
    range_m: float
    bfs_distance_cells: int | None


@dataclass(frozen=True)
class DerivedLabelStep:
    """One step of derived labels, one-to-one with the input :class:`PoseStep`."""

    timestamp_ns: int
    env_idx: int
    episode_id: int
    episode_step: int
    scene_id: str
    # Scene-scoped graph location
    cell_id: int
    nearest_cell_distance_m: float
    local_graph_type: str
    # Heading
    yaw_bin: int
    yaw_bin_count: int
    # Safety
    clearance_m: float
    traversability_forward_m: float
    stuck_label: bool
    # Landmark visibility / topology
    landmark_observations: tuple[LandmarkObservation, ...]
    # Integrated body-frame motion over the predictor's action block (5 ticks)
    # and over an H-JEPA history window (16 ticks). Populated by the
    # post-pass once a full window is available; default (0, 0, 0).
    integrated_body_motion_block: tuple[float, float, float] = (0.0, 0.0, 0.0)
    integrated_body_motion_window: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DerivedLabelConfig:
    """Per-rollout knobs for the label pass."""

    yaw_bins: int = DEFAULT_YAW_BINS
    block_size_ticks: int = 5  # data spec §3 macro_step = 5 raw ticks
    history_window_ticks: int = 16  # H-JEPA H default (data spec §3, v3 §5.1)
    stuck_speed_threshold_mps: float = 0.05
    traversability_forward_max_m: float = 2.5
    landmark_visibility_max_m: float = 12.0
    nearest_cell_distance_max_m: float = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DerivedLabelComputer:
    """Stateful per-step label computer.

    Stream usage::

        computer = DerivedLabelComputer(manifest)
        for pose in pose_stream:
            label = computer.step(pose)
            ...

    Per-env state (rolling pose history) is kept internally so callers don't
    have to manage window buffers — the integrated-motion fields are filled
    in once enough history accumulates.
    """

    def __init__(
        self,
        manifest: SceneManifest,
        *,
        config: DerivedLabelConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.scene = SceneGraph(manifest)
        self.config = config or DerivedLabelConfig()
        self._local_graph_type = local_graph_type_per_node(manifest)
        # BFS distance table: cell_id -> {landmark_object_id -> hops or None}
        self._landmark_bfs = _precompute_landmark_bfs(self.scene)
        # Per-env pose history for the integrated-motion windows.
        self._history: dict[int, list[PoseStep]] = {}
        max_window = max(self.config.block_size_ticks, self.config.history_window_ticks)
        self._max_history = max_window + 1

    # ------------------------------------------------------------------

    def step(self, pose: PoseStep) -> DerivedLabelStep:
        history = self._history.setdefault(pose.env_idx, [])
        # New episode → drop prior history so windows never cross resets.
        if history and history[-1].episode_id != pose.episode_id:
            history.clear()
        history.append(pose)
        if len(history) > self._max_history:
            history.pop(0)

        hit = self.scene.locate(pose.position_xy_world)
        yaw_bin = _yaw_to_bin(pose.yaw_world_rad, self.config.yaw_bins)
        clearance = self.scene.clearance_to_walls(pose.position_xy_world)
        traversability = self._traversability_forward(pose)
        local_graph_type = self._local_graph_type.get(hit.cell_id, "unknown")

        landmarks = self._landmark_observations(pose, hit.cell_id)

        # Stuck heuristic: commanded nonzero forward/yaw but observed twist
        # in the last command was below threshold. Per spec §10 we want
        # ~100k of these in train.
        cmd_mag = max(
            abs(pose.last_command[0]),
            abs(pose.last_command[1]),
            abs(pose.last_command[2]),
        )
        speed_proxy = (
            math.hypot(
                pose.position_xy_world[0] - history[-2].position_xy_world[0],
                pose.position_xy_world[1] - history[-2].position_xy_world[1],
            )
            if len(history) >= 2
            else 0.0
        )
        # Convert displacement-per-tick to m/s using inter-step dt.
        dt_s = (
            (pose.timestamp_ns - history[-2].timestamp_ns) / 1e9
            if len(history) >= 2 and pose.timestamp_ns > history[-2].timestamp_ns
            else 0.0
        )
        speed = speed_proxy / dt_s if dt_s > 0.0 else 0.0
        stuck = bool(
            len(history) >= 2
            and cmd_mag > 1e-3
            and speed < self.config.stuck_speed_threshold_mps
        )

        block_motion = _integrated_body_motion(history, self.config.block_size_ticks)
        window_motion = _integrated_body_motion(history, self.config.history_window_ticks)

        return DerivedLabelStep(
            timestamp_ns=pose.timestamp_ns,
            env_idx=pose.env_idx,
            episode_id=pose.episode_id,
            episode_step=pose.episode_step,
            scene_id=self.manifest.scene_id,
            cell_id=int(hit.cell_id),
            nearest_cell_distance_m=float(hit.distance_m),
            local_graph_type=local_graph_type,
            yaw_bin=yaw_bin,
            yaw_bin_count=int(self.config.yaw_bins),
            clearance_m=float(clearance),
            traversability_forward_m=float(traversability),
            stuck_label=stuck,
            landmark_observations=landmarks,
            integrated_body_motion_block=block_motion,
            integrated_body_motion_window=window_motion,
        )

    def stream(self, poses: Iterable[PoseStep]) -> Iterator[DerivedLabelStep]:
        for pose in poses:
            yield self.step(pose)

    # ------------------------------------------------------------------

    def _landmark_observations(
        self, pose: PoseStep, current_cell: int
    ) -> tuple[LandmarkObservation, ...]:
        max_range = self.config.landmark_visibility_max_m
        results: list[LandmarkObservation] = []
        for landmark in self.manifest.landmarks:
            lm_xy = (float(landmark.center_xyz_m[0]), float(landmark.center_xyz_m[1]))
            dx = lm_xy[0] - pose.position_xy_world[0]
            dy = lm_xy[1] - pose.position_xy_world[1]
            range_m = math.hypot(dx, dy)
            bearing_world = math.atan2(dy, dx)
            bearing_body = wrap_angle_pi(bearing_world - pose.yaw_world_rad)
            in_range = range_m <= max_range
            visible = in_range and not _segment_intersects_any_wall(
                self.scene, pose.position_xy_world, lm_xy
            )
            bfs = self._landmark_bfs.get(current_cell, {}).get(landmark.object_id)
            results.append(
                LandmarkObservation(
                    object_id=landmark.object_id,
                    visible=visible,
                    bearing_body_rad=float(bearing_body),
                    range_m=float(range_m),
                    bfs_distance_cells=bfs,
                )
            )
        return tuple(results)

    def _traversability_forward(self, pose: PoseStep) -> float:
        """Distance of free space in the body-forward direction (capped)."""

        max_m = self.config.traversability_forward_max_m
        # Coarse line-search: step forward in 0.10 m increments until we hit
        # an AABB. Cheap and bounded; precision isn't critical for the
        # downstream encoder.
        step_m = 0.10
        steps = int(max_m / step_m)
        x0, y0 = pose.position_xy_world
        cos_y, sin_y = math.cos(pose.yaw_world_rad), math.sin(pose.yaw_world_rad)
        for i in range(1, steps + 1):
            x = x0 + cos_y * step_m * i
            y = y0 + sin_y * step_m * i
            if self.scene.clearance_to_walls((x, y)) <= 0.02:
                return float(step_m * (i - 1))
        return float(max_m)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_to_bin(yaw_rad: float, n_bins: int) -> int:
    if n_bins <= 0:
        raise ValueError("yaw_bins must be positive")
    normalised = (yaw_rad + math.pi) / (2.0 * math.pi)
    normalised -= math.floor(normalised)
    bin_idx = int(normalised * n_bins)
    if bin_idx == n_bins:  # exact +pi maps to bin 0
        bin_idx = 0
    return bin_idx


def pose_steps_from_message_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[PoseStep], MessagePoseJoinSummary]:
    """Extract per-step pose labels from compact raw-rollout records.

    The accepted input shape is the JSON record of a rollout's ``messages.jsonl``
    (`go2nav.corpus.messages`). Records may be single-env (no ``env_index`` field) or
    per-env Genesis topics (``/env_NN/...`` with ``canonical_topic`` set).

    Command blocks are joined by ``(env_index, sequence_id)`` so a delayed
    ``executed_command_block`` record can still provide the actual clipped
    command values for base-state ticks that occurred earlier in the file.
    If no matching command block exists, ``PoseStep.last_command`` is zeroed
    and the summary reports the miss.
    """

    collected = _CollectedRecords()
    for record in records:
        collected.add(record)
    return collected.pose_steps()


def pose_steps_from_messages_jsonl(
    messages_path: str | PathLike[str],
) -> tuple[list[PoseStep], MessagePoseJoinSummary]:
    """Load pose steps from a compact ``messages.jsonl`` raw-rollout file."""

    path = Path(messages_path)
    with path.open(encoding="utf-8") as stream:
        records = (json.loads(line) for line in stream if line.strip())
        return pose_steps_from_message_records(records)


def _precompute_landmark_bfs(scene: SceneGraph) -> dict[int, dict[str, int | None]]:
    """For every cell, BFS distance to every landmark cell."""

    table: dict[int, dict[str, int | None]] = {}
    landmark_cells = scene.landmark_cells
    blocked = getattr(scene, "nav_blocked_cells", frozenset())
    for cell in range(scene.n_nodes):
        per_landmark: dict[str, int | None] = {}
        for object_id, lm_cell in landmark_cells:
            per_landmark[object_id] = scene.bfs_distance(cell, lm_cell, transit_blocked=blocked)
        table[cell] = per_landmark
    return table


@dataclass(frozen=True)
class _CommandSchedule:
    env_key: int | None
    sequence_id: int
    start_ns: int
    command_dt_ns: int
    commands: tuple[tuple[float, float, float], ...]

    def command_at(self, timestamp_ns: int) -> tuple[float, float, float]:
        if not self.commands or self.command_dt_ns <= 0:
            return (0.0, 0.0, 0.0)
        offset_ns = max(0, int(timestamp_ns) - int(self.start_ns))
        # Rollout base_state is stamped after the command tick has completed:
        # a block starting at 0.0s with dt=0.1s emits its first base pose at
        # 0.1s. Treat exact boundaries as the tick that just ran.
        tick = 0 if offset_ns <= 0 else (offset_ns - 1) // int(self.command_dt_ns)
        tick = min(len(self.commands) - 1, tick)
        return self.commands[int(tick)]


@dataclass(frozen=True)
class _BaseRecord:
    env_key: int | None
    env_idx: int
    timestamp_ns: int
    position_xy_world: tuple[float, float]
    yaw_world_rad: float
    fallback_episode_id: int
    fallback_episode_step: int


@dataclass
class _CollectedRecords:
    source_record_count: int = 0
    base_records: list[_BaseRecord] = field(default_factory=list)
    command_starts: dict[tuple[int | None, int], _CommandSchedule] = field(
        default_factory=dict
    )
    executed_commands: dict[
        tuple[int | None, int], tuple[tuple[float, float, float], ...]
    ] = field(default_factory=dict)
    episode_by_env: dict[int | None, dict[str, int]] = field(default_factory=dict)
    base_count_by_env: dict[int | None, int] = field(default_factory=dict)
    command_block_count: int = 0
    executed_command_block_count: int = 0
    episode_info_count: int = 0
    missing_episode_info_count: int = 0

    def add(self, record: dict[str, Any]) -> None:
        self.source_record_count += 1
        topic = _canonical_topic(record)
        payload = record.get("payload", {})
        env_key = _env_key(record)

        if topic == EPISODE_INFO_TOPIC:
            self.episode_info_count += 1
            self.episode_by_env[env_key] = {
                "episode_id": int(payload.get("episode_id", 0)),
                "episode_step": int(payload.get("episode_step", 0)),
            }
            return

        if topic == COMMAND_BLOCK_TOPIC:
            schedule = _command_schedule_from_payload(env_key, payload, executed=False)
            if schedule is not None:
                self.command_block_count += 1
                self.command_starts[(env_key, schedule.sequence_id)] = schedule
            return

        if topic == EXECUTED_COMMAND_BLOCK_TOPIC:
            schedule = _command_schedule_from_payload(env_key, payload, executed=True)
            if schedule is not None:
                self.executed_command_block_count += 1
                self.executed_commands[(env_key, schedule.sequence_id)] = schedule.commands
            return

        if topic != BASE_STATE_TOPIC:
            return

        self.base_count_by_env[env_key] = self.base_count_by_env.get(env_key, 0) + 1
        episode = self.episode_by_env.get(env_key)
        if episode is None:
            self.missing_episode_info_count += 1
            episode_id = 0
            episode_step = self.base_count_by_env[env_key] - 1
        else:
            episode_id = int(episode["episode_id"])
            episode_step = int(episode["episode_step"])

        timestamp_ns = _payload_stamp_ns(payload)
        if timestamp_ns is None:
            timestamp_ns = int(record.get("timestamp_ns", 0))

        pose_world = payload.get("pose_world", {})
        position = pose_world.get("position", {})
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))
        yaw = payload["yaw_rad"]

        self.base_records.append(
            _BaseRecord(
                env_key=env_key,
                env_idx=0 if env_key is None else int(env_key),
                timestamp_ns=int(timestamp_ns),
                position_xy_world=(x, y),
                yaw_world_rad=float(yaw),
                fallback_episode_id=episode_id,
                fallback_episode_step=episode_step,
            )
        )

    def pose_steps(self) -> tuple[list[PoseStep], MessagePoseJoinSummary]:
        schedules_by_env: dict[int | None, list[_CommandSchedule]] = {}
        for key, schedule in self.command_starts.items():
            commands = self.executed_commands.get(key, schedule.commands)
            schedule = _CommandSchedule(
                env_key=schedule.env_key,
                sequence_id=schedule.sequence_id,
                start_ns=schedule.start_ns,
                command_dt_ns=schedule.command_dt_ns,
                commands=commands,
            )
            schedules_by_env.setdefault(schedule.env_key, []).append(schedule)
        for schedules in schedules_by_env.values():
            schedules.sort(key=lambda item: item.start_ns)

        schedule_starts = {
            env_key: [schedule.start_ns for schedule in schedules]
            for env_key, schedules in schedules_by_env.items()
        }

        missing_command_count = 0
        poses: list[PoseStep] = []
        for base in sorted(
            self.base_records,
            key=lambda item: (item.timestamp_ns, -1 if item.env_key is None else item.env_key),
        ):
            command = (0.0, 0.0, 0.0)
            schedules = schedules_by_env.get(base.env_key, ())
            starts = schedule_starts.get(base.env_key, ())
            if schedules and starts:
                idx = bisect_right(starts, base.timestamp_ns) - 1
                if idx >= 0:
                    command = schedules[idx].command_at(base.timestamp_ns)
                else:
                    missing_command_count += 1
            else:
                missing_command_count += 1

            poses.append(
                PoseStep(
                    timestamp_ns=base.timestamp_ns,
                    env_idx=base.env_idx,
                    episode_id=base.fallback_episode_id,
                    episode_step=base.fallback_episode_step,
                    position_xy_world=base.position_xy_world,
                    yaw_world_rad=base.yaw_world_rad,
                    last_command=command,
                )
            )

        env_indices = tuple(sorted({pose.env_idx for pose in poses}))
        summary = MessagePoseJoinSummary(
            source_record_count=int(self.source_record_count),
            base_state_count=len(self.base_records),
            command_block_count=int(self.command_block_count),
            executed_command_block_count=int(self.executed_command_block_count),
            episode_info_count=int(self.episode_info_count),
            pose_step_count=len(poses),
            env_indices=env_indices,
            missing_episode_info_count=int(self.missing_episode_info_count),
            missing_command_count=int(missing_command_count),
        )
        return poses, summary


def _canonical_topic(record: dict[str, Any]) -> str:
    topic = str(record.get("canonical_topic") or record.get("topic") or "")
    match = ENV_TOPIC_RE.match(topic)
    if match:
        return match.group(2)
    return topic


def _env_key(record: dict[str, Any]) -> int | None:
    env = record.get("env_index")
    if env is not None:
        return int(env)
    topic = str(record.get("topic") or "")
    match = ENV_TOPIC_RE.match(topic)
    if match:
        return int(match.group(1))
    return None


def _payload_stamp_ns(payload: dict[str, Any]) -> int | None:
    stamp = payload.get("header", {}).get("stamp")
    if not isinstance(stamp, dict):
        return None
    return int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nanosec", 0))


def _command_schedule_from_payload(
    env_key: int | None,
    payload: dict[str, Any],
    *,
    executed: bool,
) -> _CommandSchedule | None:
    sequence_id = payload.get("sequence_id")
    if sequence_id is None:
        return None
    stamp_ns = _payload_stamp_ns(payload)
    if stamp_ns is None:
        return None
    command_dt_ns = int(round(float(payload.get("command_dt_s", 0.10)) * 1e9))
    if executed:
        vx = _float_sequence(payload.get("executed_vx_body_mps", ()))
        vy = _float_sequence(payload.get("executed_vy_body_mps", ()))
        yaw = _float_sequence(payload.get("executed_yaw_rate_radps", ()))
    else:
        vx = _float_sequence(payload.get("vx_body_mps", ()))
        vy = _float_sequence(payload.get("vy_body_mps", ()))
        yaw = _float_sequence(payload.get("yaw_rate_radps", ()))
    n = min(len(vx), len(vy), len(yaw))
    commands = tuple((vx[i], vy[i], yaw[i]) for i in range(n))
    return _CommandSchedule(
        env_key=env_key,
        sequence_id=int(sequence_id),
        start_ns=int(stamp_ns),
        command_dt_ns=command_dt_ns,
        commands=commands,
    )


def _float_sequence(value: Sequence[Any] | Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(float(item) for item in value)




def _integrated_body_motion(
    history: list[PoseStep], window_ticks: int
) -> tuple[float, float, float]:
    """Body-frame ``(dx, dy, dyaw)`` over the last ``window_ticks`` poses."""

    if len(history) <= 1 or window_ticks <= 0:
        return (0.0, 0.0, 0.0)
    span = min(window_ticks, len(history) - 1)
    start = history[-1 - span]
    end = history[-1]
    cos_y, sin_y = math.cos(start.yaw_world_rad), math.sin(start.yaw_world_rad)
    dx_world = end.position_xy_world[0] - start.position_xy_world[0]
    dy_world = end.position_xy_world[1] - start.position_xy_world[1]
    dx_body = cos_y * dx_world + sin_y * dy_world
    dy_body = -sin_y * dx_world + cos_y * dy_world
    dyaw = wrap_angle_pi(end.yaw_world_rad - start.yaw_world_rad)
    return (float(dx_body), float(dy_body), float(dyaw))


def _segment_intersects_any_wall(
    scene: SceneGraph,
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    *,
    samples: int = 24,
) -> bool:
    """Coarse line-of-sight: sample points along segment, test clearance.

    Genuine segment / AABB intersection would be cleaner but the SceneGraph
    only exposes a ``clearance_to_walls`` query. The dynamic sample count keeps
    adjacent samples no more than 0.04 m apart, half the spec minimum wall
    thickness.
    """

    if samples <= 1:
        return False
    ax, ay = a_xy
    bx, by = b_xy
    segment_len = math.hypot(bx - ax, by - ay)
    dynamic_samples = max(samples, int(math.ceil(segment_len / 0.04)))
    for i in range(1, dynamic_samples):
        t = float(i) / float(dynamic_samples)
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t
        if scene.clearance_to_walls((x, y)) <= 0.0:
            return True
    return False


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def label_to_jsonable(label: DerivedLabelStep) -> dict:
    return {
        "timestamp_ns": int(label.timestamp_ns),
        "env_idx": int(label.env_idx),
        "episode_id": int(label.episode_id),
        "episode_step": int(label.episode_step),
        "scene_id": str(label.scene_id),
        "cell_id": int(label.cell_id),
        "nearest_cell_distance_m": float(label.nearest_cell_distance_m),
        "local_graph_type": str(label.local_graph_type),
        "yaw_bin": int(label.yaw_bin),
        "yaw_bin_count": int(label.yaw_bin_count),
        "clearance_m": float(label.clearance_m),
        "traversability_forward_m": float(label.traversability_forward_m),
        "stuck_label": bool(label.stuck_label),
        "integrated_body_motion_block": list(label.integrated_body_motion_block),
        "integrated_body_motion_window": list(label.integrated_body_motion_window),
        "bfs_distance_to_landmark": {
            lm.object_id: (
                int(lm.bfs_distance_cells) if lm.bfs_distance_cells is not None else None
            )
            for lm in label.landmark_observations
        },
        "landmarks": [
            {
                "object_id": lm.object_id,
                "visible": bool(lm.visible),
                "bearing_body_rad": float(lm.bearing_body_rad),
                "range_m": float(lm.range_m),
                "bfs_distance_cells": (
                    int(lm.bfs_distance_cells) if lm.bfs_distance_cells is not None else None
                ),
            }
            for lm in label.landmark_observations
        ],
    }


def derive(scene_dir: Path, corpus: Path, out: Path) -> dict:
    """Write out/labels.jsonl and out/summary.json for one bulk-rollout scene; returns the summary."""
    from go2nav.corpus.manifest import manifest_sha256, parse_scene_manifest_dict
    scene_dir, corpus, out = Path(scene_dir), Path(corpus), Path(out)
    rollout = json.loads((scene_dir/'summary.json').read_text())
    manifest_path = corpus/rollout['split']/rollout['family']/rollout['scene_id']/'manifest.json'
    manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
    poses, join = pose_steps_from_messages_jsonl(scene_dir/'messages.jsonl')
    out.mkdir(parents=True, exist_ok=True)
    count, graph_types = 0, {}
    with (out/'labels.jsonl').open('w', encoding='utf-8') as stream:
        for label in DerivedLabelComputer(manifest, config=DerivedLabelConfig()).stream(poses):
            payload = label_to_jsonable(label)
            graph_types[payload['local_graph_type']] = graph_types.get(payload['local_graph_type'], 0)+1
            stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':'))+'\n')
            count += 1
    summary = dict(scene_id=manifest.scene_id, family=manifest.family, split=manifest.split,
                   manifest_sha256=manifest_sha256(manifest), labels=count, environments=list(join.env_indices),
                   missing_commands=join.missing_command_count, local_graph_type_counts=dict(sorted(graph_types.items())))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scene_dir', type=Path)
    p.add_argument('corpus', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    summary = derive(a.scene_dir, a.corpus, a.out)
    print(json.dumps(dict(scene_id=summary['scene_id'], labels=summary['labels'])))


if __name__ == '__main__':
    main()
