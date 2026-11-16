"""Replay scheduling helpers for deterministic GPU rendering.

The functions in this module deliberately avoid importing Genesis at module
import time. They define the replay contract consumed by a GPU renderer:
``raw_rollout`` state and command data in, deterministic per-frame camera/base
poses out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lewm_genesis.lewm_contract import rotate_body_to_world




def build_render_replay_plan(
    raw_rollout_dir: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "genesis",
    camera_hz: float = 10.0,
    platform_manifest: str | Path | None = None,
    max_frames: int | None = None,
    require_quality_pass: bool = True,
) -> dict[str, Any]:
    """Write a GPU render-replay plan for a converted ``raw_rollout``.

    The output is not a rendered dataset. It is the immutable job contract that a
    backend such as Genesis must consume to produce ``rendered_vision``. Keeping
    this step separate lets Gazebo remain the dynamics oracle while rendering
    can scale on GPU hardware.
    """

    raw_dir = Path(raw_rollout_dir).resolve()
    out_dir = Path(output_dir).resolve()
    summary_path = raw_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"raw_rollout summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_bag_summary = _load_source_bag_summary(summary)
    data_quality = summary.get("data_quality_audit", {})
    contract = summary.get("contract_audit", {})
    if require_quality_pass and not contract.get("pass", False):
        raise ValueError(f"raw_rollout contract audit failed: {contract.get('issues', [])}")
    if require_quality_pass and not data_quality.get("pass", False):
        raise ValueError(f"raw_rollout data-quality audit failed: {data_quality.get('issues', [])}")

    messages_path = Path(summary["messages_jsonl"])
    if not messages_path.is_absolute():
        messages_path = raw_dir / messages_path
    platform = _load_platform_manifest(platform_manifest)
    camera = platform.get("camera", {})

    frames = _extract_base_state_frames(
        messages_path,
        camera_hz=camera_hz,
        max_frames=max_frames,
        camera_mount_body=_camera_mount(camera),
    )
    if not frames:
        raise ValueError(f"no /lewm/go2/base_state frames found in {messages_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    plan = {
        "schema": "lewm_render_replay_plan_v0",
        "backend": backend,
        "gpu_required": True,
        "render_status": "planned_not_rendered",
        "raw_rollout_dir": str(raw_dir),
        "raw_summary_json": str(summary_path),
        "source_bag": summary.get("source_bag"),
        "source_bag_summary_json": source_bag_summary.get("_summary_path"),
        "scene_id": source_bag_summary.get("scene_id"),
        "scene_family": source_bag_summary.get("family"),
        "split": source_bag_summary.get("split"),
        "manifest_sha256": source_bag_summary.get("manifest_sha256"),
        "source_messages_jsonl": str(messages_path),
        "output_dir": str(out_dir),
        "frames_jsonl": str(frames_path),
        "frame_count": len(frames),
        "source_env_count": _source_env_count(frames),
        "camera_hz": camera_hz,
        "first_frame_timestamp_ns": frames[0]["timestamp_ns"],
        "last_frame_timestamp_ns": frames[-1]["timestamp_ns"],
        "raw_contract_audit_pass": contract.get("pass", False),
        "raw_data_quality_audit_pass": data_quality.get("pass", False),
        "raw_quality_profile": data_quality.get("profile"),
        "camera": {
            "native_resolution": camera.get("native_resolution"),
            "training_resolution": camera.get("training_resolution"),
            "fov_axis": camera.get("fov_axis"),
            "fov_deg": camera.get("fov_deg"),
            "near_m": camera.get("near_m"),
            "far_m": camera.get("far_m"),
            "encoding": camera.get("encoding"),
            "mount_body": _camera_mount(camera),
        },
        "backend_contract": {
            "must_preserve_frame_order": True,
            "must_write_camera_valid_flags": True,
            "must_not_modify_reset_or_command_arrays": True,
            "must_report_invalid_frame_rate": True,
            "depth_required_for_validity_audit": True,
        },
    }
    plan_path = out_dir / "render_replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def _extract_base_state_frames(
    messages_path: Path,
    *,
    camera_hz: float,
    max_frames: int | None,
    camera_mount_body: dict[str, Any],
) -> list[dict[str, Any]]:
    if camera_hz <= 0.0:
        raise ValueError("camera_hz must be positive")

    period_ns = int(round(1_000_000_000 / camera_hz))
    frames: list[dict[str, Any]] = []
    latest_episode_by_env: dict[int | None, dict[str, Any]] = {}
    latest_joint_state_by_env: dict[int | None, dict[str, Any]] = {}
    latest_command_by_env: dict[int | None, dict[str, Any]] = {}
    next_frame_ns_by_env: dict[int | None, int] = {}
    source_line = 0

    with messages_path.open(encoding="utf-8") as stream:
        for line in stream:
            source_line += 1
            record = json.loads(line)
            source_topic = record.get("topic")
            topic = record.get("canonical_topic", source_topic)
            env_index = record.get("env_index")
            env_key = int(env_index) if env_index is not None else None
            payload = record.get("payload", {})
            if topic == "/lewm/episode_info":
                latest_episode_by_env[env_key] = {
                    "scene_id": payload.get("scene_id"),
                    "episode_id": payload.get("episode_id"),
                    "episode_step": payload.get("episode_step"),
                    "reset_count": payload.get("reset_count"),
                    "split": payload.get("split"),
                    "manifest_sha256": payload.get("manifest_sha256"),
                }
                continue
            if topic == "/joint_states":
                latest_joint_state_by_env[env_key] = {
                    "names": payload.get("name"),
                    "position": payload.get("position"),
                    "velocity": payload.get("velocity"),
                    "effort": payload.get("effort"),
                }
                continue
            if topic == "/lewm/go2/command_block":
                latest_command_by_env[env_key] = {
                    "sequence_id": payload.get("sequence_id"),
                    "primitive_name": payload.get("primitive_name"),
                    "command_source": payload.get("command_source"),
                    "route_target_id": payload.get("route_target_id"),
                    "next_waypoint_id": payload.get("next_waypoint_id"),
                    "vx_body_mps": payload.get("vx_body_mps"),
                    "vy_body_mps": payload.get("vy_body_mps"),
                    "yaw_rate_radps": payload.get("yaw_rate_radps"),
                    "command_dt_s": payload.get("command_dt_s"),
                    "block_size": payload.get("block_size"),
                    "source_line": source_line,
                    "timestamp_ns": int(record["timestamp_ns"]),
                }
                continue
            if topic != "/lewm/go2/base_state":
                continue

            timestamp_ns = _payload_stamp_ns(payload)
            if timestamp_ns is None:
                timestamp_ns = int(record["timestamp_ns"])
            next_frame_ns = next_frame_ns_by_env.get(env_key)
            if next_frame_ns is None:
                next_frame_ns = timestamp_ns
            if timestamp_ns + 1_000 < next_frame_ns:
                continue

            frames.append(
                {
                    "frame_index": len(frames),
                    "env_index": env_index,
                    "source_line": source_line,
                    "source_topic": source_topic,
                    "canonical_topic": topic,
                    "timestamp_ns": timestamp_ns,
                    "timestamp_s": round(timestamp_ns / 1e9, 9),
                    "record_timestamp_ns": int(record["timestamp_ns"]),
                    "episode": dict(latest_episode_by_env.get(env_key, {})),
                    "base_pose_world": payload.get("pose_world"),
                    "base_quat_world_xyzw": payload.get("quat_world_xyzw"),
                    "base_rpy_rad": {
                        "roll": payload.get("roll_rad"),
                        "pitch": payload.get("pitch_rad"),
                        "yaw": payload.get("yaw_rad"),
                    },
                    "twist_body": payload.get("twist_body"),
                    "command_context": latest_command_by_env.get(env_key),
                    "joint_state": latest_joint_state_by_env.get(env_key),
                    "camera_pose_world": _camera_pose_from_payload(
                        payload,
                        camera_mount_body,
                    ),
                    "camera_mount_body": camera_mount_body,
                }
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
            next_frame_ns_by_env[env_key] = timestamp_ns + period_ns

    return frames


def _source_env_count(frames: list[dict[str, Any]]) -> int:
    env_indices = [int(frame.get("env_index") or 0) for frame in frames]
    return max(1, max(env_indices, default=0) + 1)


def _load_platform_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"platform manifest not found: {manifest_path}")
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def _load_source_bag_summary(raw_summary: dict[str, Any]) -> dict[str, Any]:
    source_bag = raw_summary.get("source_bag")
    if not source_bag:
        return {}
    summary_path = Path(source_bag) / "summary.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_summary_path"] = str(summary_path)
    return summary


def _camera_mount(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_link": camera.get("parent_link"),
        "xyz_body_m": camera.get("xyz_body_m"),
        "rpy_body_rad": camera.get("rpy_body_rad"),
    }


def _camera_pose_from_payload(
    base_state_payload: dict[str, Any],
    camera_mount_body: dict[str, Any],
) -> dict[str, Any] | None:
    position = (
        base_state_payload.get("pose_world", {})
        .get("position", {})
    )
    quat_xyzw = base_state_payload.get("quat_world_xyzw")
    mount_xyz = camera_mount_body.get("xyz_body_m")
    mount_rpy = camera_mount_body.get("rpy_body_rad")
    if not position or quat_xyzw is None or mount_xyz is None or mount_rpy is None:
        return None
    base_pos = np.array(
        [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ],
        dtype=np.float32,
    )
    base_quat = np.asarray(quat_xyzw, dtype=np.float32)
    mount_rot = _rpy_matrix(tuple(float(v) for v in mount_rpy))
    mount_offset = np.asarray(tuple(float(v) for v in mount_xyz), dtype=np.float32)
    forward_body = mount_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up_body = mount_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_pos = base_pos + rotate_body_to_world(mount_offset, base_quat).astype(np.float32)
    forward_world = rotate_body_to_world(forward_body, base_quat).astype(np.float32)
    up_world = rotate_body_to_world(up_body, base_quat).astype(np.float32)
    lookat = cam_pos + forward_world
    return {
        "position": [float(v) for v in cam_pos],
        "lookat": [float(v) for v in lookat],
        "up": [float(v) for v in up_world],
    }


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _payload_stamp_ns(payload: dict[str, Any]) -> int | None:
    stamp = payload.get("header", {}).get("stamp")
    if not isinstance(stamp, dict):
        return None
    return int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nanosec", 0))
