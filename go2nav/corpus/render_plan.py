"""Render plans: the camera pose of every 100-ms frame of a bulk rollout, for the renderer (`go2nav.corpus.render`).

    python -m go2nav.corpus.render_plan ROLLOUT_SCENE_DIR OUT_DIR

Reads ROLLOUT_SCENE_DIR/messages.jsonl and summary.json (`go2nav.corpus.bulk`) and writes
- OUT_DIR/frames.jsonl: per environment, one frame per 100 ms of base_state messages: the base pose, the episode, the
  latest command block and joint state, and the camera pose from the platform manifest's camera mount;
- OUT_DIR/render_replay_plan.json: the scene, the camera and the frame count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from go2nav.corpus.contract import rotate_body_to_world

PLATFORM = Path(__file__).resolve().parents[2]/'configs/corpus/go2_platform_manifest.yaml'


def build_plan(scene_dir: str | Path, out_dir: str | Path, *, camera_hz: float = 10.0) -> dict[str, Any]:
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    summary = json.loads((scene_dir/'summary.json').read_text())
    camera = (yaml.safe_load(PLATFORM.read_text()) or {}).get('camera', {})
    frames = _extract_base_state_frames(scene_dir/'messages.jsonl', camera_hz=camera_hz, max_frames=None,
                                        camera_mount_body=_camera_mount(camera))
    if not frames:
        raise ValueError(f'no /lewm/go2/base_state frames in {scene_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/'frames.jsonl').open('w', encoding='utf-8') as stream:
        for frame in frames:
            stream.write(json.dumps(frame, sort_keys=True, separators=(',', ':'))+'\n')
    plan = dict(schema='lewm_render_replay_plan_v0', scene_id=summary['scene_id'], scene_family=summary['family'],
                split=summary['split'], manifest_sha256=summary['manifest_sha256'],
                source_messages_jsonl=str(scene_dir/'messages.jsonl'), frames_jsonl=str(out_dir/'frames.jsonl'),
                frame_count=len(frames), source_env_count=_source_env_count(frames), camera_hz=camera_hz,
                first_frame_timestamp_ns=frames[0]['timestamp_ns'], last_frame_timestamp_ns=frames[-1]['timestamp_ns'],
                camera={k: camera.get(k) for k in ('native_resolution', 'training_resolution', 'fov_axis', 'fov_deg',
                                                   'near_m', 'far_m', 'encoding')} | {'mount_body': _camera_mount(camera)})
    (out_dir/'render_replay_plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True), encoding='utf-8')
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scene_dir', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    plan = build_plan(a.scene_dir, a.out)
    print(json.dumps(dict(scene_id=plan['scene_id'], frames=plan['frame_count'])))


if __name__ == '__main__':
    main()
