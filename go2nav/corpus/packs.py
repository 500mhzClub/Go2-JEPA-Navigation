"""Load a corpus scene directory into a typed ``ScenePack``.

A pack combines the scene's ``manifest.json`` and ``genesis_scene.json`` (written by `go2nav.corpus.build`) with
the platform manifest (``configs/corpus/go2_platform_manifest.yaml``): the resolved Go2 URDF and spawn pose, the
camera mount, physics timing, static objects, randomization, and the provenance stamped on episode and reset
records. Pure Python; `go2nav.corpus.world` builds the Genesis scene from it.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from go2nav.corpus.manifest import SceneManifest, parse_scene_manifest_dict
from go2nav.corpus.graph import SceneGraph


# Unitree Go2 URDF link names in LeWM fl/fr/rl/rr order.
# These match the standard ``unitree_go2_description`` xacro output.
DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER: tuple[str, str, str, str] = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)


@dataclass(frozen=True)
class StaticObject:
    """One simulator-side static object (wall, obstacle, or landmark)."""

    object_id: str
    kind: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float
    material_id: str
    # roll/pitch let an object represent a slope or tilted ramp; default 0 so
    # axis-aligned walls/obstacles are unchanged from the legacy schema.
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass(frozen=True)
class MaterialOverride:
    material_id: str
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class LightingSpec:
    direction: tuple[float, float, float]
    diffuse_rgb: tuple[float, float, float]
    specular_rgb: tuple[float, float, float]
    ambient_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class VisualRandomization:
    material_overrides: tuple[MaterialOverride, ...]
    lighting: LightingSpec
    # Distractors are folded into ``static_objects`` at load time so the
    # scene builder needs no second loop; this field is kept for callers
    # that want to introspect distractor presence/count separately.
    distractor_object_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicsRandomization:
    floor_friction_mu: float
    floor_restitution: float
    obstacle_friction_mu: float
    obstacle_restitution: float


@dataclass(frozen=True)
class CameraExtrinsicJitter:
    xyz_offset_m: tuple[float, float, float]
    rpy_offset_rad: tuple[float, float, float]


@dataclass(frozen=True)
class RobotSpec:
    """Resolved Go2 description and spawn pose."""

    urdf_path: Path
    spawn_xyz_m: tuple[float, float, float]
    spawn_quat_wxyz: tuple[float, float, float, float]
    foot_links_in_lewm_order: tuple[str, str, str, str]


@dataclass(frozen=True)
class CameraMount:
    """Camera intrinsics + body-frame mount pose from the platform manifest."""

    parent_link: str
    xyz_body_m: tuple[float, float, float]
    rpy_body_rad: tuple[float, float, float]
    native_resolution: tuple[int, int]
    training_resolution: tuple[int, int]
    fov_axis: str
    fov_deg: float
    near_m: float
    far_m: float
    encoding: str


@dataclass(frozen=True)
class PhysicsTiming:
    """Physics, policy, and command timing from the platform manifest."""

    physics_dt_s: float
    policy_dt_s: float
    command_dt_s: float
    action_block_size: int

    @property
    def policy_decimation(self) -> int:
        """Physics ticks per policy tick."""

        return max(1, int(round(self.policy_dt_s / self.physics_dt_s)))

    @property
    def command_ticks_per_block(self) -> int:
        """Policy ticks per command-block tick."""

        return max(1, int(round(self.command_dt_s / self.policy_dt_s)))


@dataclass(frozen=True)
class ScenePack:
    """Everything the Genesis builder needs to construct one scene."""

    scene_id: str
    family: str
    split: str
    difficulty_tier: str
    manifest_sha256: str
    physics_seed: int
    topology_seed: int
    visual_seed: int
    world_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    static_objects: tuple[StaticObject, ...]
    robot: RobotSpec
    camera: CameraMount
    timing: PhysicsTiming
    camera_constraints: dict[str, float]
    source_dir: Path
    visual_randomization: VisualRandomization | None = None
    physics_randomization: PhysicsRandomization | None = None
    camera_extrinsic_jitter: CameraExtrinsicJitter | None = None
    # Canonical scene manifest and pre-built scene graph. Optional so legacy
    # consumers that constructed ScenePack directly (e.g. small test fixtures)
    # still compile; the production loader always populates them.
    manifest: SceneManifest | None = None
    scene_graph: SceneGraph | None = None


def load_platform_manifest(path: str | Path) -> dict[str, Any]:
    """Load and parse the platform YAML manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"platform manifest not found: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"platform manifest must be a mapping: {manifest_path}")
    return data


def camera_mount_from_platform(platform_manifest: dict[str, Any]) -> CameraMount:
    """Extract the camera mount + intrinsics from the platform manifest."""

    camera = platform_manifest.get("camera", {})
    native = camera.get("native_resolution", [640, 480])
    training = camera.get("training_resolution", list(native))
    xyz = camera.get("xyz_body_m", [0.0, 0.0, 0.0])
    rpy = camera.get("rpy_body_rad", [0.0, 0.0, 0.0])
    return CameraMount(
        parent_link=str(camera.get("parent_link", "camera_link")),
        xyz_body_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        rpy_body_rad=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        native_resolution=(int(native[0]), int(native[1])),
        training_resolution=(int(training[0]), int(training[1])),
        fov_axis=str(camera.get("fov_axis", "horizontal")),
        fov_deg=float(camera.get("fov_deg", 78.323)),
        near_m=float(camera.get("near_m", 0.05)),
        far_m=float(camera.get("far_m", 200.0)),
        encoding=str(camera.get("encoding", "rgb8")),
    )


def physics_timing_from_platform(platform_manifest: dict[str, Any]) -> PhysicsTiming:
    """Extract physics/policy/command timing from the platform manifest."""

    timing = platform_manifest.get("timing", {})
    return PhysicsTiming(
        physics_dt_s=float(timing.get("physics_dt_s", 0.002)),
        policy_dt_s=float(timing.get("policy_dt_s", 0.02)),
        command_dt_s=float(timing.get("command_dt_s", 0.10)),
        action_block_size=int(timing.get("action_block_size", 5)),
    )


def _load_static_objects(genesis_scene: dict[str, Any]) -> tuple[StaticObject, ...]:
    objects: list[StaticObject] = []
    for obj in genesis_scene.get("objects", []):
        center = obj["center_xyz_m"]
        size = obj["size_xyz_m"]
        objects.append(
            StaticObject(
                object_id=str(obj["object_id"]),
                kind=str(obj["kind"]),
                center_xyz_m=(float(center[0]), float(center[1]), float(center[2])),
                size_xyz_m=(float(size[0]), float(size[1]), float(size[2])),
                yaw_rad=float(obj.get("yaw_rad", 0.0)),
                material_id=str(obj.get("material_id", "")),
                roll_rad=float(obj.get("roll_rad", 0.0)),
                pitch_rad=float(obj.get("pitch_rad", 0.0)),
            )
        )
    return tuple(objects)


def _load_visual_randomization(
    genesis_scene: dict[str, Any]
) -> VisualRandomization | None:
    payload = genesis_scene.get("visual_randomization")
    if not payload:
        return None
    overrides = tuple(
        MaterialOverride(
            material_id=str(entry["material_id"]),
            rgba=tuple(float(v) for v in entry["rgba"]),
        )
        for entry in payload.get("material_overrides", [])
    )
    light_payload = payload.get("lighting", {})
    lighting = LightingSpec(
        direction=tuple(float(v) for v in light_payload.get("direction", (0.0, 0.0, -1.0))),
        diffuse_rgb=tuple(float(v) for v in light_payload.get("diffuse_rgb", (0.8, 0.8, 0.8))),
        specular_rgb=tuple(float(v) for v in light_payload.get("specular_rgb", (0.2, 0.2, 0.2))),
        ambient_rgb=tuple(float(v) for v in light_payload.get("ambient_rgb", (0.25, 0.25, 0.25))),
    )
    distractor_ids = tuple(
        str(entry["object_id"]) for entry in payload.get("distractor_objects", [])
    )
    return VisualRandomization(
        material_overrides=overrides,
        lighting=lighting,
        distractor_object_ids=distractor_ids,
    )


def _load_physics_randomization(
    genesis_scene: dict[str, Any]
) -> PhysicsRandomization | None:
    payload = genesis_scene.get("physics_randomization")
    if not payload:
        return None
    return PhysicsRandomization(
        floor_friction_mu=float(payload.get("floor_friction_mu", 1.0)),
        floor_restitution=float(payload.get("floor_restitution", 0.0)),
        obstacle_friction_mu=float(payload.get("obstacle_friction_mu", 0.85)),
        obstacle_restitution=float(payload.get("obstacle_restitution", 0.0)),
    )


def _load_camera_jitter(
    genesis_scene: dict[str, Any]
) -> CameraExtrinsicJitter | None:
    payload = genesis_scene.get("camera_extrinsic_jitter")
    if not payload:
        return None
    return CameraExtrinsicJitter(
        xyz_offset_m=tuple(
            float(v) for v in payload.get("xyz_offset_m", (0.0, 0.0, 0.0))
        ),
        rpy_offset_rad=tuple(
            float(v) for v in payload.get("rpy_offset_rad", (0.0, 0.0, 0.0))
        ),
    )


def _verify_manifest_consistency(
    manifest: dict[str, Any], genesis_scene: dict[str, Any]
) -> None:
    """Cross-check the two on-disk JSON files agree on identity and hash."""

    for key in ("scene_id", "family", "manifest_sha256", "physics_seed"):
        if manifest.get(key) != genesis_scene.get(key):
            raise ValueError(
                f"manifest.json and genesis_scene.json disagree on {key!r}: "
                f"{manifest.get(key)!r} vs {genesis_scene.get(key)!r}"
            )


def load_scene_pack(
    scene_dir: str | Path,
    *,
    platform_manifest: dict[str, Any] | str | Path,
    workspace_root: str | Path,
    foot_links_in_lewm_order: tuple[str, str, str, str] = DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
) -> ScenePack:
    """Load one scene directory into a fully-resolved ``ScenePack``."""

    scene_path = Path(scene_dir).resolve()
    manifest_path = scene_path / "manifest.json"
    genesis_path = scene_path / "genesis_scene.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing under {scene_path}")
    if not genesis_path.is_file():
        raise FileNotFoundError(f"genesis_scene.json missing under {scene_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    genesis_scene = json.loads(genesis_path.read_text(encoding="utf-8"))
    _verify_manifest_consistency(manifest, genesis_scene)
    scene_manifest = parse_scene_manifest_dict(manifest)
    scene_graph = SceneGraph(scene_manifest)

    if isinstance(platform_manifest, (str, Path)):
        platform = load_platform_manifest(platform_manifest)
    else:
        platform = platform_manifest

    urdf_path = resolve_go2_urdf(platform, Path(workspace_root))
    spawn = genesis_scene["spawn"]
    bounds = genesis_scene["world_bounds_xy_m"]

    return ScenePack(
        scene_id=str(manifest["scene_id"]),
        family=str(manifest["family"]),
        split=str(manifest.get("split", "")),
        difficulty_tier=str(manifest.get("difficulty_tier", "")),
        manifest_sha256=str(manifest["manifest_sha256"]),
        physics_seed=int(manifest["physics_seed"]),
        topology_seed=int(manifest.get("topology_seed", 0)),
        visual_seed=int(manifest.get("visual_seed", 0)),
        world_bounds_xy_m=(
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
        ),
        static_objects=_load_static_objects(genesis_scene),
        robot=RobotSpec(
            urdf_path=urdf_path,
            spawn_xyz_m=(
                float(spawn["xyz_m"][0]),
                float(spawn["xyz_m"][1]),
                float(spawn["xyz_m"][2]),
            ),
            spawn_quat_wxyz=(
                float(spawn["quat_wxyz"][0]),
                float(spawn["quat_wxyz"][1]),
                float(spawn["quat_wxyz"][2]),
                float(spawn["quat_wxyz"][3]),
            ),
            foot_links_in_lewm_order=foot_links_in_lewm_order,
        ),
        camera=camera_mount_from_platform(platform),
        timing=physics_timing_from_platform(platform),
        camera_constraints=dict(genesis_scene.get("camera_constraints", {})),
        source_dir=scene_path,
        visual_randomization=_load_visual_randomization(genesis_scene),
        physics_randomization=_load_physics_randomization(genesis_scene),
        camera_extrinsic_jitter=_load_camera_jitter(genesis_scene),
        manifest=scene_manifest,
        scene_graph=scene_graph,
    )


def find_scene_dirs(
    corpus_root: str | Path,
    *,
    split: str | None = None,
    family: str | None = None,
) -> list[Path]:
    """Return scene directories under ``corpus_root``.

    The corpus layout is ``<corpus_root>/<split>/<family>/<scene_id>/``. A scene
    directory is one that contains both ``manifest.json`` and
    ``genesis_scene.json``. Returned in sorted order for determinism.
    """

    root = Path(corpus_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root is not a directory: {root}")

    splits = [split] if split is not None else None
    found: list[Path] = []
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        if splits is not None and split_dir.name not in splits:
            continue
        for family_dir in sorted(split_dir.iterdir()):
            if not family_dir.is_dir():
                continue
            if family is not None and family_dir.name != family:
                continue
            for scene_dir in sorted(family_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                if (scene_dir / "manifest.json").is_file() and (
                    scene_dir / "genesis_scene.json"
                ).is_file():
                    found.append(scene_dir)
    return found


# -- Go2 robot asset -------------------------------------------------------------------------------------------------

GENESIS_BUILTIN_GO2_URDF = "genesis_builtin_go2"


def resolve_go2_urdf(platform_manifest: dict[str, Any], workspace_root: Path) -> Path:
    """The Go2 URDF the platform manifest names: Genesis's bundled Go2 (``genesis_builtin_go2``) or a path relative
    to ``workspace_root``."""
    value = str(platform_manifest.get("robot", {}).get("genesis_urdf", GENESIS_BUILTIN_GO2_URDF))
    if value != GENESIS_BUILTIN_GO2_URDF:
        return Path(workspace_root) / value
    spec = importlib.util.find_spec("genesis")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("robot.genesis_urdf is genesis_builtin_go2, but the genesis package is not importable")
    return Path(next(iter(spec.submodule_search_locations))) / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"
