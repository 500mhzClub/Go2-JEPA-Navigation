"""Canonical scene manifest schema and dispatcher into per-family builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    center_xy_m: tuple[float, float]
    width_m: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    width_m: float
    traversable: bool = True


@dataclass(frozen=True)
class BoxObject:
    object_id: str
    kind: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float
    material_id: str
    # Roll/pitch let a box represent a tilted ramp or step face. They default
    # to zero so existing axis-aligned walls/obstacles/landmarks are unchanged.
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass(frozen=True)
class SpawnSpec:
    xyz_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraValidityConstraints:
    min_wall_thickness_m: float
    near_m: float
    far_m: float
    min_camera_clearance_m: float


# ---------------------------------------------------------------------------
# Domain randomization schema (data-spec §14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialOverride:
    """Per-scene RGBA override for a material id used in the scene palette.

    ``material_id`` matches the ids attached to ``BoxObject`` and the synthetic
    ``floor`` material used by the SDF and Genesis exporters. Landmark ids are
    deliberately *not* overridden by the visual-randomization helpers because
    landmark identity is part of the task signal.
    """

    material_id: str
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class LightingSpec:
    """Single directional light spec used by the SDF and Genesis exporters.

    Direction is a unit vector in world coordinates; positive ``z`` is up. The
    diffuse/specular/ambient triplets are linear RGB in ``[0, 1]``.
    """

    direction: tuple[float, float, float]
    diffuse_rgb: tuple[float, float, float]
    specular_rgb: tuple[float, float, float]
    ambient_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class VisualRandomization:
    """Per-scene material and lighting variation.

    The randomization is fully described by the manifest so that any renderer
    consuming ``manifest.json`` or ``genesis_scene.json`` reproduces the same
    visual scene deterministically from ``visual_seed``.
    """

    material_overrides: tuple[MaterialOverride, ...]
    lighting: LightingSpec
    distractor_objects: tuple[BoxObject, ...] = ()


@dataclass(frozen=True)
class PhysicsRandomization:
    """Per-scene physics variation (friction + restitution for floor/objects).

    Values are intended to be passed directly to the underlying simulator
    surface APIs (Genesis ``gs.materials.Rigid(friction=...)`` and SDF
    ``<surface><friction>``). Restitution is reported even when the simulator
    only consumes friction so the manifest stays the single source of truth.
    """

    floor_friction_mu: float
    floor_restitution: float
    obstacle_friction_mu: float
    obstacle_restitution: float


@dataclass(frozen=True)
class CameraExtrinsicJitter:
    """Per-scene offset applied to the platform camera mount.

    Offsets are *additive* to the platform manifest's nominal ``xyz_body_m``
    and ``rpy_body_rad``. The renderer applies them at scene build time so the
    same scene seed produces the same camera pose across CPU/AMDGPU backends.
    """

    xyz_offset_m: tuple[float, float, float]
    rpy_offset_rad: tuple[float, float, float]


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    family: str
    difficulty_tier: str
    topology_seed: int
    visual_seed: int
    physics_seed: int
    world_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    spawn: SpawnSpec
    graph_nodes: tuple[GraphNode, ...]
    graph_edges: tuple[GraphEdge, ...]
    obstacles: tuple[BoxObject, ...]
    landmarks: tuple[BoxObject, ...]
    camera_constraints: CameraValidityConstraints
    split: str | None = None
    walls: tuple[BoxObject, ...] = field(default_factory=tuple)
    # Domain randomization (data-spec §14). Optional so manifests built by
    # callers that predate the schema additions still validate; the smoke and
    # standard plans always populate them via ``families.py``.
    visual_randomization: VisualRandomization | None = None
    physics_randomization: PhysicsRandomization | None = None
    camera_extrinsic_jitter: CameraExtrinsicJitter | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def static_objects(self) -> tuple[BoxObject, ...]:
        distractors: tuple[BoxObject, ...] = ()
        if self.visual_randomization is not None:
            distractors = self.visual_randomization.distractor_objects
        return (*self.walls, *self.obstacles, *self.landmarks, *distractors)


def parse_scene_manifest_dict(payload: dict[str, Any]) -> SceneManifest:
    """Reconstruct a :class:`SceneManifest` from its ``to_dict`` representation.

    Optional fields default to ``None``/empty so old on-disk manifests
    predating the §14 randomization additions still parse.
    """

    def _tuple3(seq: Any, *, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
        if seq is None:
            return default
        return (float(seq[0]), float(seq[1]), float(seq[2]))

    def _tuple4(seq: Any, *, default: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> tuple[float, float, float, float]:
        if seq is None:
            return default
        return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))

    def _parse_box(entry: dict[str, Any]) -> BoxObject:
        return BoxObject(
            object_id=str(entry["object_id"]),
            kind=str(entry["kind"]),
            center_xyz_m=_tuple3(entry["center_xyz_m"]),
            size_xyz_m=_tuple3(entry["size_xyz_m"]),
            yaw_rad=float(entry.get("yaw_rad", 0.0)),
            material_id=str(entry.get("material_id", "")),
            roll_rad=float(entry.get("roll_rad", 0.0)),
            pitch_rad=float(entry.get("pitch_rad", 0.0)),
        )

    def _parse_node(entry: dict[str, Any]) -> GraphNode:
        return GraphNode(
            node_id=int(entry["node_id"]),
            center_xy_m=(float(entry["center_xy_m"][0]), float(entry["center_xy_m"][1])),
            width_m=float(entry["width_m"]),
            tags=tuple(str(t) for t in entry.get("tags", ())),
        )

    def _parse_edge(entry: dict[str, Any]) -> GraphEdge:
        return GraphEdge(
            source=int(entry["source"]),
            target=int(entry["target"]),
            width_m=float(entry["width_m"]),
            traversable=bool(entry.get("traversable", True)),
        )

    def _parse_visual(payload_v: dict[str, Any] | None) -> VisualRandomization | None:
        if not payload_v:
            return None
        overrides = tuple(
            MaterialOverride(
                material_id=str(entry["material_id"]),
                rgba=tuple(float(v) for v in entry["rgba"]),
            )
            for entry in payload_v.get("material_overrides", ())
        )
        light = payload_v.get("lighting", {})
        lighting = LightingSpec(
            direction=tuple(float(v) for v in light.get("direction", (0.0, 0.0, -1.0))),
            diffuse_rgb=tuple(float(v) for v in light.get("diffuse_rgb", (0.8, 0.8, 0.8))),
            specular_rgb=tuple(float(v) for v in light.get("specular_rgb", (0.2, 0.2, 0.2))),
            ambient_rgb=tuple(float(v) for v in light.get("ambient_rgb", (0.25, 0.25, 0.25))),
        )
        distractors = tuple(
            _parse_box(entry) for entry in payload_v.get("distractor_objects", ())
        )
        return VisualRandomization(
            material_overrides=overrides,
            lighting=lighting,
            distractor_objects=distractors,
        )

    def _parse_physics(payload_p: dict[str, Any] | None) -> PhysicsRandomization | None:
        if not payload_p:
            return None
        return PhysicsRandomization(
            floor_friction_mu=float(payload_p.get("floor_friction_mu", 1.0)),
            floor_restitution=float(payload_p.get("floor_restitution", 0.0)),
            obstacle_friction_mu=float(payload_p.get("obstacle_friction_mu", 0.85)),
            obstacle_restitution=float(payload_p.get("obstacle_restitution", 0.0)),
        )

    def _parse_camera_jitter(payload_c: dict[str, Any] | None) -> CameraExtrinsicJitter | None:
        if not payload_c:
            return None
        return CameraExtrinsicJitter(
            xyz_offset_m=_tuple3(payload_c.get("xyz_offset_m", (0.0, 0.0, 0.0))),
            rpy_offset_rad=_tuple3(payload_c.get("rpy_offset_rad", (0.0, 0.0, 0.0))),
        )

    spawn_payload = payload.get("spawn", {})
    spawn = SpawnSpec(
        xyz_m=_tuple3(spawn_payload.get("xyz_m", (0.0, 0.0, 0.375))),
        quat_wxyz=_tuple4(spawn_payload.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
    )

    camera_payload = payload.get("camera_constraints", {})
    camera_constraints = CameraValidityConstraints(
        min_wall_thickness_m=float(camera_payload.get("min_wall_thickness_m", 0.08)),
        near_m=float(camera_payload.get("near_m", 0.05)),
        far_m=float(camera_payload.get("far_m", 200.0)),
        min_camera_clearance_m=float(camera_payload.get("min_camera_clearance_m", 0.10)),
    )

    bounds = payload.get("world_bounds_xy_m", [[-1.0, -1.0], [1.0, 1.0]])
    world_bounds = (
        (float(bounds[0][0]), float(bounds[0][1])),
        (float(bounds[1][0]), float(bounds[1][1])),
    )

    return SceneManifest(
        scene_id=str(payload["scene_id"]),
        family=str(payload["family"]),
        difficulty_tier=str(payload.get("difficulty_tier", "")),
        topology_seed=int(payload.get("topology_seed", 0)),
        visual_seed=int(payload.get("visual_seed", 0)),
        physics_seed=int(payload.get("physics_seed", 0)),
        world_bounds_xy_m=world_bounds,
        spawn=spawn,
        graph_nodes=tuple(_parse_node(n) for n in payload.get("graph_nodes", ())),
        graph_edges=tuple(_parse_edge(e) for e in payload.get("graph_edges", ())),
        obstacles=tuple(_parse_box(o) for o in payload.get("obstacles", ())),
        landmarks=tuple(_parse_box(l) for l in payload.get("landmarks", ())),
        camera_constraints=camera_constraints,
        split=payload.get("split"),
        walls=tuple(_parse_box(w) for w in payload.get("walls", ())),
        visual_randomization=_parse_visual(payload.get("visual_randomization")),
        physics_randomization=_parse_physics(payload.get("physics_randomization")),
        camera_extrinsic_jitter=_parse_camera_jitter(payload.get("camera_extrinsic_jitter")),
    )


def build_scene_manifest(
    scene_seed: int,
    family: str = "open_obstacle_field",
    *,
    split: str | None = None,
    difficulty_tier: str | None = None,
) -> SceneManifest:
    """Return a deterministic scene manifest for the requested family.

    The actual per-family geometry lives in :mod:`go2nav.corpus.families`. This
    function is the stable entry point used by the smoke generator, the corpus
    builder, and downstream renderers.
    """

    from go2nav.corpus.families import build_family_manifest

    return build_family_manifest(
        scene_seed=int(scene_seed),
        family=family,
        split=split,
        difficulty_tier=difficulty_tier,
    )


def manifest_sha256(manifest: SceneManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_scene_id(family: str, scene_seed: int) -> str:
    digest = hashlib.sha256(f"{family}:{scene_seed}".encode("utf-8")).hexdigest()
    return f"{family}_{digest[:12]}"
