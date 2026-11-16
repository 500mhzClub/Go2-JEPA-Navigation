"""Domain-randomization samplers (data-spec §14).

Visual, physics, and camera-extrinsic randomization is sampled from per-scene
seeds so that the same ``visual_seed`` / ``physics_seed`` always reproduces
the same scene. The randomization budget is captured by a :class:`Profile`
that families pick to express how aggressive randomization should be — most
families use :data:`DEFAULT_PROFILE`; the visual-stress family uses
:data:`VISUAL_STRESS_PROFILE`; the rough-dynamics family uses
:data:`ROUGH_DYNAMICS_PROFILE`.

Landmark colors are deliberately *not* randomized: landmark identity is part
of the task signal and changing colors per scene would break the goal-image
contract that GoalAdapter relies on later (see ``docs/v3_topological_nav_plan.md``).
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass
from typing import Iterable

from go2nav.corpus.manifest import (
    BoxObject,
    CameraExtrinsicJitter,
    LightingSpec,
    MaterialOverride,
    PhysicsRandomization,
    VisualRandomization,
)


# ---------------------------------------------------------------------------
# Base palette (matches the SDF/Genesis exporter palette)
# ---------------------------------------------------------------------------


_LANDMARK_PREFIX = "landmark_"

BASE_PALETTE: dict[str, tuple[float, float, float, float]] = {
    "floor": (0.45, 0.47, 0.42, 1.0),
    "wall_interior": (0.62, 0.62, 0.60, 1.0),
    "wall_perimeter": (0.40, 0.40, 0.42, 1.0),
    "mat_obstacle_0": (0.55, 0.55, 0.52, 1.0),
    "mat_obstacle_1": (0.42, 0.48, 0.38, 1.0),
    "mat_obstacle_2": (0.56, 0.42, 0.35, 1.0),
    "ramp_concrete": (0.55, 0.55, 0.55, 1.0),
    "step_platform": (0.50, 0.45, 0.40, 1.0),
    "slick_patch": (0.30, 0.40, 0.55, 1.0),
    "distractor_pole": (0.65, 0.55, 0.40, 1.0),
}

# Landmark colors stay fixed — task identity, not background visual.
LANDMARK_PALETTE: dict[str, tuple[float, float, float, float]] = {
    "landmark_red": (0.85, 0.12, 0.08, 1.0),
    "landmark_blue": (0.10, 0.22, 0.85, 1.0),
    "landmark_green": (0.10, 0.65, 0.18, 1.0),
    "landmark_yellow": (0.92, 0.80, 0.10, 1.0),
}




# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """Bounded ranges for visual / physics / camera randomization.

    Each per-scene draw uniform-samples from these ranges using a single
    deterministic ``random.Random`` seeded from the scene's visual/physics
    seed plus a per-family salt.
    """

    # Visual ranges (HSV deltas applied to base palette; alpha is not jittered)
    hue_jitter: float
    saturation_jitter: float
    value_jitter: float

    # Lighting
    light_elevation_deg_range: tuple[float, float]
    light_intensity_range: tuple[float, float]
    light_ambient_range: tuple[float, float]

    # Physics
    floor_friction_range: tuple[float, float]
    floor_restitution_range: tuple[float, float]
    obstacle_friction_range: tuple[float, float]
    obstacle_restitution_range: tuple[float, float]

    # Camera mount jitter (added to platform manifest's nominal mount pose)
    camera_xyz_jitter_m: float
    camera_rpy_jitter_rad: float

    # Distractor objects: small landmark-like-but-misleading box clutter.
    n_distractors_range: tuple[int, int]


# Default profile — most families use this. Tightly bounded so the LeWM
# encoder still sees a recognizable visual distribution across scenes.
DEFAULT_PROFILE = Profile(
    hue_jitter=0.06,
    saturation_jitter=0.15,
    value_jitter=0.18,
    light_elevation_deg_range=(40.0, 75.0),
    light_intensity_range=(0.70, 0.95),
    light_ambient_range=(0.18, 0.30),
    floor_friction_range=(0.85, 1.15),
    floor_restitution_range=(0.0, 0.05),
    obstacle_friction_range=(0.70, 1.00),
    obstacle_restitution_range=(0.0, 0.05),
    camera_xyz_jitter_m=0.005,
    camera_rpy_jitter_rad=0.020,
    n_distractors_range=(0, 0),
)

# Rough/local dynamics — friction range widens significantly; visual stays
# muted so the model can isolate the physics signal.
ROUGH_DYNAMICS_PROFILE = Profile(
    hue_jitter=0.06,
    saturation_jitter=0.15,
    value_jitter=0.18,
    light_elevation_deg_range=(40.0, 75.0),
    light_intensity_range=(0.70, 0.95),
    light_ambient_range=(0.18, 0.30),
    floor_friction_range=(0.35, 1.40),
    floor_restitution_range=(0.0, 0.15),
    obstacle_friction_range=(0.40, 1.20),
    obstacle_restitution_range=(0.0, 0.10),
    camera_xyz_jitter_m=0.005,
    camera_rpy_jitter_rad=0.020,
    n_distractors_range=(0, 0),
)

# Visual / sensor stress — wide color and lighting variation, multiple
# distractor objects with landmark-adjacent colors, larger camera jitter.
VISUAL_STRESS_PROFILE = Profile(
    hue_jitter=0.20,
    saturation_jitter=0.40,
    value_jitter=0.40,
    light_elevation_deg_range=(20.0, 85.0),
    light_intensity_range=(0.45, 1.10),
    light_ambient_range=(0.10, 0.40),
    floor_friction_range=(0.85, 1.15),
    floor_restitution_range=(0.0, 0.05),
    obstacle_friction_range=(0.70, 1.00),
    obstacle_restitution_range=(0.0, 0.05),
    camera_xyz_jitter_m=0.020,
    camera_rpy_jitter_rad=0.060,
    n_distractors_range=(6, 12),
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def draw_visual_randomization(
    *,
    visual_seed: int,
    family: str,
    profile: Profile = DEFAULT_PROFILE,
    extra_palette: Iterable[str] = (),
    distractor_bounds_half_m: float = 3.0,
    distractor_height_range_m: tuple[float, float] = (0.4, 0.9),
) -> VisualRandomization:
    rng = random.Random(f"visual:{family}:{visual_seed}")

    palette_keys = list(BASE_PALETTE)
    for key in extra_palette:
        if key not in palette_keys and key not in LANDMARK_PALETTE:
            palette_keys.append(key)

    overrides: list[MaterialOverride] = []
    for material_id in palette_keys:
        base = BASE_PALETTE.get(material_id)
        if base is None:
            # Caller passed an unknown material id without a base color; fall
            # back to a neutral grey so we always produce a deterministic
            # palette entry.
            base = (0.55, 0.55, 0.55, 1.0)
        overrides.append(
            MaterialOverride(
                material_id=material_id,
                rgba=_jitter_rgba(rng, base, profile),
            )
        )

    lighting = _sample_lighting(rng, profile)
    distractors = _sample_distractors(
        rng,
        profile=profile,
        bounds_half_m=distractor_bounds_half_m,
        height_range_m=distractor_height_range_m,
    )
    return VisualRandomization(
        material_overrides=tuple(overrides),
        lighting=lighting,
        distractor_objects=distractors,
    )


def draw_physics_randomization(
    *, physics_seed: int, family: str, profile: Profile = DEFAULT_PROFILE
) -> PhysicsRandomization:
    rng = random.Random(f"physics:{family}:{physics_seed}")
    return PhysicsRandomization(
        floor_friction_mu=round(_uniform(rng, profile.floor_friction_range), 3),
        floor_restitution=round(_uniform(rng, profile.floor_restitution_range), 3),
        obstacle_friction_mu=round(_uniform(rng, profile.obstacle_friction_range), 3),
        obstacle_restitution=round(_uniform(rng, profile.obstacle_restitution_range), 3),
    )


def draw_camera_extrinsic_jitter(
    *, physics_seed: int, family: str, profile: Profile = DEFAULT_PROFILE
) -> CameraExtrinsicJitter:
    # Camera jitter is keyed off the physics seed (not the visual seed) so
    # the same physical scene gets the same mount across visual themes.
    rng = random.Random(f"camera:{family}:{physics_seed}")
    j_xyz = profile.camera_xyz_jitter_m
    j_rpy = profile.camera_rpy_jitter_rad
    return CameraExtrinsicJitter(
        xyz_offset_m=(
            round(rng.uniform(-j_xyz, j_xyz), 4),
            round(rng.uniform(-j_xyz, j_xyz), 4),
            round(rng.uniform(-j_xyz, j_xyz), 4),
        ),
        rpy_offset_rad=(
            round(rng.uniform(-j_rpy, j_rpy), 4),
            round(rng.uniform(-j_rpy, j_rpy), 4),
            round(rng.uniform(-j_rpy, j_rpy), 4),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if hi < lo:
        lo, hi = hi, lo
    return rng.uniform(lo, hi)


def _jitter_rgba(
    rng: random.Random,
    base: tuple[float, float, float, float],
    profile: Profile,
) -> tuple[float, float, float, float]:
    r, g, b, a = base
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.uniform(-profile.hue_jitter, profile.hue_jitter)) % 1.0
    s = _clip01(s + rng.uniform(-profile.saturation_jitter, profile.saturation_jitter))
    v = _clip01(v + rng.uniform(-profile.value_jitter, profile.value_jitter))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (round(r2, 4), round(g2, 4), round(b2, 4), round(a, 4))


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sample_lighting(rng: random.Random, profile: Profile) -> LightingSpec:
    azimuth = rng.uniform(-math.pi, math.pi)
    elevation_deg = _uniform(rng, profile.light_elevation_deg_range)
    elevation = math.radians(elevation_deg)
    # Direction points *from* the sun *toward* the scene (Gazebo convention).
    dx = math.cos(elevation) * math.cos(azimuth)
    dy = math.cos(elevation) * math.sin(azimuth)
    dz = -math.sin(elevation)
    intensity = _uniform(rng, profile.light_intensity_range)
    ambient = _uniform(rng, profile.light_ambient_range)
    diffuse = (round(intensity, 3),) * 3
    # Specular is a fraction of diffuse with mild color tint variation.
    specular_scale = rng.uniform(0.15, 0.30)
    specular = (round(intensity * specular_scale, 3),) * 3
    ambient_rgb = (round(ambient, 3),) * 3
    return LightingSpec(
        direction=(round(dx, 4), round(dy, 4), round(dz, 4)),
        diffuse_rgb=diffuse,
        specular_rgb=specular,
        ambient_rgb=ambient_rgb,
    )


def _sample_distractors(
    rng: random.Random,
    *,
    profile: Profile,
    bounds_half_m: float,
    height_range_m: tuple[float, float],
) -> tuple[BoxObject, ...]:
    lo, hi = profile.n_distractors_range
    if hi < lo:
        lo, hi = hi, lo
    count = rng.randint(lo, hi)
    if count <= 0:
        return ()

    # Distractor colors deliberately reuse the landmark palette so visual-
    # stress scenes give the model false-positive candidates without changing
    # the actual landmark palette. ``kind="distractor"`` keeps privileged
    # labels honest: the renderer paints with landmark color but the object is
    # not a real landmark.
    distractor_palette = (
        "landmark_red",
        "landmark_blue",
        "landmark_green",
        "landmark_yellow",
        "distractor_pole",
    )
    distractors: list[BoxObject] = []
    for index in range(count):
        material = rng.choice(distractor_palette)
        size_z = round(_uniform(rng, height_range_m), 3)
        size_x = round(rng.uniform(0.20, 0.45), 3)
        size_y = round(rng.uniform(0.20, 0.45), 3)
        x = round(rng.uniform(-bounds_half_m, bounds_half_m), 3)
        y = round(rng.uniform(-bounds_half_m, bounds_half_m), 3)
        distractors.append(
            BoxObject(
                object_id=f"distractor_{index:02d}_{material}",
                kind="distractor",
                center_xyz_m=(x, y, size_z * 0.5),
                size_xyz_m=(size_x, size_y, size_z),
                yaw_rad=round(rng.uniform(-math.pi, math.pi), 4),
                material_id=material,
            )
        )
    return tuple(distractors)
