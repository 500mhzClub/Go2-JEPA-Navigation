"""Build the Genesis physics scene for a bulk rollout from a `ScenePack`.

The scene is the floor plane, the manifest's static boxes (with the scene's friction and colours) and the Go2 at its
spawn pose, built with `n_envs` parallel environments. It carries no camera: frames are rendered afterwards from the
recorded poses (`go2nav.corpus.render`, which builds its own textured scene). Genesis is imported lazily, so this
module imports without it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from go2nav.corpus.packs import ScenePack
from go2nav.corpus.randomization import BASE_PALETTE, LANDMARK_PALETTE

_GENESIS_INITIALIZED = False


def _import_genesis():
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError('Genesis is not installed; go2nav.corpus.world needs it at run time') from error
    return gs


def initialize_genesis(backend: str = 'cpu', seed: int | None = None) -> None:
    """Initialise Genesis once per process (`gs.init` is process-global); later calls are no-ops."""
    global _GENESIS_INITIALIZED
    if _GENESIS_INITIALIZED:
        return
    gs = _import_genesis()
    backend_obj = getattr(gs, {'amd': 'amdgpu', 'hip': 'amdgpu', 'rocm': 'amdgpu'}.get(backend, backend), None)
    if backend_obj is None:
        raise RuntimeError(f'Genesis backend {backend!r} is unavailable in genesis-world {gs.__version__}')
    kwargs: dict[str, Any] = {'backend': backend_obj}
    if seed is not None:
        kwargs['seed'] = int(seed) & 0x7FFF_FFFF  # the compiler accepts signed 32-bit seeds
    gs.init(**kwargs)
    _GENESIS_INITIALIZED = True


@dataclass
class SceneBuild:
    """The Genesis entities produced from a `ScenePack`."""

    scene: Any
    robot: Any
    pack: ScenePack
    n_envs: int


def build_scene_from_pack(pack: ScenePack, *, n_envs: int, backend: str = 'cpu') -> SceneBuild:
    """Build the physics scene with `n_envs` parallel environments. Genesis is initialised with the first scene's
    physics seed (later scenes in the same process reuse that initialisation)."""
    gs = _import_genesis()
    initialize_genesis(backend=backend, seed=pack.physics_seed)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=float(pack.timing.physics_dt_s), gravity=(0., 0., -9.81)),
                     show_viewer=False, renderer=gs.renderers.Rasterizer())
    colours = _material_colours(pack)
    floor = {}
    if pack.physics_randomization is not None:
        floor['material'] = _rigid_material(gs, pack.physics_randomization.floor_friction_mu)
    floor['surface'] = _surface(gs, 'floor', colours)
    scene.add_entity(gs.morphs.Plane(), **{k: v for k, v in floor.items() if v is not None})
    for obj in pack.static_objects:
        kwargs = {'name': obj.object_id}
        if pack.physics_randomization is not None:
            kwargs['material'] = _rigid_material(gs, pack.physics_randomization.obstacle_friction_mu)
        kwargs['surface'] = _surface(gs, obj.material_id, colours)
        euler_deg = tuple(math.degrees(float(a)) for a in (obj.roll_rad, obj.pitch_rad, obj.yaw_rad))
        scene.add_entity(gs.morphs.Box(pos=obj.center_xyz_m, size=obj.size_xyz_m, euler=euler_deg, fixed=True),
                         **{k: v for k, v in kwargs.items() if v is not None})
    robot = scene.add_entity(gs.morphs.URDF(file=str(pack.robot.urdf_path), pos=pack.robot.spawn_xyz_m,
                                            quat=pack.robot.spawn_quat_wxyz, fixed=False, visualization=False,
                                            collision=True), name='go2')
    scene.build(n_envs=int(n_envs))
    return SceneBuild(scene=scene, robot=robot, pack=pack, n_envs=int(n_envs))


def _rigid_material(gs, friction: float):
    """Genesis rigid friction must lie in [0.01, 5]; an out-of-range manifest value is clamped to the edge."""
    return gs.materials.Rigid(friction=max(1e-2, min(5., float(friction))))


def _material_colours(pack: ScenePack) -> dict[str, tuple[float, float, float, float]]:
    """material_id -> RGBA: the scene's colour overrides, then the landmark and base palettes."""
    colours = {}
    if pack.visual_randomization is not None:
        for override in pack.visual_randomization.material_overrides:
            colours[override.material_id] = override.rgba
    for palette in (LANDMARK_PALETTE, BASE_PALETTE):
        for key, rgba in palette.items():
            colours.setdefault(key, rgba)
    return colours


def _surface(gs, material_id: str, colours):
    rgba = colours.get(material_id) if material_id else None
    return gs.surfaces.Default(color=tuple(rgba)) if rgba is not None else None
