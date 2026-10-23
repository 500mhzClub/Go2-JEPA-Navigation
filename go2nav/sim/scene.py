"""The Genesis scene of one maze episode: collision geometry, the robot, the visual surfaces and the camera.

Built in a fixed order, as the original did, so that physics and rendering reproduce exactly (CPU backend, Genesis
seeded with the episode's simulation seed):
1. a collision plane and one fixed box per wall (both invisible), friction 1.0;
2. the Go2 (Genesis' bundled URDF) at the episode's home pose, 0.375 m up;
3. the visual-only floor and wall surfaces (`appearance`);
4. one 640 x 480 camera (78.323° horizontal field of view, near 5 mm, far 200 m), moved to each mount before each
   render (`go2nav.sim.session`).
After the build: a fixed raster order (floor, walls, then the robot's visual geometry), the episode's friction on the
robot and the ground, and the policy's joint gains (kp 20, kv 0.5).

With moving obstacles (`dynamics.obstacles`, `go2nav.sim.obstacles`), one fixed, visible and colliding cylinder per
obstacle is added after the robot, and drawn after the walls; without them the scene is exactly as before.
"""
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from types import MethodType

import numpy as np

from go2nav.mazes import grid
from go2nav.sim import appearance, demo_appearance, random_appearance

PHYSICS_DT_S, POLICY_DT_S, COMMAND_DT_S = .002, .02, .1
PHYSICS_STEPS_PER_POLICY = 10
POLICY_STEPS_PER_COMMAND = 5
SPAWN_HEIGHT_M = .375
RESOLUTION = (640, 480)
HORIZONTAL_FOV_DEG, NEAR_M, FAR_M = 78.323, .005, 200.
CAMERA_MOUNT_M = (.326, 0., .043)
MATERIAL_FRICTION = 1.
GAINS = dict(kp=20., kv=.5)
# Rollout joint order (all hips, then thighs, then calves; legs FL, FR, RL, RR).
LEG_JOINTS = tuple(f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR'))


def go2_urdf():
    """The Go2 URDF bundled with the installed Genesis."""
    spec = importlib.util.find_spec('genesis')
    return Path(next(iter(spec.submodule_search_locations)))/'assets'/'urdf'/'go2'/'urdf'/'go2.urdf'


def vertical_fov_deg():
    """Genesis' `fov` is vertical; the platform records the horizontal one."""
    width, height = RESOLUTION
    return math.degrees(2.*math.atan(math.tan(math.radians(HORIZONTAL_FOV_DEG)*.5)/(float(width)/float(height))))


def spawn_pose(home_se2):
    """World spawn position and wxyz quaternion for a home pose (x, y, yaw)."""
    x, y, yaw = home_se2
    return (x, y, SPAWN_HEIGHT_M), (math.cos(yaw/2), 0., 0., math.sin(yaw/2))


def initial_camera_pose(spawn_xyz, spawn_wxyz):
    """Camera position and look-at point at the spawn (the camera is re-posed before every render)."""
    qw, qx, qy, qz = spawn_wxyz
    yaw = math.atan2(2.*(qw*qz+qx*qy), 1.-2.*(qy*qy+qz*qz))
    mx, my, mz = CAMERA_MOUNT_M
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = spawn_xyz[0]+c*mx-s*my, spawn_xyz[1]+s*mx+c*my, spawn_xyz[2]+mz
    return (x, y, z), (x+c*1., y+s*1., z-.1-math.tan(0.))


@dataclass
class MazeScene:
    scene: object
    robot: object
    camera: object
    floor: object                 # the collision plane
    walls: list                   # the collision boxes, in wall-box order
    surfaces: list                # the visual floor and walls
    leg_dofs: list                # robot DOF indices in rollout joint order
    links: dict                   # contact topology: robot, support (calves and feet) and ground link indices
    guard_geoms: dict             # robot, foot (the calf spheres) and ground geom indices
    obstacles: list = None        # the moving obstacles' cylinders (`go2nav.sim.obstacles`), if any


def init_genesis(seed):
    import genesis as gs
    gs.init(backend=gs.cpu, seed=int(seed) & 0x7FFF_FFFF, logging_level='warning')


def build(spec, output_dir, dynamics=None, robot_visible=True, demo_textures=False, random_look=None):
    """Build the scene of a maze spec (with the episode's `procedural_seed` and home pose in `geometry.spawn_se2_world`).
    Genesis must already be initialised (`init_genesis`). Visual meshes are written to `output_dir`. `dynamics`
    (`go2nav.sim.dynamics.Dynamics`): marked strips are tinted on the floor, and a uniform friction replaces the
    episode's after the build. `robot_visible=False` builds the robot without visual geometry (the older corridor
    collections rendered only the floor and walls). `demo_textures` draws the demonstration brick and parquet surfaces
    (`go2nav.sim.demo_appearance`) instead of the maze's appearance arm. `random_look` (a `random_appearance.sample`
    record; the LeWM corpus's free units only) draws its surfaces and lighting instead."""
    import genesis as gs
    from go2nav.sim.dynamics import tint_floor
    boxes = spec['geometry']['wall_boxes']
    if random_look is not None:
        if dynamics is not None and dynamics.marked:
            raise ValueError('marked strips are drawn on the standard floor only')
        meshes = random_appearance.surfaces(boxes, random_look)
    elif demo_textures:
        if dynamics is not None and dynamics.marked:
            raise ValueError('marked strips are drawn on the standard floor only')
        meshes = demo_appearance.surfaces(boxes)
    else:
        meshes = appearance.surfaces(boxes, spec['appearance_arm'], spec['appearance_seed'])
    if dynamics is not None and dynamics.marked:
        floor_mesh, tinted = tint_floor(meshes[0][1], dynamics.strips)
        if dynamics.strips and not tinted:
            raise ValueError('marked strips tint no floor cell')
        meshes[0] = (meshes[0][0], floor_mesh)
    spawn_xyz, spawn_wxyz = spawn_pose(spec['geometry']['spawn_se2_world'])
    lighting = dict(vis_options=random_appearance.vis_options(random_look['lighting'])) if random_look is not None else {}
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=PHYSICS_DT_S, gravity=(0., 0., -9.81)), show_viewer=False,
                     renderer=gs.renderers.Rasterizer(), **lighting)
    floor_material, wall_material = gs.materials.Rigid(friction=MATERIAL_FRICTION), gs.materials.Rigid(friction=MATERIAL_FRICTION)
    floor = scene.add_entity(gs.morphs.Plane(visualization=False), material=floor_material, name='ground_plane')
    walls = [scene.add_entity(gs.morphs.Box(pos=b['centre_xyz'], size=b['size_xyz'], euler=(0., 0., math.degrees(b['yaw_rad'])),
                                            fixed=True, visualization=False), material=wall_material, name=b['wall_id'])
             for b in boxes]
    robot = scene.add_entity(gs.morphs.URDF(file=str(go2_urdf()), pos=spawn_xyz, quat=spawn_wxyz, fixed=False,
                                            visualization=robot_visible, collision=True), name='go2')
    placement = dynamics.obstacles if dynamics is not None and dynamics.obstacles else []
    obstacles = [scene.add_entity(gs.morphs.Cylinder(pos=(*(np.asarray(o['start_cell'], float)*grid.PITCH_M), o['height_m']/2),
                                                     radius=o['radius_m'], height=o['height_m'], fixed=True,
                                                     visualization=True, collision=True),
                                  material=gs.materials.Rigid(friction=MATERIAL_FRICTION),
                                  surface=gs.surfaces.Default(color=tuple(o['colour'])), name=f'obstacle_{k}')
                 for k, o in enumerate(placement)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = []
    for name, mesh in meshes:
        loaded = appearance.through_ply(mesh, output_dir/f'{name}.ply')
        surfaces.append(scene.add_entity(gs.morphs.MeshSet(files=[loaded], fixed=True, collision=False, visualization=True,
                                                           decimate=False, convexify=False, align=False,
                                                           file_meshes_are_zup=True), name=name))
    position, lookat = initial_camera_pose(spawn_xyz, spawn_wxyz)
    camera = scene.add_camera(res=RESOLUTION, pos=position, lookat=lookat, fov=vertical_fov_deg(), near=NEAR_M, far=FAR_M,
                              GUI=False)
    scene.build(n_envs=1)
    fix_raster_order(camera, surfaces, robot, obstacles)
    set_friction(robot, floor, spec['friction_mu'])
    if dynamics is not None and dynamics.friction_mu is not None:
        set_friction(robot, floor, dynamics.friction_mu)
    joints = {str(j.name): j for j in robot.joints}
    leg_dofs = [int(np.asarray(joints[name].dofs_idx).reshape(-1)[0]) for name in LEG_JOINTS]
    robot.set_dofs_kp([GAINS['kp']]*12, leg_dofs)
    robot.set_dofs_kv([GAINS['kv']]*12, leg_dofs)
    links = dict(robot={int(l.idx) for l in robot.links},
                 support={int(l.idx) for l in robot.links if 'foot' in str(l.name).lower() or 'calf' in str(l.name).lower()},
                 ground={int(l.idx) for l in floor.links})
    feet = [int(g.idx) for g in robot.geoms if g.type.name == 'SPHERE' and str(g.link.name).endswith('_calf')]
    if len(feet) != 4:
        raise ValueError('four calf foot spheres expected')
    guard_geoms = dict(robot=[int(g.idx) for g in robot.geoms], foot=sorted(feet), ground=[int(g.idx) for g in floor.geoms])
    return MazeScene(scene, robot, camera, floor, walls, surfaces, leg_dofs, links, guard_geoms, obstacles)


def fix_raster_order(camera, surfaces, robot, obstacles=()):
    """Draw the floor, then the walls, then any obstacles, then the robot, in every render (the rasterizer's own
    order is not fixed)."""
    context = camera._rasterizer._context
    raster = context._scene
    ordered = [context.rigid_nodes[g.uid] for entity in surfaces for g in entity.vgeoms]+\
              [context.rigid_nodes[g.uid] for entity in obstacles for g in entity.vgeoms]+\
              [context.rigid_nodes[g.uid] for g in robot.vgeoms]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(raster.mesh_nodes):
        raise ValueError('the visible population is not floor, walls, obstacles and robot')

    def fixed_order(this):
        if set(this.mesh_nodes) != set(ordered):
            raise ValueError('visible population changed')
        return list(ordered)
    raster.sorted_mesh_nodes = MethodType(fixed_order, raster)
    raster._meshes_updated = True


def set_friction(robot, floor, mu):
    """The episode's friction on every robot geometry and the ground (Genesis combines a pair by the maximum)."""
    robot.set_friction(mu)
    floor.set_friction(mu)
