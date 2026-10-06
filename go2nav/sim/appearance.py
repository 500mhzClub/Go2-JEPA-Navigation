"""What the cameras see: visual-only floor and wall surfaces, separate from the collision geometry.

The floor is one 32 x 32 m square; the walls are the outer surface of the union of all wall boxes (no hidden internal
faces). Both are tiled in 12.5-cm cells of uniform grey, so the visual odometry has texture to track:
- `distinctive` (the capability mazes): each cell an independent grey level 40-230, seeded per face from the maze's
  appearance seed and the face's own coordinates;
- `repeated`: a 50/205 checkerboard; `neutral`: uniform grey (128; 89 on walls).

Coordinates are snapped to a 1-um grid so the union is exact. The meshes are written as PLY and read back before they
reach the renderer (vertices become float32 there), as in the original.
"""
import hashlib
import json
import math

import numpy as np
import trimesh

ARMS = ('neutral', 'repeated', 'distinctive')
CELL_M = .125
FLOOR_HALF_M = 16
UM = 1e6
MAX_CELLS = 65536


def patch(origin, u, v, lengths, *, arm, seed, neutral=128):
    """A rectangle origin + s·u + t·v (0 ≤ s, t ≤ lengths) tiled in coloured cells: two triangles per cell, four
    unshared vertices per cell."""
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    lengths = np.asarray(lengths, float)
    if arm not in ARMS or type(seed) is not int or seed < 0:
        raise ValueError('appearance arm and non-negative integer seed required')
    n, m = np.ceil(lengths/CELL_M).astype(int)
    x, y = np.meshgrid(np.linspace(0, lengths[0], n+1), np.linspace(0, lengths[1], m+1))
    p = origin+x[..., None]*u+y[..., None]*v
    quads = np.stack((p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]), axis=2).reshape(-1, 4, 3)
    vertices = quads.reshape(-1, 3)
    base = np.arange(len(quads))*4
    faces = np.stack((base[:, None]+[0, 1, 2], base[:, None]+[0, 2, 3]), axis=1).reshape(-1, 3)
    if arm == 'neutral':
        gray = np.full(len(quads), neutral, np.uint8)
    elif arm == 'repeated':
        gray = np.where(np.indices((m, n)).sum(0).ravel() % 2, 205, 50).astype(np.uint8)
    else:
        gray = np.random.default_rng(seed).integers(40, 231, len(quads), dtype=np.uint8)
    color = np.column_stack((np.repeat(gray, 4)[:, None]*np.ones((1, 3), np.uint8), np.full(len(vertices), 255, np.uint8)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=color, process=False)


def union_faces(boxes):
    """The exposed faces of the union of axis-aligned wall boxes, sorted: dicts of axis, plane_um, u0_um, u1_um, v0_um,
    v1_um, normal_sign, owner_wall_id (the face's in-plane axes are cyclic: u = axis+1, v = axis+2)."""
    bounds = {}
    for box in boxes:
        if box['yaw_rad'] != 0 or box['wall_id'] in bounds:
            raise ValueError('unique axis-aligned walls required')
        centre, half = np.asarray(box['centre_xyz'], float), np.asarray(box['size_xyz'], float)/2
        bounds[box['wall_id']] = np.rint(np.stack([centre-half, centre+half])*UM).astype(np.int64)
    if any(b[0, 2] != 0 for b in bounds.values()):
        raise ValueError('walls must stand on the floor')
    coordinates = [sorted({int(v) for b in bounds.values() for v in b[:, a]}) for a in range(3)]
    shape = tuple(len(c)-1 for c in coordinates)
    if math.prod(shape) > MAX_CELLS:
        raise ValueError('too many cells in the compressed wall union')
    names = sorted(bounds)
    owner = np.full(shape, -1, dtype=np.int64)
    for i, name in enumerate(names):
        b = bounds[name]
        lo = [coordinates[a].index(int(b[0, a])) for a in range(3)]
        hi = [coordinates[a].index(int(b[1, a])) for a in range(3)]
        cells = owner[tuple(slice(l, h) for l, h in zip(lo, hi))]
        cells[cells < 0] = i
    faces = []
    for index in zip(*np.nonzero(owner >= 0)):
        for axis in range(3):
            j, k = (axis+1) % 3, (axis+2) % 3
            for sign in (-1, 1):
                adjacent = list(index)
                adjacent[axis] += sign
                if 0 <= adjacent[axis] < shape[axis] and owner[tuple(adjacent)] >= 0:
                    continue
                faces.append(dict(axis=axis, plane_um=coordinates[axis][index[axis]+int(sign > 0)],
                                  u0_um=coordinates[j][index[j]], u1_um=coordinates[j][index[j]+1],
                                  v0_um=coordinates[k][index[k]], v1_um=coordinates[k][index[k]+1],
                                  normal_sign=sign, owner_wall_id=names[int(owner[index])]))
    fields = ('axis', 'plane_um', 'normal_sign', 'u0_um', 'u1_um', 'v0_um', 'v1_um', 'owner_wall_id')
    faces.sort(key=lambda f: tuple(f[k] for k in fields))
    return faces


def face_patch(face, *, arm, seed):
    """The patch covering one union face, oriented along its outward normal."""
    axis, sign = face['axis'], face['normal_sign']
    j, k = (axis+1) % 3, (axis+2) % 3
    origin = np.zeros(3)
    origin[axis] = face['plane_um']/UM
    origin[j] = face['u0_um']/UM
    origin[k] = face['v0_um' if sign == 1 else 'v1_um']/UM
    u, v = np.eye(3)[j], sign*np.eye(3)[k]
    lengths = np.array([face['u1_um']-face['u0_um'], face['v1_um']-face['v0_um']])/UM
    return patch(origin, u, v, lengths, arm=arm, seed=seed, neutral=89)


def face_seed(appearance_seed, face):
    identity = json.dumps(dict(appearance_seed=appearance_seed, face=face), sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big')


def surfaces(wall_boxes, arm, seed):
    """[('ground_visual', floor mesh), ('wall_union_visual', wall mesh)] for a maze's wall boxes."""
    if arm not in ARMS or type(seed) is not int or seed < 0:
        raise ValueError('appearance arm and non-negative integer seed required')
    walls = [face_patch(f, arm=arm, seed=face_seed(seed, f)) for f in union_faces(wall_boxes)]
    floor = patch([-FLOOR_HALF_M, -FLOOR_HALF_M, 0], [1, 0, 0], [0, 1, 0], [2*FLOOR_HALF_M, 2*FLOOR_HALF_M], arm=arm, seed=seed)
    return [('ground_visual', floor), ('wall_union_visual', trimesh.util.concatenate(walls))]


def through_ply(mesh, path):
    """Write the mesh as PLY at a new path and read it back (float32 vertices), checking the round trip."""
    with open(path, 'xb') as stream:
        mesh.export(stream, file_type='ply')
    loaded = trimesh.load_mesh(path, file_type='ply', process=False)
    np.testing.assert_array_equal(loaded.vertices, mesh.vertices.astype(np.float32))
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
    np.testing.assert_array_equal(loaded.visual.vertex_colors, mesh.visual.vertex_colors)
    return loaded
