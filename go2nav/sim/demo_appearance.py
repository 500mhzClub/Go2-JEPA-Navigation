"""Demonstration surfaces: brickwork walls and a parquet floor in place of the random-grey cells (presentation only).

A mission run with `--demo-textures` (`go2nav.eval.mission`) draws its two visual surfaces with these meshes instead of
`go2nav.sim.appearance`; geometry, collision, depth, the draw order and everything else are unchanged. The models were
never trained on these surfaces, and demonstration videos must say so.

- **Walls:** running-bond brickwork (21.5 x 6.5 cm bricks, 1-cm mortar joints) on every vertical face of the wall
  union, each brick its own quad with a colour sampled from the CC0 texture Bricks097 and mortar from its joints;
  light stone caps on the tops.
- **Floor:** wood-strip parquet (45 x 9 cm strips, 6-mm dark seams, each row at a random offset) over the maze's
  bounding box plus 1 m, each strip shaded along its length from WoodFloor043; a coarse surround keeps the floor's
  exact [-16, 16] m extent (the scene's raster order identifies the floor by it).

Every brick, strip, joint and seam has its own vertices, so edges stay crisp: the visual tracker needs corners spread
over the image (smoothly sampled texture colours gave a tenth of the keypoints and lost tracking within 5 s).
"""
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from go2nav.sim.appearance import UM, union_faces

REPO = Path(__file__).resolve().parents[2]
FLOOR_TEXTURE = REPO/'assets/textures/floor/WoodFloor043.jpg'
WALL_TEXTURE = REPO/'assets/textures/wall/Bricks097.jpg'
SEED = 20261005
BRICK = dict(length=.225, height=.075, joint=.010)
STRIP = dict(length=.45, width=.09, seam=.006, segments=3)


def _texture(path, pixels=256):
    return np.asarray(Image.open(path).convert('RGB').resize((pixels, pixels), Image.Resampling.LANCZOS), np.float32)


def palettes(rng):
    """Brick and mortar colours from Bricks097; wood colours along the grain from WoodFloor043."""
    bricks = _texture(WALL_TEXTURE)
    n = bricks.shape[0]
    centres = rng.integers(3, n-3, size=(4000, 2))
    patches = np.array([bricks[i-2:i+3, j-2:j+3].reshape(-1, 3).mean(0) for i, j in centres])
    red = patches[(patches[:, 0]-patches[:, 2] > 30) & (patches.sum(1) < 450)]
    flat = bricks.reshape(-1, 3)
    light = flat[flat.sum(1) > np.quantile(flat.sum(1), .85)]
    wood = _texture(FLOOR_TEXTURE)
    rows = rng.integers(0, wood.shape[0], size=512)
    cols = rng.integers(0, wood.shape[1]-40, size=512)
    grain = np.stack([wood[r, c:c+40:13][:STRIP['segments']+1] for r, c in zip(rows, cols)])
    return red[:512], light.mean(0), grain


class Quads:
    """Coloured quads with unshared vertices (per-vertex colours), as one mesh."""

    def __init__(self):
        self.points, self.colours = [], []

    def add(self, corners, colours):
        self.points.append(np.asarray(corners, float))
        self.colours.append(np.broadcast_to(np.asarray(colours, float), (4, 3)))

    def mesh(self):
        points, n = np.concatenate(self.points), len(self.points)
        base = 4*np.arange(n)[:, None]
        faces = np.concatenate((base+[0, 1, 2], base+[0, 2, 3]))
        rgba = np.concatenate((np.clip(np.concatenate(self.colours), 0, 255), np.full((4*n, 1), 255.)), 1).astype(np.uint8)
        return trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=rgba, process=False)


def _bond(s0, s1, z0, z1, *, length, height, joint, offsets):
    """Running bond (or strip rows) over [s0, s1] x [z0, z1]: yields (rect, kind, course, unit), rect = (sa, sb, za, zb)."""
    for course in range(int(np.ceil((z1-z0)/height-1e-9))):
        za, zb = z0+course*height, min(z0+(course+1)*height, z1)
        yield (s0, s1, za, min(za+joint, zb)), 'joint', course, 0
        if za+joint >= zb:
            continue
        offset = offsets(course)
        for unit in range(int(np.floor((s0-offset)/length)), int(np.ceil((s1-offset)/length))):
            a = offset+unit*length
            for (sa, sb), kind in (((a, a+joint), 'joint'), ((a+joint, a+length), 'unit')):
                sa, sb = max(sa, s0), min(sb, s1)
                if sb-sa > 1e-6:
                    yield (sa, sb, za+joint, zb), kind, course, unit


def surfaces(boxes):
    """[('ground_visual', parquet floor), ('wall_union_visual', brick walls)] for a maze's wall boxes."""
    rng = np.random.default_rng(SEED)
    brick, mortar, grain = palettes(rng)
    shade = rng.uniform(.85, 1.12, size=4096)
    xs, ys = [b['centre_xyz'][0] for b in boxes], [b['centre_xyz'][1] for b in boxes]
    x0, x1, y0, y1 = min(xs)-1., max(xs)+1., min(ys)-1., max(ys)+1.
    row_offsets = rng.uniform(0, STRIP['length'], size=4096)
    seam = grain.reshape(-1, 3).mean(0)*.35
    floor, segments = Quads(), STRIP['segments']
    for (sa, sb, za, zb), kind, row, unit in _bond(x0, x1, y0, y1, length=STRIP['length'], height=STRIP['width'],
                                                    joint=STRIP['seam'], offsets=lambda r: row_offsets[r % 4096]):
        if kind == 'joint':
            floor.add([[sa, za, 0], [sb, za, 0], [sb, zb, 0], [sa, zb, 0]], seam)
            continue
        h = (row*7919+unit*104729) % 4096
        colours = grain[h % len(grain)]*shade[h]
        cuts = np.linspace(sa, sb, segments+1)
        for i in range(segments):
            floor.add([[cuts[i], za, 0], [cuts[i+1], za, 0], [cuts[i+1], zb, 0], [cuts[i], zb, 0]],
                      [colours[i], colours[i+1], colours[i+1], colours[i]])
    tone = grain.reshape(-1, 3).mean(0)*.6
    for a0, a1, b0, b1 in ((-16, 16, -16, y0), (-16, 16, y1, 16), (-16, x0, y0, y1), (x1, 16, y0, y1)):
        floor.add([[a0, b0, 0], [a1, b0, 0], [a1, b1, 0], [a0, b1, 0]], tone)
    walls = Quads()
    for face in union_faces(boxes):
        axis, sign = face['axis'], face['normal_sign']
        j, k = (axis+1) % 3, (axis+2) % 3
        origin = np.zeros(3)
        origin[axis], origin[j] = face['plane_um']/UM, face['u0_um']/UM
        origin[k] = face['v0_um' if sign == 1 else 'v1_um']/UM
        u, v = np.eye(3)[j], sign*np.eye(3)[k]
        uj, vk = np.array([face['u0_um'], face['u1_um']])/UM, np.array([face['v0_um'], face['v1_um']])/UM

        def emit(cj, ck, colour):
            """A world rectangle cj x ck on this face, as a quad in the face's own (u, v) orientation (same winding)."""
            a, b = sorted(c-origin[j] for c in cj), sorted((c-origin[k])*sign for c in ck)
            walls.add([origin+a[0]*u+b[0]*v, origin+a[1]*u+b[0]*v, origin+a[1]*u+b[1]*v, origin+a[0]*u+b[1]*v], colour)
        if axis == 2:  # a top: stone cap
            emit(uj, vk, mortar*.92)
            continue
        # axis 0: (u, v) = (y, z); axis 1: (u, v) = (z, x). s is the horizontal coordinate along the face.
        s_range, z_range = (uj, vk) if axis == 0 else (vk, uj)
        for (sa, sb, za, zb), kind, course, unit in _bond(*s_range, *z_range,
                                                        offsets=lambda c: (c % 2)*BRICK['length']/2, **BRICK):
            if kind == 'joint':
                colour = mortar
            else:
                h = (course*7919+unit*104729+axis*15485863) % 4096
                colour = brick[h % len(brick)]*shade[h]
            if axis == 0:
                emit((sa, sb), (za, zb), colour)
            else:
                emit((za, zb), (sa, sb), colour)
    return [('ground_visual', floor.mesh()), ('wall_union_visual', walls.mesh())]
