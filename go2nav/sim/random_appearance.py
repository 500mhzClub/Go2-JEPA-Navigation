"""Randomised surfaces and lighting for the LeWM corpus's free units: training data only. No mission, benchmark or
evaluation scene uses them; those keep `go2nav.sim.appearance` and the renderer's default light.

Every benchmark maze is tiled in 12.5-cm cells of independent grey (40-230) under one directional light (intensity 5,
from (-1, -1, -1)) with ambient 0.1 and shadows. A model trained only on that learns grey-cell statistics; the free
units vary the look instead. Surfaces stay vertex-coloured cells, which keep crisp corners for the visual tracker.
Walls and floor are drawn independently from these families:
- `grey`: independent grey cells, with a random level range (contrast and brightness) and cell size;
- `tinted`: grey cells times a random colour;
- `colour`: cells of random hues around a random base hue, random saturation and value;
- `checker`: two random colours alternating;
- `texture`: each cell the colour of a CC0 photo texture (assets/textures) at that place, the photo tiled every
  0.5-2 m: walls Concrete045, PaintedPlaster017, Plaster001; floors Concrete034, PavingStones131, WoodFloor043.
Cell sizes: 6.25, 12.5 or 25 cm on walls, 12.5 or 25 cm on the floor (the floor spans 32 x 32 m).

Lighting: one directional light, 20-80 degrees above the horizon from a uniform azimuth, intensity 3-8, colour from
warm to cool white; ambient grey 0.03-0.25; shadows on 4 times in 5. The benchmark's light (intensity 5, ambient 0.1)
lies inside these ranges.

**Held out** (`held_out=True`, the transfer units): walls Bricks097 and floor Tiles093 (never in training),
under a dim light (intensity 1.5-2.5) with a strong blue or orange cast. Training units never draw these.
"""
import colorsys
import functools
import math
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from go2nav.sim.appearance import FLOOR_HALF_M, UM, union_faces

REPO = Path(__file__).resolve().parents[2]
TEXTURES = REPO/'assets/textures'
WALL_TEXTURES = ('Concrete045', 'PaintedPlaster017', 'Plaster001')
FLOOR_TEXTURES = ('Concrete034', 'PavingStones131', 'WoodFloor043')
HELD_OUT = dict(wall='Bricks097', floor='Tiles093')
FAMILIES = ('grey', 'tinted', 'colour', 'checker', 'texture')
WALL_CELLS, FLOOR_CELLS = (.0625, .125, .25), (.125, .25)


def _surface(rng, role, held_out=False):
    cell = float(rng.choice(WALL_CELLS if role == 'wall' else FLOOR_CELLS))
    if held_out:
        return dict(family='texture', texture=HELD_OUT[role], cell_m=cell, tile_m=float(rng.uniform(.5, 2.)), seed=int(rng.integers(2**31)))
    family = str(rng.choice(FAMILIES))
    out = dict(family=family, cell_m=cell, seed=int(rng.integers(2**31)))
    if family in ('grey', 'tinted'):
        lo = int(rng.integers(0, 120))
        out.update(low=lo, high=int(rng.integers(lo+60, 256)))
        if family == 'tinted':
            out['tint'] = [float(v) for v in colorsys.hsv_to_rgb(rng.random(), rng.uniform(.2, .8), 1.)]
    elif family == 'colour':
        out.update(hue=float(rng.random()), hue_spread=float(rng.uniform(.03, .3)), saturation=[float(v) for v in sorted(rng.uniform(.1, .9, 2))],
                   value=[float(v) for v in sorted(rng.uniform(.2, 1., 2))])
    elif family == 'checker':
        out['colours'] = [[int(c) for c in rng.integers(0, 256, 3)] for _ in range(2)]
    else:
        out.update(texture=str(rng.choice(WALL_TEXTURES if role == 'wall' else FLOOR_TEXTURES)), tile_m=float(rng.uniform(.5, 2.)))
    return out


def sample(seed, *, held_out=False):
    """An appearance (walls, floor, lighting) drawn from `seed`: the training ranges, or the held-out ones."""
    rng = np.random.default_rng(seed)
    walls, floor = _surface(rng, 'wall', held_out), _surface(rng, 'floor', held_out)
    azimuth, elevation = rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(20, 80))
    direction = [-math.cos(elevation)*math.cos(azimuth), -math.cos(elevation)*math.sin(azimuth), -math.sin(elevation)]
    if held_out:
        cast = (1., .55, .3) if rng.random() < .5 else (.35, .55, 1.)
        lighting = dict(dir=direction, color=list(cast), intensity=float(rng.uniform(1.5, 2.5)), ambient=float(rng.uniform(.03, .1)),
                        shadow=True)
    else:
        warmth = rng.uniform(-1, 1)  # -1 cool, +1 warm
        color = [1., 1.-.08*abs(warmth), 1.-.2*max(warmth, 0)] if warmth > 0 else [1.-.2*abs(warmth), 1.-.08*abs(warmth), 1.]
        lighting = dict(dir=direction, color=color, intensity=float(rng.uniform(3, 8)), ambient=float(rng.uniform(.03, .25)),
                        shadow=bool(rng.random() < .8))
    return dict(schema='go2nav_random_appearance.v1', held_out=held_out, walls=walls, floor=floor, lighting=lighting)


@functools.lru_cache(maxsize=None)
def _texture(name, pixels=256):
    path = next((TEXTURES/kind/f'{name}.jpg' for kind in ('wall', 'floor', 'obstacle') if (TEXTURES/kind/f'{name}.jpg').exists()), None)
    if path is None:
        raise FileNotFoundError(f'texture {name} not in assets/textures')
    return np.asarray(Image.open(path).convert('RGB').resize((pixels, pixels), Image.Resampling.LANCZOS), np.float32)


def _colours(surface, centres_uv, count):
    """uint8 RGB per cell; `centres_uv` (count, 2) are cell centres in the patch plane (m), for textures/checkers."""
    rng = np.random.default_rng(surface['seed'])
    family = surface['family']
    if family in ('grey', 'tinted'):
        grey = rng.integers(surface['low'], surface['high']+1, count).astype(np.float32)
        rgb = grey[:, None]*np.asarray(surface.get('tint', (1., 1., 1.)), np.float32)[None]
    elif family == 'colour':
        h = (surface['hue']+rng.uniform(-surface['hue_spread'], surface['hue_spread'], count)) % 1.
        sat, val = rng.uniform(*surface['saturation'], count), rng.uniform(*surface['value'], count)
        rgb = 255.*np.array([colorsys.hsv_to_rgb(*c) for c in zip(h, sat, val)], np.float32).reshape(count, 3)
    elif family == 'checker':
        index = np.floor(centres_uv/surface['cell_m']).astype(np.int64).sum(axis=1) % 2
        rgb = np.asarray(surface['colours'], np.float32)[index]
    else:
        image = _texture(surface['texture'])
        uv = (centres_uv/surface['tile_m']) % 1.
        rgb = image[(uv[:, 1]*(len(image)-1)).astype(int), (uv[:, 0]*(len(image)-1)).astype(int)]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def patch(origin, u, v, lengths, surface, offset_uv=(0., 0.)):
    """A rectangle origin + s u + t v tiled in `surface`'s cells (two triangles and four vertices per cell)."""
    origin, u, v, lengths = (np.asarray(x, float) for x in (origin, u, v, lengths))
    cell = surface['cell_m']
    n, m = np.maximum(np.ceil(lengths/cell).astype(int), 1)
    x, y = np.meshgrid(np.linspace(0, lengths[0], n+1), np.linspace(0, lengths[1], m+1))
    p = origin+x[..., None]*u+y[..., None]*v
    quads = np.stack((p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]), axis=2).reshape(-1, 4, 3)
    centres = np.stack(((x[:-1, :-1]+x[1:, 1:])/2, (y[:-1, :-1]+y[1:, 1:])/2), axis=-1).reshape(-1, 2)+np.asarray(offset_uv)
    vertices = quads.reshape(-1, 3)
    base = np.arange(len(quads))*4
    faces = np.stack((base[:, None]+[0, 1, 2], base[:, None]+[0, 2, 3]), axis=1).reshape(-1, 3)
    rgb = np.repeat(_colours(surface, centres, len(quads)), 4, axis=0)
    color = np.column_stack((rgb, np.full(len(vertices), 255, np.uint8)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=color, process=False)


def surfaces(wall_boxes, appearance):
    """[('ground_visual', floor mesh), ('wall_union_visual', wall mesh)], as `appearance.surfaces` returns them."""
    walls = []
    for i, face in enumerate(union_faces(wall_boxes)):
        axis, sign = face['axis'], face['normal_sign']
        j, k = (axis+1) % 3, (axis+2) % 3
        origin = np.zeros(3)
        origin[axis] = face['plane_um']/UM
        origin[j] = face['u0_um']/UM
        origin[k] = face['v0_um' if sign == 1 else 'v1_um']/UM
        lengths = np.array([face['u1_um']-face['u0_um'], face['v1_um']-face['v0_um']])/UM
        surface = appearance['walls'] | dict(seed=appearance['walls']['seed']+i)
        walls.append(patch(origin, np.eye(3)[j], sign*np.eye(3)[k], lengths, surface,
                           offset_uv=(face['u0_um']/UM, face['v0_um']/UM)))
    floor = patch([-FLOOR_HALF_M, -FLOOR_HALF_M, 0], [1, 0, 0], [0, 1, 0], [2*FLOOR_HALF_M, 2*FLOOR_HALF_M], appearance['floor'])
    return [('ground_visual', floor), ('wall_union_visual', trimesh.util.concatenate(walls))]


def vis_options(lighting):
    """Genesis visual options for a lighting record."""
    import genesis as gs
    from genesis.options.vis import DirectionalLight
    return gs.options.VisOptions(lights=[DirectionalLight(dir=tuple(lighting['dir']), color=tuple(lighting['color']),
                                                           intensity=lighting['intensity'])],
                                 ambient_light=(lighting['ambient'],)*3, shadow=lighting['shadow'])
