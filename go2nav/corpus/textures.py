"""Render-time surface textures (CC0 diffuse maps) — data-spec §14 visuals.

This module is pure Python (no ``genesis`` import) so it can be unit-tested
without the simulator. It does two jobs:

1. **Selection** — map a static object's ``kind`` to a texture *category*
   (floor / wall / obstacle) and deterministically pick one CC0 diffuse map
   per category per scene, keyed by the scene's ``visual_seed`` + ``scene_id``
   so re-renders are reproducible (data-spec §15). Landmarks and visual-stress
   distractors are deliberately left untextured — their solid colors are task
   identity / decoys, not background visual (see ``randomization.py``).

2. **Geometry** — Genesis box *primitives* cannot carry a texture (their
   meshes have no usable UVs), so wall/obstacle boxes are rebuilt as a small
   UV-mapped cube OBJ whose per-face UVs are scaled to the box dimensions, so
   one texel ~= ``1 / tiles_per_m`` metres and the map tiles instead of
   stretching. OBJs are cached on disk keyed by (size, tiling).

Textures are vendored under ``<repo>/assets/textures/<category>/`` (CC0, see
that directory's README). Set ``GO2NAV_TEXTURE_ROOT`` to override the location.
"""

from __future__ import annotations

import os
import random
import tempfile
from functools import lru_cache
from pathlib import Path

CATEGORIES: tuple[str, ...] = ("floor", "wall", "obstacle")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_DEFAULT_TILES_PER_M = 0.7  # one texture repeat roughly every ~1.4 m


def texture_root() -> Path:
    override = os.environ.get("GO2NAV_TEXTURE_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "assets" / "textures"


@lru_cache(maxsize=None)
def available_textures(category: str) -> tuple[str, ...]:
    """Sorted absolute paths of vendored maps for a category (empty if none)."""

    root = texture_root() / category
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            str(p)
            for p in root.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS and p.is_file()
        )
    )


# Static-object ``kind`` -> texture category. Only structural surfaces are
# textured. Omitted kinds keep their solid color: ``landmark`` (navigation
# target identity), ``distractor`` (false-positive decoy identity), and
# ``slick_patch`` (a deliberate visual marker of a low-friction floor patch).
_KIND_TO_CATEGORY: dict[str, str] = {
    "wall": "wall",
    "box_obstacle": "obstacle",
    "ramp": "obstacle",
    "step": "obstacle",
}


def category_for_kind(kind: str) -> str | None:
    """Texture category for a static object's ``kind`` (``None`` => solid color)."""

    return _KIND_TO_CATEGORY.get((kind or "").strip().lower())


def select_scene_textures(*, visual_seed: int, scene_id: str) -> dict[str, str | None]:
    """Deterministically pick one diffuse map per category for a scene.

    One texture per category per scene gives each environment a coherent look
    (all walls share a material, etc.) while varying across scenes. Returns
    ``None`` for a category with no vendored assets so callers fall back to the
    solid-color surface and never crash on a missing asset dir.
    """

    rng = random.Random(f"texture:{int(visual_seed)}:{scene_id}")
    selection: dict[str, str | None] = {}
    for category in CATEGORIES:
        options = available_textures(category)
        selection[category] = rng.choice(options) if options else None
    return selection


# ---------------------------------------------------------------------------
# UV-mapped cube mesh (so wall/obstacle boxes can be textured)
# ---------------------------------------------------------------------------




def box_obj_text(size_xyz_m: tuple[float, float, float], *, tiles_per_m: float) -> str:
    """OBJ text for an axis-aligned box centered at origin with tiling UVs.

    Vertices are authored in the box's local frame in metres (Z up). Each of
    the six faces gets its own 4 vertices / 4 UVs / 1 normal; UVs span the
    face's two in-plane dimensions times ``tiles_per_m`` so the texture repeats
    across large faces instead of stretching.
    """

    sx, sy, sz = (float(s) for s in size_xyz_m)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    tpm = float(tiles_per_m)

    # (corner quad CCW, normal, (u_extent, v_extent)) per face.
    faces = [
        ([(+hx, -hy, -hz), (+hx, +hy, -hz), (+hx, +hy, +hz), (+hx, -hy, +hz)], (1, 0, 0), (sy, sz)),
        ([(-hx, +hy, -hz), (-hx, -hy, -hz), (-hx, -hy, +hz), (-hx, +hy, +hz)], (-1, 0, 0), (sy, sz)),
        ([(+hx, +hy, -hz), (-hx, +hy, -hz), (-hx, +hy, +hz), (+hx, +hy, +hz)], (0, 1, 0), (sx, sz)),
        ([(-hx, -hy, -hz), (+hx, -hy, -hz), (+hx, -hy, +hz), (-hx, -hy, +hz)], (0, -1, 0), (sx, sz)),
        ([(-hx, -hy, +hz), (+hx, -hy, +hz), (+hx, +hy, +hz), (-hx, +hy, +hz)], (0, 0, 1), (sx, sy)),
        ([(-hx, +hy, -hz), (+hx, +hy, -hz), (+hx, -hy, -hz), (-hx, -hy, -hz)], (0, 0, -1), (sx, sy)),
    ]

    lines = ["# lewm textured box (z-up, metres)"]
    vt_lines: list[str] = []
    vn_lines: list[str] = []
    f_lines: list[str] = []
    vi = 1
    for corners, normal, (u_ext, v_ext) in faces:
        uvs = [(0.0, 0.0), (u_ext * tpm, 0.0), (u_ext * tpm, v_ext * tpm), (0.0, v_ext * tpm)]
        for (x, y, z) in corners:
            lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
        for (u, v) in uvs:
            vt_lines.append(f"vt {u:.5f} {v:.5f}")
        vn_lines.append(f"vn {normal[0]} {normal[1]} {normal[2]}")
        ni = len(vn_lines)
        a, b, c, d = vi, vi + 1, vi + 2, vi + 3
        f_lines.append(f"f {a}/{a}/{ni} {b}/{b}/{ni} {c}/{c}/{ni}")
        f_lines.append(f"f {a}/{a}/{ni} {c}/{c}/{ni} {d}/{d}/{ni}")
        vi += 4
    return "\n".join(lines + vt_lines + vn_lines + f_lines) + "\n"


def cached_box_obj(
    size_xyz_m: tuple[float, float, float],
    *,
    tiles_per_m: float = _DEFAULT_TILES_PER_M,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Path to a cached UV box OBJ for this size+tiling, writing it if absent.

    Sizes repeat heavily (uniform walls), so caching by rounded dimensions
    keeps the OBJ count tiny. Writes are atomic (temp + rename) so concurrent
    render workers don't race.
    """

    sx, sy, sz = (round(float(s), 3) for s in size_xyz_m)
    root = Path(cache_dir) if cache_dir is not None else (
        Path(os.environ.get("GO2NAV_CACHE", Path.home()/".cache/go2nav")) / "box_meshes"
    )
    root.mkdir(parents=True, exist_ok=True)
    name = f"box_{sx:.3f}x{sy:.3f}x{sz:.3f}_t{float(tiles_per_m):.2f}.obj"
    path = root / name
    if not path.is_file():
        text = box_obj_text((sx, sy, sz), tiles_per_m=tiles_per_m)
        fd, tmp = tempfile.mkstemp(dir=str(root), suffix=".obj.tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return str(path)
