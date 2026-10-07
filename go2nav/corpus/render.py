"""Render a scene's camera frames from its render plan (the May v03 render: genesis-world 0.3.14, Vulkan backend).

    PYTHONPATH=. EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl RENDER_VENV/bin/python -m go2nav.corpus.render \
        --plan PLAN.json --scene-corpus CORPUS --out OUT [--resolution 224] [--textures] [--max-frames N]
        [--frames LIST]

Replays every recorded camera pose in the plan's frames.jsonl (`go2nav.corpus.render_plan`) over the scene's static
geometry, one environment at a time, and writes OUT/rgb/frame_<frame>_env_<env>.png plus a summary. Surfaces take
the manifest's material colours and the landmark palette; --textures applies the scene's deterministic CC0 textures
(`go2nav.corpus.textures`), as the May render did. --frames renders only the frame files named in LIST (one file
name per line, e.g. frame_000246_env_06.png); each frame is rendered independently, so a subset matches the full
render. The rollouts ran on Genesis 0.4.6; the render runs in a separate
Genesis 0.3.14 environment, and EGL_DEVICE_ID selects the GPU that renders (device 1 on the development machine).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from go2nav.corpus.textures import cached_box_obj, category_for_kind, select_scene_textures

# Landmark beacon palette (the colored goal markers) — used when a landmark's
# material_id has no explicit override.
_LANDMARK_PALETTE = {
    "landmark_red": (0.85, 0.13, 0.13, 1.0),
    "landmark_blue": (0.13, 0.22, 0.85, 1.0),
    "landmark_green": (0.13, 0.70, 0.22, 1.0),
    "landmark_yellow": (0.92, 0.82, 0.10, 1.0),
    "landmark_magenta": (0.80, 0.13, 0.75, 1.0),
    "landmark_cyan": (0.13, 0.78, 0.82, 1.0),
}
_DEFAULT_WALL = (0.52, 0.52, 0.52, 1.0)
_DEFAULT_OBSTACLE = (0.58, 0.46, 0.38, 1.0)
_DEFAULT_FLOOR = (0.42, 0.42, 0.42, 1.0)
_DEFAULT_LANDMARK = (0.85, 0.13, 0.13, 1.0)


def _color_map(manifest: dict) -> dict:
    out: dict = {}
    vr = manifest.get("visual_randomization") or {}
    for ov in vr.get("material_overrides") or []:
        mid = ov.get("material_id")
        if mid and ov.get("rgba"):
            out[mid] = tuple(ov["rgba"])
    return out


def _surface(gs, rgba):
    if rgba is None:
        return None
    try:
        return gs.surfaces.Default(color=tuple(rgba))
    except Exception:
        return None


def _texture_surface(gs, image_path: str | None):
    if not image_path:
        return None
    try:
        return gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=str(image_path),
                encoding="srgb",
            )
        )
    except Exception:
        return None


def _euler_deg(o: dict):
    return (
        math.degrees(float(o.get("roll_rad", 0.0))),
        math.degrees(float(o.get("pitch_rad", 0.0))),
        math.degrees(float(o.get("yaw_rad", 0.0))),
    )


def build_scene(
    gs,
    manifest: dict,
    *,
    fov: float,
    near: float,
    far: float,
    res,
    textures: bool = False,
):
    cmap = _color_map(manifest)
    scene = gs.Scene(show_viewer=False)
    scene_textures = (
        select_scene_textures(
            visual_seed=int(manifest.get("visual_seed") or 0),
            scene_id=str(manifest.get("scene_id") or ""),
        )
        if textures
        else {}
    )

    floor_surface = (
        _texture_surface(gs, scene_textures.get("floor"))
        or _surface(gs, cmap.get("floor", _DEFAULT_FLOOR))
    )
    if floor_surface is not None:
        scene.add_entity(gs.morphs.Plane(), surface=floor_surface)
    else:
        scene.add_entity(gs.morphs.Plane())

    def add_box(o: dict, default_rgba):
        mid = o.get("material_id", "")
        rgba = cmap.get(mid) or _LANDMARK_PALETTE.get(mid) or default_rgba
        tex_surface = None
        if textures:
            category = category_for_kind(str(o.get("kind") or ""))
            tex_surface = _texture_surface(gs, scene_textures.get(category))
        surf = tex_surface or _surface(gs, rgba)
        if tex_surface is not None:
            morph = gs.morphs.Mesh(
                file=cached_box_obj(tuple(float(v) for v in o["size_xyz_m"])),
                pos=tuple(o["center_xyz_m"]),
                euler=_euler_deg(o),
                fixed=True,
                collision=True,
                convexify=True,
                file_meshes_are_zup=True,
            )
        else:
            morph = gs.morphs.Box(
                pos=tuple(o["center_xyz_m"]),
                size=tuple(o["size_xyz_m"]),
                euler=_euler_deg(o),
                fixed=True,
            )
        if surf is not None:
            scene.add_entity(morph, surface=surf)
        else:
            scene.add_entity(morph)

    for w in manifest.get("walls", []) or []:
        add_box(w, _DEFAULT_WALL)
    for ob in manifest.get("obstacles", []) or []:
        add_box(ob, _DEFAULT_OBSTACLE)
    for lm in manifest.get("landmarks", []) or []:
        add_box(lm, _DEFAULT_LANDMARK)

    cam = scene.add_camera(
        res=tuple(res), pos=(0.0, 0.0, 1.0), lookat=(1.0, 0.0, 1.0),
        fov=float(fov), near=float(near), far=float(far), GUI=False,
    )
    scene.build(n_envs=1)
    return scene, cam


def _to_hwc_uint8(rgb):
    import numpy as np
    arr = np.asarray(rgb)
    while arr.ndim > 3 and arr.shape[0] == 1:  # (1,H,W,C) -> (H,W,C)
        arr = arr[0]
    if arr.ndim == 4:  # batched but >1 leading: take env 0
        arr = arr[0]
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
    return arr


def _find_manifest(corpus: Path, plan: dict) -> Path:
    sid = plan["scene_id"]
    split = plan.get("split")
    fam = plan.get("scene_family") or plan.get("family")
    if split and fam:
        cand = corpus / split / fam / sid / "manifest.json"
        if cand.is_file():
            return cand
    # fall back to a glob by scene_id
    hits = list(corpus.glob(f"*/*/{sid}/manifest.json"))
    if not hits:
        raise FileNotFoundError(f"manifest for {sid} not found under {corpus}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--scene-corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--frames", type=Path, help="render only the frame files named in this list")
    ap.add_argument(
        "--textures",
        action="store_true",
        help="Use deterministic CC0 diffuse textures on structural surfaces.",
    )
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    sid = plan["scene_id"]
    cam_d = plan.get("camera", {}) or {}
    fov = float(cam_d.get("fov_deg", 78.0))
    near = float(cam_d.get("near_m") or cam_d.get("near") or 0.05)
    far = float(cam_d.get("far_m") or cam_d.get("far") or 200.0)
    frames_jsonl = Path(plan["frames_jsonl"])
    manifest = json.loads(_find_manifest(args.scene_corpus, plan).read_text())

    import genesis as gs
    from PIL import Image
    gs.init(backend=gs.vulkan, logging_level="error")
    res = (int(args.resolution), int(args.resolution))
    scene, cam = build_scene(
        gs,
        manifest,
        fov=fov,
        near=near,
        far=far,
        res=res,
        textures=bool(args.textures),
    )

    rgb_dir = args.out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    t0 = time.time()
    wanted = set(args.frames.read_text().split()) if args.frames else None
    with frames_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fr = json.loads(line)
            name = f"frame_{int(fr['frame_index']):06d}_env_{int(fr.get('env_index', 0)):02d}.png"
            if wanted is not None and name not in wanted:
                continue
            cp = fr.get("camera_pose_world") or {}
            pos = tuple(cp.get("position", (0.0, 0.0, 1.0)))
            lookat = tuple(cp.get("lookat", (1.0, 0.0, 1.0)))
            up = tuple(cp.get("up", (0.0, 0.0, 1.0)))
            cam.set_pose(pos=pos, lookat=lookat, up=up)
            out = cam.render(rgb=True, depth=False)
            rgb = out[0] if isinstance(out, (tuple, list)) else out
            arr = _to_hwc_uint8(rgb)
            Image.fromarray(arr).save(rgb_dir / name)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    dt = time.time() - t0

    summary = {
        "schema": "lewm_rendered_vision_v03",
        "render_status": "complete",
        "scene_id": sid,
        "split": plan.get("split"),
        "family": plan.get("scene_family"),
        "frame_count": n,
        "resolution": int(args.resolution),
        "renderer": "genesis-0.3.14/vulkan",
        "visuals": "textured_v03" if args.textures else "material_color",
        "textures_enabled": bool(args.textures),
        "fps": (n / dt) if dt > 0 else 0.0,
        "plan": str(args.plan),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"RENDER_OK {sid} frames={n} fps={n/dt:.1f}" if dt > 0 else f"RENDER_OK {sid} frames={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
