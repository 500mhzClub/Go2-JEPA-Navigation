"""Go2 asset resolution helpers for Genesis rendering."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


GENESIS_BUILTIN_GO2_URDF = "genesis_builtin_go2"


def resolve_go2_urdf(platform_manifest: dict[str, Any], workspace_root: Path) -> Path:
    """Resolve the Go2 URDF/Xacro path named by the platform manifest."""

    root = Path(workspace_root)
    robot = platform_manifest.get("robot", {})
    urdf_values = [
        robot.get("genesis_urdf"),
        robot.get("urdf_xacro"),
        "third_party/unitree_go2_ros2/unitree_go2_description/urdf/unitree_go2_robot.xacro",
    ]
    requested = [str(value) for value in urdf_values if value]
    for urdf_value in requested:
        try:
            candidate = _resolve_urdf_value(urdf_value, root)
        except FileNotFoundError:
            continue
        if candidate.is_file():
            return candidate

    upstream = platform_manifest.get("robot", {}).get("upstream_base", {})
    local_path = root / str(upstream.get("local_path", "third_party/unitree_go2_ros2"))
    package_path = str(platform_manifest.get("robot", {}).get("description_package", "unitree_go2_description"))
    fallback = local_path / package_path / "urdf" / "unitree_go2_robot.xacro"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"could not resolve Go2 URDF from manifest paths: {requested}")


def _resolve_urdf_value(urdf_value: str, workspace_root: Path) -> Path:
    if urdf_value == GENESIS_BUILTIN_GO2_URDF:
        return _resolve_genesis_builtin_go2_urdf()
    return Path(workspace_root) / urdf_value


def _resolve_genesis_builtin_go2_urdf() -> Path:
    """Locate the Go2 URDF bundled in the installed ``genesis`` package."""

    spec = importlib.util.find_spec("genesis")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "robot.genesis_urdf is genesis_builtin_go2, but the genesis package is not importable"
        )
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    return package_dir / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"
