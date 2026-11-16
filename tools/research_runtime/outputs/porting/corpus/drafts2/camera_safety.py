"""Egocentric camera placement guards for wall-heavy scenes.

The renderer keeps the platform camera mount as the nominal pose, then applies
the same safety idea used in LeWMQuad-v2: cast a few frustum rays against scene
boxes and retract the optical centre along ``-forward`` only when the nominal
camera would clip into or through geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from lewm_genesis.lewm_contract import rotate_body_to_world


BODY_ENVELOPE_FORWARD_M = 0.06
DEFAULT_MAX_RETRACT_M = 0.08
DEFAULT_INSIDE_MARGIN_M = 0.02
RETRACT_MARGIN_M = 0.005


@dataclass(frozen=True)
class CameraSafetyConfig:
    safe_clearance_m: float
    near_plane_m: float
    fov_deg: float
    fov_axis: str = "horizontal"
    aspect_ratio: float = 4.0 / 3.0
    lookat_dist_m: float = 1.0
    inside_margin_m: float = DEFAULT_INSIDE_MARGIN_M
    max_retract_m: float = DEFAULT_MAX_RETRACT_M


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray
    lookat: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    rotation: np.ndarray



# Static object kinds whose meshes do NOT occlude the camera. Empty by
# default: beacons (kind="landmark") used to be excluded so the lens could
# pass through them in tight motif scenes, but that produced training frames
# of the camera staring at the inside of a beacon mesh. We now treat
# landmarks as occluders too — the route teacher's standoff logic keeps the
# robot out of beacon cells, and the rare collector-driven beacon-cell
# occupancy correctly marks those frames invalid.
NON_OCCLUDING_KINDS: frozenset[str] = frozenset()










































