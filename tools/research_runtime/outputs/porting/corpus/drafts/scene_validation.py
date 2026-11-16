"""Reachability validation for generated scenes.

The route teacher plans on the configuration-space inflated grid (see
:mod:`go2nav.corpus.grid`). A scene is *valid for mass data
generation* iff every random spawn pose can navigate to every beacon —
otherwise the rollout produces episodes whose ground-truth solution
depends on which disconnected free-space component the spawn happened
to land in.

This module exposes a single audit function used both by the corpus
builder (to retry seeds that produce fragmented scenes) and by tests
(to assert that planning-grid-based collectors see solvable scenes).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from go2nav.corpus.manifest import SceneManifest
from go2nav.corpus.grid import (
    InflatedOccupancyGrid,
    corridor_width_m,
    safe_standoff_xys,
)
from go2nav.corpus.graph import SceneGraph


# A beacon arrival is only practically claimable if at least one LOS-valid
# standoff sits in a corridor the trained locomotion policy can traverse.
# `local_composite_motifs` seeds frequently place beacons in alcoves whose
# only line-of-sight standoff is in a ~0.30 m corridor (50% of that family's
# beacons had no navigable-width LOS standoff in the 2026-05-19 corpus scan),
# so route-teacher yield caps at ~75% there. This threshold mirrors the
# route teacher's `_STANDOFF_CORRIDOR_SAFE_WIDTH_M`.
NAVIGABLE_STANDOFF_WIDTH_M = 0.50


@dataclass(frozen=True)
class SceneReachabilityReport:
    """Summary of a scene's grid-level reachability properties."""

    is_valid: bool
    beacon_count: int
    component_count: int
    largest_component_size: int
    # The free component that contains every beacon. The spawn sampler
    # should restrict to grid cells in this component so the robot
    # always starts somewhere it can reach all goals. ``-1`` when no
    # such component exists (scene is fragmented across beacon rooms).
    canonical_component_id: int
    unreachable_beacons: tuple[int, ...]
    spawn_candidate_count: int
    spawn_candidates_in_canonical: int
    # Beacons that sit in the canonical component but expose no LOS-valid
    # standoff in a navigable-width corridor — practically unclaimable by
    # the trained locomotion policy even though grid-reachable. Empty
    # unless ``min_navigable_standoffs_per_beacon`` > 0.
    beacons_without_navigable_standoff: tuple[int, ...] = ()
    failure_reason: str = ""


def audit_scene_reachability(
    manifest: SceneManifest,
    *,
    cell_size_m: float = 0.05,
    inflation_m: float = 0.20,
    standoff_m: float = 0.85,
    n_standoff_candidates: int = 12,
    spawn_clearance_floor_m: float = 0.20,
    min_spawn_cells_in_canonical: int = 1,
    min_navigable_standoffs_per_beacon: int = 0,
    navigable_standoff_width_m: float = NAVIGABLE_STANDOFF_WIDTH_M,
) -> SceneReachabilityReport:
    """Return a reachability audit of ``manifest``.

    A scene is *valid* when:

    1. There exists a single grid-free connected component that contains
       at least one LOS-valid standoff for every beacon (the
       *canonical* component).
    2. That component contains at least ``min_spawn_cells_in_canonical``
       spawn-candidate cells, so the spawn sampler can actually drop the
       robot somewhere it can reach every goal.
    3. (Optional, when ``min_navigable_standoffs_per_beacon`` > 0) every
       beacon exposes at least that many LOS-valid standoffs in the
       canonical component whose corridor width is at least
       ``navigable_standoff_width_m``. This rejects alcove beacons whose
       only sightline is through a corridor too narrow for the trained
       locomotion policy — grid-reachable but not practically claimable.

    The spawn sampler is then expected to restrict its picks to the
    canonical component — cells in smaller isolated free regions (e.g.
    closets behind walls) are not used as spawn poses.
    """

    grid = InflatedOccupancyGrid(
        manifest, cell_size_m=cell_size_m, inflation_m=inflation_m
    )
    components = _component_index(grid.free_mask)
    n_components = int(components.max())  # labels are 1..n_components
    sizes = np.bincount(components.ravel())
    largest = int(sizes[1:].max()) if n_components > 0 else 0

    scene = SceneGraph(manifest)
    beacon_components: list[tuple[int, set[int]]] = []
    # Per-beacon: component id -> count of LOS standoffs in a navigable
    # corridor. Only populated when the navigable criterion is enabled.
    beacon_navigable_by_comp: dict[int, dict[int, int]] = {}
    check_navigable = int(min_navigable_standoffs_per_beacon) > 0
    for _name, beacon_cell in scene.landmark_cells:
        beacon_xy = scene.landmark_xy_for_cell(beacon_cell)
        if beacon_xy is None:
            continue
        candidates = safe_standoff_xys(
            grid,
            beacon_xy,
            standoff_m=standoff_m,
            n_candidates=n_standoff_candidates,
        )
        comp_set: set[int] = set()
        nav_by_comp: dict[int, int] = {}
        for standoff in candidates:
            if not scene.has_line_of_sight(
                standoff, beacon_xy, exclude_landmark_xy=beacon_xy
            ):
                continue
            ix, iy = grid.to_grid(standoff)
            nx, ny = grid.shape
            if 0 <= ix < nx and 0 <= iy < ny and components[ix, iy] > 0:
                comp = int(components[ix, iy])
                comp_set.add(comp)
                if check_navigable and corridor_width_m(grid, standoff) >= float(
                    navigable_standoff_width_m
                ):
                    nav_by_comp[comp] = nav_by_comp.get(comp, 0) + 1
        beacon_components.append((int(beacon_cell), comp_set))
        beacon_navigable_by_comp[int(beacon_cell)] = nav_by_comp

    # Canonical component = a single component shared by all beacons.
    # If beacons disagree on which component they sit in, no canonical
    # component exists.
    if beacon_components and all(comps for _, comps in beacon_components):
        candidates = set.intersection(*(comps for _, comps in beacon_components))
    else:
        candidates = set()
    # Among shared components, pick the largest (most spawn options).
    canonical_id = -1
    if candidates:
        canonical_id = max(candidates, key=lambda c: int(sizes[c]))

    unreachable = tuple(
        int(b) for b, comps in beacon_components if canonical_id < 0 or canonical_id not in comps
    )

    spawn_cells = _spawn_candidate_grid_cells(
        manifest, grid, spawn_clearance_floor_m
    )
    spawn_in_canonical = (
        sum(1 for i, j in spawn_cells if int(components[i, j]) == canonical_id)
        if canonical_id > 0
        else 0
    )

    # Beacons that are grid-reachable in the canonical component but have
    # too few navigable-width LOS standoffs there to be practically
    # claimable. Only evaluated when the navigable criterion is enabled
    # and a canonical component exists.
    without_navigable: tuple[int, ...] = ()
    if check_navigable and canonical_id > 0:
        without_navigable = tuple(
            int(b)
            for b, _comps in beacon_components
            if beacon_navigable_by_comp.get(int(b), {}).get(canonical_id, 0)
            < int(min_navigable_standoffs_per_beacon)
        )

    failure = ""
    is_valid = True
    if canonical_id < 0:
        is_valid = False
        failure = "no_component_contains_all_beacons"
    elif spawn_in_canonical < int(min_spawn_cells_in_canonical):
        is_valid = False
        failure = "canonical_component_has_no_spawn_candidates"
    elif without_navigable:
        is_valid = False
        failure = "beacon_lacks_navigable_standoff"

    return SceneReachabilityReport(
        is_valid=is_valid,
        beacon_count=len(manifest.landmarks),
        component_count=int(n_components),
        largest_component_size=largest,
        canonical_component_id=int(canonical_id),
        unreachable_beacons=unreachable,
        spawn_candidate_count=len(spawn_cells),
        spawn_candidates_in_canonical=int(spawn_in_canonical),
        beacons_without_navigable_standoff=without_navigable,
        failure_reason=failure,
    )




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _component_index(free_mask: np.ndarray) -> np.ndarray:
    """Label connected free-space components, 0 = obstacle."""

    nx, ny = free_mask.shape
    labels = np.zeros((nx, ny), dtype=np.int32)
    label = 0
    for i0 in range(nx):
        for j0 in range(ny):
            if not free_mask[i0, j0] or labels[i0, j0] != 0:
                continue
            label += 1
            stack: deque[tuple[int, int]] = deque([(i0, j0)])
            while stack:
                i, j = stack.pop()
                if i < 0 or i >= nx or j < 0 or j >= ny:
                    continue
                if labels[i, j] != 0 or not free_mask[i, j]:
                    continue
                labels[i, j] = label
                stack.append((i + 1, j))
                stack.append((i - 1, j))
                stack.append((i, j + 1))
                stack.append((i, j - 1))
    return labels


def _spawn_candidate_grid_cells(
    manifest: SceneManifest,
    grid: InflatedOccupancyGrid,
    clearance_floor_m: float,
) -> list[tuple[int, int]]:
    """Return grid cells corresponding to room-cell spawn candidates."""

    scene = SceneGraph(manifest)
    out: list[tuple[int, int]] = []
    nx, ny = grid.shape
    for node in manifest.graph_nodes:
        xy = node.center_xy_m
        if scene.clearance_to_walls(xy) < clearance_floor_m:
            continue
        snapped = grid.nearest_free(xy, max_radius_m=4.0 * grid.inflation_m)
        if snapped is None:
            continue
        i, j = grid.to_grid(snapped)
        if 0 <= i < nx and 0 <= j < ny and grid.free_mask[i, j]:
            out.append((int(i), int(j)))
    return out
