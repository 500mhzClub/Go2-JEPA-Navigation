"""Configuration-space occupancy grid + A* for privileged navigation.

The route teacher (and any future privileged planner) needs to know which
world-frame xy points the robot body can safely occupy, and which xy paths
connect two safe points. The coarse room-cell graph on :class:`SceneGraph`
does neither well: it has one node per ~1.2 m corridor cell, and the
clearance test runs only at each cell's center point, so a 0.30 m beacon
that takes up half a cell silently makes the cell unreachable.

This module replaces that mechanism with a config-space planner:

1. Rasterize every wall, obstacle, and beacon AABB into a high-resolution
   2D grid (default 5 cm) sized to the scene's ``world_bounds_xy_m``.
2. Mark a grid cell *free* iff every point within
   ``robot_radius + safety_margin`` of the cell center lies outside every
   obstacle. This is a Minkowski inflation of the obstacles by the body
   radius — paths on the free grid are guaranteed to give the body room
   to follow them.
3. 8-connected A* on the free grid returns world-frame waypoints suitable
   for a pure-pursuit follower. Diagonal moves are disallowed when either
   cardinal neighbour is occupied, so the path never shaves a corner.

The grid is intentionally the authoritative oracle: the route teacher
does not consult the room-cell graph or its ``nav_blocked_cells`` for
planning. The room-cell graph is still used by the frontier/recovery
teachers (which operate at room granularity) and by privileged label
derivation.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from go2nav.corpus.manifest import BoxObject, SceneManifest


@dataclass(frozen=True)
class GridPath:
    """Result of an A* query — waypoints in world coordinates."""

    waypoints_xy: tuple[tuple[float, float], ...]
    cost_cells: float


class InflatedOccupancyGrid:
    """Free-space occupancy grid for body-radius motion planning."""

    def __init__(
        self,
        manifest: SceneManifest,
        *,
        cell_size_m: float = 0.05,
        inflation_m: float = 0.20,
        treat_landmarks_as_obstacles: bool = True,
        treat_distractors_as_obstacles: bool = True,
    ) -> None:
        (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
        pad = float(inflation_m) + float(cell_size_m)
        self._origin = (float(x_lo) - pad, float(y_lo) - pad)
        self._cell_size = float(cell_size_m)
        self._inflation = float(inflation_m)
        self._nx = int(math.ceil((float(x_hi) - float(x_lo) + 2.0 * pad) / self._cell_size))
        self._ny = int(math.ceil((float(y_hi) - float(y_lo) + 2.0 * pad) / self._cell_size))

        obstacles = _collect_aabbs(manifest.walls) + _collect_aabbs(manifest.obstacles)
        landmark_boxes = _collect_aabbs(manifest.landmarks)
        if treat_landmarks_as_obstacles:
            obstacles = obstacles + landmark_boxes
        # Distractors are visual-randomization decorations stored under
        # ``manifest.visual_randomization.distractor_objects``, but the
        # scene_builder folds them into ``static_objects`` at load time, so
        # they are *real* collision geometry the robot can crash into.
        # Without this branch the route teacher plans paths straight through
        # them, which manifests as crashes + wedge episodes in
        # ``visual_sensor_stress`` (the only family that spawns distractors
        # inside the maze interior).
        if treat_distractors_as_obstacles:
            distractor_boxes = _distractor_aabbs(manifest)
            if distractor_boxes:
                obstacles = obstacles + distractor_boxes
        self._obstacle_aabbs = (
            np.asarray(obstacles, dtype=np.float32)
            if obstacles
            else np.zeros((0, 5), dtype=np.float32)
        )
        self._landmark_aabbs = (
            np.asarray(landmark_boxes, dtype=np.float32)
            if landmark_boxes
            else np.zeros((0, 5), dtype=np.float32)
        )

        ix = np.arange(self._nx, dtype=np.float32)
        iy = np.arange(self._ny, dtype=np.float32)
        xs = self._origin[0] + (ix + 0.5) * self._cell_size
        ys = self._origin[1] + (iy + 0.5) * self._cell_size
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        clearance = _batch_distance_to_boxes(
            gx.ravel(), gy.ravel(), self._obstacle_aabbs
        ).reshape(self._nx, self._ny)
        self._obstacle_clearance = clearance
        # Point-to-box distance is exactly zero throughout an obstacle's
        # interior.  The usual inflated-space predicate is inclusive at the
        # requested clearance, but zero inflation is the physical occupancy
        # special case: obstacle interiors and boundaries are still occupied.
        self._free = (
            clearance > 0.0
            if self._inflation == 0.0
            else clearance >= self._inflation
        )

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def cell_size_m(self) -> float:
        return self._cell_size

    @property
    def inflation_m(self) -> float:
        return self._inflation

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nx, self._ny)

    @property
    def origin_xy(self) -> tuple[float, float]:
        return self._origin

    @property
    def free_mask(self) -> np.ndarray:
        """Read-only view of the underlying boolean free-space grid."""

        return self._free

    @property
    def obstacle_clearance_grid_m(self) -> np.ndarray:
        """Read-only view of point-to-obstacle clearance at grid-cell centers."""

        return self._obstacle_clearance

    def to_grid(self, xy: tuple[float, float]) -> tuple[int, int]:
        ix = int(math.floor((float(xy[0]) - self._origin[0]) / self._cell_size))
        iy = int(math.floor((float(xy[1]) - self._origin[1]) / self._cell_size))
        return ix, iy

    def to_world(self, ij: tuple[int, int]) -> tuple[float, float]:
        x = self._origin[0] + (float(ij[0]) + 0.5) * self._cell_size
        y = self._origin[1] + (float(ij[1]) + 0.5) * self._cell_size
        return x, y

    def is_free(self, xy: tuple[float, float]) -> bool:
        ix, iy = self.to_grid(xy)
        if 0 <= ix < self._nx and 0 <= iy < self._ny:
            return bool(self._free[ix, iy])
        return False

    def obstacle_clearance_m(self, xy: tuple[float, float]) -> float:
        """Return continuous point clearance to the nearest obstacle surface.

        Unlike :meth:`is_free`, this query is evaluated at the exact world
        coordinate rather than the containing raster cell. Points outside the
        finite planning grid return ``0.0`` so benchmark trajectories cannot
        treat leaving the represented world as open space.
        """

        ix, iy = self.to_grid(xy)
        if not (0 <= ix < self._nx and 0 <= iy < self._ny):
            return 0.0
        point_x = np.asarray([float(xy[0])], dtype=np.float32)
        point_y = np.asarray([float(xy[1])], dtype=np.float32)
        return float(_batch_distance_to_boxes(point_x, point_y, self._obstacle_aabbs)[0])

    def configuration_clearance_m(self, xy: tuple[float, float]) -> float:
        """Return signed clearance after applying the configured body inflation.

        Positive values are collision-free margins, zero lies on the inflated
        configuration-space boundary, and negative values are unsafe. The
        query is continuous with respect to the obstacle geometry even though
        :meth:`is_free` remains a raster lookup for A*.
        """

        return self.obstacle_clearance_m(xy) - self._inflation

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nearest_free(
        self,
        xy: tuple[float, float],
        *,
        max_radius_m: float | None = None,
    ) -> tuple[float, float] | None:
        """Return the closest free-grid cell center to ``xy``.

        Used to snap a noisy world xy (e.g. the robot's pose after a step
        of physics drift carried it 5 mm into the inflated region of a
        wall) onto the planner's grid without giving up the whole plan.
        """

        ix, iy = self.to_grid(xy)
        if 0 <= ix < self._nx and 0 <= iy < self._ny and self._free[ix, iy]:
            return self.to_world((ix, iy))
        max_r_cells = (
            int(math.ceil(float(max_radius_m) / self._cell_size))
            if max_radius_m is not None
            else max(self._nx, self._ny)
        )
        for radius in range(1, max_r_cells + 1):
            for di in range(-radius, radius + 1):
                for dj in (-radius, radius):
                    ii, jj = ix + di, iy + dj
                    if (
                        0 <= ii < self._nx
                        and 0 <= jj < self._ny
                        and self._free[ii, jj]
                    ):
                        return self.to_world((ii, jj))
            for dj in range(-radius + 1, radius):
                for di in (-radius, radius):
                    ii, jj = ix + di, iy + dj
                    if (
                        0 <= ii < self._nx
                        and 0 <= jj < self._ny
                        and self._free[ii, jj]
                    ):
                        return self.to_world((ii, jj))
        return None

    def has_free_line(
        self,
        src_xy: tuple[float, float],
        dst_xy: tuple[float, float],
    ) -> bool:
        """True iff every grid cell along ``src→dst`` is free.

        Sampled at quarter-cell resolution. Used by the pure-pursuit
        follower to extend the lookahead point along the planned path
        without crossing into an inflated obstacle.
        """

        sx, sy = float(src_xy[0]), float(src_xy[1])
        dx, dy = float(dst_xy[0]), float(dst_xy[1])
        length = math.hypot(dx - sx, dy - sy)
        if length <= 1e-6:
            return self.is_free(src_xy)
        steps = max(1, int(math.ceil(length / (0.25 * self._cell_size))))
        for k in range(steps + 1):
            t = k / steps
            x = sx + (dx - sx) * t
            y = sy + (dy - sy) * t
            if not self.is_free((x, y)):
                return False
        return True

    def astar(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        clearance_weight: float = 0.0,
        clearance_target_m: float = 0.0,
        allow_diagonal: bool = True,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> GridPath | None:
        """8-connected A* with octile heuristic. Returns world waypoints or None."""

        start = self.to_grid(start_xy)
        goal = self.to_grid(goal_xy)
        nx, ny = self._nx, self._ny
        free = self._free
        blocked = set(blocked_cells or ())

        if not (0 <= goal[0] < nx and 0 <= goal[1] < ny) or not free[goal]:
            return None
        if not (0 <= start[0] < nx and 0 <= start[1] < ny) or not free[start]:
            snapped = self.nearest_free(start_xy, max_radius_m=4.0 * self._inflation)
            if snapped is None:
                return None
            start = self.to_grid(snapped)
            if not free[start]:
                return None
        if start == goal:
            return GridPath(waypoints_xy=(self.to_world(goal),), cost_cells=0.0)

        diag = math.sqrt(2.0)
        if allow_diagonal:
            neighbors = (
                ((-1, -1), diag), ((-1, 0), 1.0), ((-1, 1), diag),
                ((0, -1), 1.0),                    ((0, 1), 1.0),
                ((1, -1), diag),  ((1, 0), 1.0),  ((1, 1), diag),
            )
        else:
            neighbors = (
                ((-1, 0), 1.0),
                ((0, -1), 1.0),
                ((0, 1), 1.0),
                ((1, 0), 1.0),
            )

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            dx_ = abs(a[0] - b[0])
            dy_ = abs(a[1] - b[1])
            if not allow_diagonal:
                return float(dx_ + dy_)
            return (dx_ + dy_) + (diag - 2.0) * min(dx_, dy_)

        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        gscore: dict[tuple[int, int], float] = {start: 0.0}
        closed: set[tuple[int, int]] = set()

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                path: list[tuple[int, int]] = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                waypoints = tuple(self.to_world(ij) for ij in path[1:])
                return GridPath(waypoints_xy=waypoints, cost_cells=gscore[goal])
            closed.add(cur)
            cur_g = gscore[cur]
            cx, cy = cur
            for (di, dj), step in neighbors:
                ni, nj = cx + di, cy + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                if not free[ni, nj]:
                    continue
                neighbour = (ni, nj)
                if neighbour in blocked and neighbour != goal:
                    continue
                if neighbour in closed:
                    continue
                # Block diagonal corner-cutting: the diagonal step is only
                # legal when both cardinal neighbours are also free,
                # otherwise the inflated path shaves an obstacle corner.
                if di != 0 and dj != 0:
                    if not free[cx + di, cy] or not free[cx, cy + dj]:
                        continue
                clearance_penalty = 0.0
                if float(clearance_weight) > 0.0:
                    clearance_deficit = max(
                        0.0,
                        float(clearance_target_m)
                        - (float(self._obstacle_clearance[ni, nj]) - self._inflation),
                    )
                    clearance_penalty = float(clearance_weight) * clearance_deficit / max(
                        self._cell_size,
                        1e-6,
                    )
                tentative = cur_g + step + clearance_penalty
                if tentative < gscore.get(neighbour, math.inf):
                    came_from[neighbour] = cur
                    gscore[neighbour] = tentative
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative + heuristic(neighbour, goal), counter, neighbour),
                    )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_aabbs(
    boxes: tuple[BoxObject, ...] | list[BoxObject],
) -> list[tuple[float, float, float, float, float]]:
    out: list[tuple[float, float, float, float, float]] = []
    for obj in boxes:
        cx, cy, _ = obj.center_xyz_m
        sx, sy, _ = obj.size_xyz_m
        out.append(
            (
                float(cx),
                float(cy),
                float(sx) * 0.5,
                float(sy) * 0.5,
                float(obj.yaw_rad),
            )
        )
    return out


def _distractor_aabbs(
    manifest: SceneManifest,
) -> list[tuple[float, float, float, float, float]]:
    """Return distractor AABBs from ``manifest.visual_randomization`` if present."""

    visual = getattr(manifest, "visual_randomization", None)
    if visual is None:
        return []
    distractors = getattr(visual, "distractor_objects", ())
    if not distractors:
        return []
    return _collect_aabbs(distractors)


def _batch_distance_to_boxes(
    xs: np.ndarray,
    ys: np.ndarray,
    aabbs: np.ndarray,
) -> np.ndarray:
    """Vectorized point-to-rotated-AABB distance, minimised across boxes."""

    if aabbs.shape[0] == 0:
        return np.full(xs.shape, np.inf, dtype=np.float32)
    cx = aabbs[:, 0]
    cy = aabbs[:, 1]
    hx = aabbs[:, 2]
    hy = aabbs[:, 3]
    yaw = aabbs[:, 4]
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    dx_w = xs[:, None] - cx[None, :]
    dy_w = ys[:, None] - cy[None, :]
    local_x = cos_y[None, :] * dx_w - sin_y[None, :] * dy_w
    local_y = sin_y[None, :] * dx_w + cos_y[None, :] * dy_w
    outside_x = np.maximum(0.0, np.abs(local_x) - hx[None, :])
    outside_y = np.maximum(0.0, np.abs(local_y) - hy[None, :])
    dists = np.hypot(outside_x, outside_y)
    return dists.min(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Standoff sampling for beacon goals
# ---------------------------------------------------------------------------


def best_free_yaw(
    grid: InflatedOccupancyGrid,
    xy: tuple[float, float],
    *,
    n_rays: int = 16,
    max_m: float = 4.0,
) -> tuple[float, float]:
    """Return ``(yaw_rad, free_distance_m)`` of the longest free ray from ``xy``.

    Used by the spawn-safety helper so a robot dropped in a narrow corridor
    starts facing down the corridor's free direction, not perpendicular to
    a wall. The yaw is the angle of the ray (atan2 convention), and the
    distance is the length we could trace before the grid call fails.
    Returns ``(0.0, 0.0)`` if ``xy`` itself is not free.
    """

    if not grid.is_free(xy):
        return 0.0, 0.0
    bx, by = float(xy[0]), float(xy[1])
    step = 0.25 * grid.cell_size_m
    max_steps = max(1, int(math.ceil(float(max_m) / step)))
    best_yaw = 0.0
    best_len = 0.0
    for k in range(int(n_rays)):
        yaw = (2.0 * math.pi * k) / float(n_rays) - math.pi
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        length = 0.0
        for s in range(1, max_steps + 1):
            d = s * step
            if not grid.is_free((bx + cos_y * d, by + sin_y * d)):
                break
            length = d
        if length > best_len:
            best_len = length
            best_yaw = yaw
    return best_yaw, best_len


def corridor_width_m(
    grid: InflatedOccupancyGrid,
    xy: tuple[float, float],
    *,
    max_cells: int = 20,
) -> float:
    """Approximate the narrower-axis corridor width at ``xy``.

    Walks the free mask outward from the cell containing ``xy`` along the
    four cardinal directions. ``min(N+S, E+W)`` cells (minus the shared
    centre cell) times the cell size gives the narrower-axis free width.
    Returns 0.0 for an out-of-bounds or obstacle cell.

    This is a cheap O(max_cells) proxy — it misses diagonal corridors —
    but it is the same measure the route teacher uses to bias its
    standoff selection, so the corpus validator and the planner agree on
    what counts as a navigable arrival corridor.
    """

    try:
        sx, sy = grid.to_grid(xy)
    except Exception:
        return 0.0
    free = grid.free_mask
    nx, ny = grid.shape
    if not (0 <= sx < nx and 0 <= sy < ny and free[sx, sy]):
        return 0.0

    def reach(dx: int, dy: int) -> int:
        x, y, n = sx, sy, 0
        while 0 <= x < nx and 0 <= y < ny and free[x, y]:
            x += dx
            y += dy
            n += 1
            if n >= max_cells:
                break
        return n

    ns = reach(0, 1) + reach(0, -1)
    ew = reach(1, 0) + reach(-1, 0)
    min_cells = max(0, min(ns, ew) - 1)
    return min_cells * grid.cell_size_m


def safe_standoff_xys(
    grid: InflatedOccupancyGrid,
    beacon_xy: tuple[float, float],
    *,
    standoff_m: float,
    n_candidates: int = 12,
) -> tuple[tuple[float, float], ...]:
    """Return free standoff anchors at ``standoff_m`` from ``beacon_xy``.

    Samples ``n_candidates`` candidates uniformly around the beacon and
    keeps the ones in free grid space. The caller picks the cheapest by
    A* path cost; the planner uses the chosen point as its A* goal and
    the beacon center as the orientation target on final approach.
    """

    bx, by = float(beacon_xy[0]), float(beacon_xy[1])
    out: list[tuple[float, float]] = []
    for k in range(int(n_candidates)):
        angle = (2.0 * math.pi * k) / float(n_candidates)
        x = bx + math.cos(angle) * float(standoff_m)
        y = by + math.sin(angle) * float(standoff_m)
        if grid.is_free((x, y)):
            out.append((x, y))
    return tuple(out)
