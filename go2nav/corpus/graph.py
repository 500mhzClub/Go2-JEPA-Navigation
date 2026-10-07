"""Read-only graph accessor over a :class:`SceneManifest`.

This module is the privileged-side scene oracle. It is used at data
generation time by collector policies (route teacher, frontier teacher,
recovery curriculum) and at label-derivation time by Phase A1 metadata
recovery. **It is never an input to the deployed model** (data spec §2).

All graph indices are scene-scoped: the caller is responsible for pairing
``cell_id`` with the corresponding ``scene_id``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from go2nav.corpus.manifest import SceneManifest


@dataclass(frozen=True)
class CellHit:
    """Result of locating a world-frame xy in the scene's cell graph."""

    cell_id: int
    distance_m: float


class SceneGraph:
    """Pre-computed adjacency, BFS, and geometry queries for a scene.

    The class is intended to be constructed once per scene (during scene
    build) and reused for every per-block decision. All methods are pure
    and depend only on the manifest, so a collector can call them safely
    across parallel envs sharing the same scene.
    """

    def __init__(self, manifest: SceneManifest) -> None:
        self.manifest = manifest
        self._node_xy = np.asarray(
            [node.center_xy_m for node in manifest.graph_nodes], dtype=np.float32
        )
        self._node_tags = tuple(node.tags for node in manifest.graph_nodes)
        self._adjacency: dict[int, list[int]] = {
            node.node_id: [] for node in manifest.graph_nodes
        }
        for edge in manifest.graph_edges:
            if not edge.traversable:
                continue
            self._adjacency[edge.source].append(edge.target)
            self._adjacency[edge.target].append(edge.source)

        # Pre-compute a flat array of wall + obstacle AABBs for fast clearance
        # queries. Each entry is (cx, cy, half_x, half_y, yaw). Landmarks are
        # kept in a separate array so the planner can ask LOS-style questions
        # that *include* beacons as occluders (a beacon mesh is solid enough
        # to block both the lens and a line-of-sight ray) while clearance
        # queries — used by the recovery interlock — stay scoped to walls.
        boxes: list[tuple[float, float, float, float, float]] = []
        for obj in (*manifest.walls, *manifest.obstacles):
            cx, cy, _cz = obj.center_xyz_m
            sx, sy, _sz = obj.size_xyz_m
            boxes.append((float(cx), float(cy), float(sx) * 0.5, float(sy) * 0.5, float(obj.yaw_rad)))
        self._box_aabbs = np.asarray(
            boxes if boxes else np.zeros((0, 5)),
            dtype=np.float32,
        ).reshape(-1, 5)

        landmark_boxes: list[tuple[float, float, float, float, float]] = []
        for obj in manifest.landmarks:
            cx, cy, _cz = obj.center_xyz_m
            sx, sy, _sz = obj.size_xyz_m
            landmark_boxes.append(
                (float(cx), float(cy), float(sx) * 0.5, float(sy) * 0.5, float(obj.yaw_rad))
            )
        self._landmark_aabbs = np.asarray(
            landmark_boxes if landmark_boxes else np.zeros((0, 5)),
            dtype=np.float32,
        ).reshape(-1, 5)

        # Landmark cells: nearest graph node per landmark, used by the
        # route teacher for goal selection.
        self._landmark_cells: tuple[tuple[str, int], ...] = tuple(
            (lm.object_id, self._nearest_node_index((lm.center_xyz_m[0], lm.center_xyz_m[1])))
            for lm in manifest.landmarks
        )
        # Set of cells containing a beacon mesh — used as the "transit
        # blocked" set in nav-flavored BFS calls. A 0.3 m beacon mesh
        # sitting at a cell center leaves only ~0.27 m of corridor either
        # side in typical maze geometry, narrower than the 0.30 m Go2
        # body. The planner needs to know it can ARRIVE at a beacon cell
        # but cannot pass THROUGH one, otherwise the BFS happily routes
        # the body into a wedge between beacon and wall.
        self._beacon_cells: frozenset[int] = frozenset(
            cell for _name, cell in self._landmark_cells
        )

        # Pre-compute cells with insufficient wall clearance. Cells whose
        # centers are within 0.30 m of a wall/obstacle are marked as
        # transit-blocked so the room-cell BFS does not route through narrow
        # gaps. Landmarks are *not* included here — every cell adjacent to a
        # beacon used to inherit the beacon's footprint and get marked
        # blocked, fragmenting the graph and stranding many beacons with no
        # safe approach. Beacon cells themselves stay in
        # ``nav_blocked_cells`` because the mesh blocks corridor transit,
        # but the surrounding room cells remain reachable.
        clearance_threshold_m = 0.30
        self._low_clearance_cells: frozenset[int] = frozenset(
            i for i in range(self.n_nodes)
            if self.clearance_to_walls(self.cell_center(i), include_landmarks=False) < clearance_threshold_m
        )
        self._nav_blocked_cells: frozenset[int] = self._beacon_cells | self._low_clearance_cells

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(self._node_xy.shape[0])


    @property
    def landmark_cells(self) -> tuple[tuple[str, int], ...]:
        """Tuple of ``(object_id, cell_id)`` for every landmark in the scene."""

        return self._landmark_cells

    @property
    def nav_blocked_cells(self) -> frozenset[int]:
        """Set of cell ids that should not be transited through (beacons or low clearance)."""

        return self._nav_blocked_cells



    def bfs_distance(
        self,
        start: int,
        goal: int,
        *,
        transit_blocked: frozenset[int] | set[int] | None = None,
    ) -> int | None:
        """Return graph distance (in hops) on the traversable subgraph, or None.

        ``transit_blocked`` is an optional set of cells that may be reached
        as a path endpoint but cannot be transited through (e.g. beacon
        cells where the mesh blocks the corridor). The start node is
        always exempt so a robot spawning on a blocked cell can still leave.
        """

        if start == goal:
            return 0
        blocked = transit_blocked or frozenset()
        visited = {int(start)}
        frontier: deque[tuple[int, int]] = deque([(int(start), 0)])
        while frontier:
            node, depth = frontier.popleft()
            if node != int(start) and node in blocked:
                continue
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in visited:
                    continue
                if neighbour == goal:
                    return depth + 1
                visited.add(neighbour)
                frontier.append((neighbour, depth + 1))
        return None


    def next_waypoint(
        self,
        start: int,
        goal: int,
        *,
        transit_blocked: frozenset[int] | set[int] | None = None,
    ) -> int | None:
        """Return the next neighbour of ``start`` that lies on a shortest path."""

        if start == goal:
            return None
        blocked = transit_blocked or frozenset()
        parent: dict[int, int] = {int(start): -1}
        frontier: deque[int] = deque([int(start)])
        found = False
        while frontier and not found:
            node = frontier.popleft()
            if node != int(start) and node in blocked:
                continue
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in parent:
                    continue
                parent[neighbour] = node
                if neighbour == goal:
                    found = True
                    break
                frontier.append(neighbour)
        if int(goal) not in parent:
            return None
        cursor = int(goal)
        while parent[cursor] != int(start):
            cursor = parent[cursor]
            if cursor == -1:
                return None
        return cursor

    def reachable_cells(
        self,
        start: int,
        *,
        transit_blocked: frozenset[int] | set[int] | None = None,
    ) -> set[int]:
        """Return the set of node ids reachable from ``start`` on traversable edges."""

        blocked = transit_blocked or frozenset()
        visited: set[int] = {int(start)}
        frontier: deque[int] = deque([int(start)])
        while frontier:
            node = frontier.popleft()
            if node != int(start) and node in blocked:
                continue
            for neighbour in self._adjacency.get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                frontier.append(neighbour)
        return visited



    # ------------------------------------------------------------------
    # World-frame queries
    # ------------------------------------------------------------------

    def locate(self, xy_world: tuple[float, float]) -> CellHit:
        """Return the nearest cell to ``xy_world`` and the Euclidean distance."""

        idx = self._nearest_node_index(xy_world)
        cx, cy = float(self._node_xy[idx, 0]), float(self._node_xy[idx, 1])
        dist = math.hypot(xy_world[0] - cx, xy_world[1] - cy)
        return CellHit(cell_id=idx, distance_m=dist)

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        idx = int(cell_id)
        return float(self._node_xy[idx, 0]), float(self._node_xy[idx, 1])

    def landmark_xy_for_cell(self, cell_id: int) -> tuple[float, float] | None:
        """Return the world-frame xy of any landmark whose cell is ``cell_id``."""

        target = int(cell_id)
        for landmark in self.manifest.landmarks:
            idx = self._nearest_node_index(
                (landmark.center_xyz_m[0], landmark.center_xyz_m[1])
            )
            if idx == target:
                return float(landmark.center_xyz_m[0]), float(landmark.center_xyz_m[1])
        return None

    def has_line_of_sight(
        self,
        src_xy: tuple[float, float],
        dst_xy: tuple[float, float],
        *,
        margin_m: float = 0.0,
        exclude_landmark_xy: tuple[float, float] | None = None,
    ) -> bool:
        """Return True iff the 2D segment ``src→dst`` is unblocked.

        Walls, obstacles, and landmark beacons are all treated as occluders —
        a beacon mesh is solid enough that the camera should not see through
        it. ``margin_m`` inflates each box's half-extents so we don't claim
        visibility through a paper-thin gap. ``exclude_landmark_xy`` lets the
        route teacher verify LOS *to* its target beacon without that beacon
        self-occluding; centers within 1 mm of the supplied xy are skipped.
        """

        sx, sy = float(src_xy[0]), float(src_xy[1])
        dx, dy = float(dst_xy[0]), float(dst_xy[1])
        rx, ry = dx - sx, dy - sy
        length = math.hypot(rx, ry)
        if length <= 1e-6:
            return True
        ux, uy = rx / length, ry / length
        m = float(margin_m)
        exclude_xy = (
            (float(exclude_landmark_xy[0]), float(exclude_landmark_xy[1]))
            if exclude_landmark_xy is not None
            else None
        )

        def _segment_blocked(aabbs: np.ndarray, *, allow_exclude: bool) -> bool:
            for cx, cy, hx, hy, yaw in aabbs:
                if (
                    allow_exclude
                    and exclude_xy is not None
                    and abs(float(cx) - exclude_xy[0]) <= 1e-3
                    and abs(float(cy) - exclude_xy[1]) <= 1e-3
                ):
                    continue
                cos_y = math.cos(-float(yaw))
                sin_y = math.sin(-float(yaw))
                ox = cos_y * (sx - float(cx)) - sin_y * (sy - float(cy))
                oy = sin_y * (sx - float(cx)) + cos_y * (sy - float(cy))
                dx_l = cos_y * ux - sin_y * uy
                dy_l = sin_y * ux + cos_y * uy
                half_x = float(hx) + m
                half_y = float(hy) + m
                t_hit = _segment_aabb_hit(ox, oy, dx_l, dy_l, half_x, half_y)
                if t_hit is not None and 0.0 < t_hit < length - 1e-3:
                    return True
            return False

        if self._box_aabbs.shape[0] and _segment_blocked(self._box_aabbs, allow_exclude=False):
            return False
        if self._landmark_aabbs.shape[0] and _segment_blocked(
            self._landmark_aabbs, allow_exclude=True
        ):
            return False
        return True

    def clearance_to_walls(
        self, xy_world: tuple[float, float], *, include_landmarks: bool = False
    ) -> float:
        """Return distance from ``xy_world`` to the nearest wall/obstacle AABB.

        AABBs are evaluated in world frame after rotating the query point by
        ``-yaw`` per box (so we can test against an axis-aligned half-extent).
        Returns ``+inf`` if the scene has no walls or obstacles.
        """

        aabbs = self._box_aabbs
        if include_landmarks and self._landmark_aabbs.shape[0] > 0:
            aabbs = np.concatenate([aabbs, self._landmark_aabbs], axis=0)

        if aabbs.shape[0] == 0:
            return float("inf")

        x, y = float(xy_world[0]), float(xy_world[1])
        cx = aabbs[:, 0]
        cy = aabbs[:, 1]
        hx = aabbs[:, 2]
        hy = aabbs[:, 3]
        yaw = aabbs[:, 4]
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        dx_world = x - cx
        dy_world = y - cy
        local_x = cos_y * dx_world - sin_y * dy_world
        local_y = sin_y * dx_world + cos_y * dy_world
        outside_x = np.maximum(0.0, np.abs(local_x) - hx)
        outside_y = np.maximum(0.0, np.abs(local_y) - hy)
        dists = np.hypot(outside_x, outside_y)
        return float(np.min(dists))

    # ------------------------------------------------------------------
    # Spawn-pose sampling
    # ------------------------------------------------------------------

    def sample_spawn_pose(
        self,
        rng: random.Random,
        *,
        clearance_floor_m: float = 0.20,
        max_attempts: int = 50,
        spawn_z_m: float = 0.375,
        restrict_to_cells: tuple[int, ...] | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], int]:
        """Sample a random spawn pose at a random cell with random yaw.

        Returns ``(xyz_m, quat_wxyz, cell_id)``. Cells whose center clearance
        falls below ``clearance_floor_m`` are rejected; if no cell can be
        sampled within ``max_attempts`` tries, falls back to the manifest's
        original spawn pose.

        ``restrict_to_cells`` (optional) limits sampling to the listed
        cell ids — used by the route-teacher data pipeline to keep
        spawns inside the canonical free-space component so every
        random episode can reach every beacon. Cells outside the list
        are still rejected even if they pass the clearance check.
        """

        if restrict_to_cells is not None:
            candidates = [int(c) for c in restrict_to_cells if 0 <= int(c) < self.n_nodes]
        else:
            candidates = list(range(self.n_nodes))
        if not candidates:
            xyz = self.manifest.spawn.xyz_m
            quat = self.manifest.spawn.quat_wxyz
            return xyz, quat, -1
        rng.shuffle(candidates)
        for cell_id in candidates[:max_attempts]:
            xy = self.cell_center(cell_id)
            if self.clearance_to_walls(xy) < clearance_floor_m:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            return (
                (float(xy[0]), float(xy[1]), float(spawn_z_m)),
                _yaw_to_wxyz(yaw),
                int(cell_id),
            )
        # Fall back to the manifest spawn so the scene always boots.
        fallback_xy = self.manifest.spawn.xyz_m
        return (
            (float(fallback_xy[0]), float(fallback_xy[1]), float(spawn_z_m)),
            self.manifest.spawn.quat_wxyz,
            int(self.locate((fallback_xy[0], fallback_xy[1])).cell_id),
        )

    def canonical_spawn_cells(
        self,
        *,
        cell_size_m: float = 0.05,
        inflation_m: float = 0.20,
        standoff_m: float = 0.85,
        clearance_floor_m: float = 0.20,
    ) -> tuple[int, ...]:
        """Return room cells in the free component containing every beacon.

        Spawns drawn from this set are guaranteed to be in the same
        grid-free connected component as a LOS-valid standoff for every
        beacon — i.e. every beacon is reachable by the route teacher.
        Returns an empty tuple if no such component exists (scene is
        fragmented across beacon rooms).
        """

        from go2nav.corpus.validation import audit_scene_reachability
        from go2nav.corpus.grid import InflatedOccupancyGrid
        from go2nav.corpus.validation import _component_index

        report = audit_scene_reachability(
            self.manifest,
            cell_size_m=cell_size_m,
            inflation_m=inflation_m,
            standoff_m=standoff_m,
            spawn_clearance_floor_m=clearance_floor_m,
        )
        if not report.is_valid or report.canonical_component_id < 0:
            return ()
        grid = InflatedOccupancyGrid(
            self.manifest, cell_size_m=cell_size_m, inflation_m=inflation_m
        )
        comps = _component_index(grid.free_mask)
        target = int(report.canonical_component_id)
        nx, ny = grid.shape
        out: list[int] = []
        for cell_id in range(self.n_nodes):
            xy = self.cell_center(cell_id)
            if self.clearance_to_walls(xy) < clearance_floor_m:
                continue
            # The spawn pose is the cell center itself (not a snapped
            # neighbour), so the center must be free of every inflated
            # obstacle — including landmarks. Cells with a beacon on top
            # used to slip through here via the nearest_free fallback.
            if not grid.is_free(xy):
                continue
            ix, iy = grid.to_grid(xy)
            if 0 <= ix < nx and 0 <= iy < ny and int(comps[ix, iy]) == target:
                out.append(int(cell_id))
        return tuple(out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest_node_index(self, xy_world: tuple[float, float]) -> int:
        if self._node_xy.shape[0] == 0:
            raise ValueError("scene has no graph nodes")
        diffs = self._node_xy - np.asarray(
            [float(xy_world[0]), float(xy_world[1])], dtype=np.float32
        )
        sq = np.einsum("ij,ij->i", diffs, diffs)
        return int(np.argmin(sq))


def _segment_aabb_hit(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    hx: float,
    hy: float,
) -> float | None:
    """2D slab test. Returns ray parameter ``t >= 0`` of first hit or None."""

    t_min = -float("inf")
    t_max = float("inf")
    for o, d, h in ((ox, dx, hx), (oy, dy, hy)):
        if abs(d) < 1e-9:
            if o < -h or o > h:
                return None
            continue
        t1 = (-h - o) / d
        t2 = (h - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return max(t_min, 0.0)


def _yaw_to_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def wrap_angle_pi(angle: float) -> float:
    """Wrap an angle in radians into ``[-pi, pi]``."""

    return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)


