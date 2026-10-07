"""Privileged route teacher (data spec §13 — 30% share).

Picks an unclaimed beacon goal per episode, computes a safe path with A*
on an inflated configuration-space grid (see
:class:`go2nav.corpus.grid.InflatedOccupancyGrid`), then follows
the path with a pure-pursuit primitive controller. Replans whenever the
current beacon is claimed, the robot deviates from the planned path, or
a periodic replan timer fires.

The grid-based planner replaces an earlier mix of room-cell BFS plus
several stacking heuristics (stuck FSM, off-center recenter, beacon
force-yaw, smoothed-waypoint LOS skip). The new design's invariant: if
A* returns a path, every cell on it has body-radius clearance, so the
follower never has to compensate for "the planner picked an unsafe
waypoint" — and the heuristics that used to mask graph-construction
bugs are no longer needed.
"""

from __future__ import annotations

import math

import numpy as np

from go2nav.corpus.collectors.base import (
    BlockChoice,
    EnvObservation,
    primitive_toward_bearing,
    wrap_angle_pi,
)
from go2nav.corpus.contract import PrimitiveRegistry, expand_primitive_to_block
from go2nav.corpus.grid import (
    GridPath,
    InflatedOccupancyGrid,
    corridor_width_m,
    safe_standoff_xys,
)


# How far along the planned path the pure-pursuit follower looks for the
# next steering target. 0.5 m sits roughly between two intersections in a
# 1 m corridor, so the body keeps committing to the next segment instead
# of stuttering at every waypoint.
_LOOKAHEAD_M = 0.5

# Number of standoff candidates sampled around each beacon. The planner
# tries each in A* and keeps the cheapest reachable one. 12 evenly spaced
# angles (every 30 deg) is the stable default; loop-alias scenes benefit
# from denser doorway/corner approach poses without showing the same
# regressions seen in visual/maze families.
_STANDOFF_CANDIDATES = 12
_LOOP_ALIAS_STANDOFF_CANDIDATES = 24
_DENSE_STANDOFF_FAMILIES = frozenset({"loop_alias_stress"})

# Small enclosed mazes showed a stable wedge signature where a slightly wider
# near-field arrival envelope recovers claims without changing other families.
# Medium/local-composite looked similar in traces but regressed under the same
# relaxation, so keep the default gate there.
_WEDGE_RELAXED_ARRIVAL_FAMILIES = frozenset({"small_enclosed_maze"})
_WEDGE_RELAXED_CLAIM_RADIUS_M = 0.15
_WEDGE_RELAXED_HALF_FOV_MARGIN_RAD = math.radians(8.0)

# A standoff that is cheap to reach can still be a bad arrival pose if the
# final path segment leaves the robot facing away from the beacon. Penalize
# those endpoints so the pure-pursuit follower tends to arrive already looking
# toward the landmark instead of needing an in-place yaw next to geometry.
_STANDOFF_HEADING_PENALTY_CELLS = 60.0

# Standoffs that are *grid-free* (clearance ≥ body radius after inflation) can
# still be unnavigable for the trained locomotion policy when the surrounding
# corridor is barely wider than the body itself. In small-enclosed-maze traces
# the failing arrivals all landed in 0.25-0.35 m corridors — geometrically free
# under 0.20 m inflation, but ≤8 cm clearance per side once the 0.40 m body
# width is laid in. Prefer wider arrival corridors by adding a soft penalty for
# standoffs whose narrower-axis corridor width is below the safe threshold.
#
# The penalty is scoped to ``_STANDOFF_CORRIDOR_PENALTY_FAMILIES``. Earlier
# sweeps that applied it corpus-wide regressed medium/large/visual yields by
# 6-19 pp because the cardinal-walk width proxy fires on moderately-narrow
# corridors that the planner *can* navigate, pushing it to longer paths that
# exhaust the block budget. Restricting to small-enclosed-maze (where the
# diagnosis was that the trained PPO is wedging on ≤0.35 m corridor pockets)
# keeps the +pp gain there without touching other families.
_STANDOFF_CORRIDOR_PENALTY_FAMILIES = frozenset({"small_enclosed_maze"})
_STANDOFF_CORRIDOR_SAFE_WIDTH_M = 0.50
_STANDOFF_CORRIDOR_PENALTY_CELLS_PER_CM = 30.0
# Max cells to walk from the standoff before declaring "wide enough"; keeps the
# cardinal-axis probe O(1) and bounds it well above the safe width threshold.
_STANDOFF_CORRIDOR_PROBE_MAX_CELLS = 20

# Near a beacon standoff, keep turning primitives translational. In-place yaw
# is fragile next to blocking geometry with the trained locomotion policy; an
# arc both rotates and moves the base enough to avoid wedge/yaw/backout loops.
# (Beacons live in open rooms now, so the arc has lateral room.)
_BEACON_APPROACH_ARC_TOLERANCE_RAD = math.pi

# Below this live wall clearance the corridor is too tight to arc through: a
# turn-while-moving sweeps the body sideways into the wall, jamming the camera
# against blank geometry. Collapsing the arc band to the forward tolerance
# (so turns become in-place yaw) keeps the body centred in the corridor.
# Room cells sit well above this (~0.5 m+), so room/open-space turns still arc.
_NARROW_CORRIDOR_CLEARANCE_M = 0.45
_NARROW_CORRIDOR_ARC_TOLERANCE_RAD = math.pi / 6

# If the teacher makes no meaningful path progress toward the current
# standoff for this many blocks, reject that standoff for the rest of the
# episode and re-shop the beacon. This handles final-approach poses that are
# geometrically valid in the grid but poor for the trained locomotion policy.
_APPROACH_RETRY_BLOCKS = 96
_APPROACH_RETRY_PROGRESS_M = 0.20
_REJECTED_STANDOFF_PENALTY_CELLS = 1.0e6
_FAILED_REPLAN_PATH_REUSE_FACTOR = 1.5

# Replan cadence in blocks. Replanning is cheap (~1 ms) so this is mostly
# a guard against accumulated controller drift driving the body off the
# planned path; the planner also replans immediately on beacon claim or
# detected deviation from the path.
_REPLAN_EVERY_BLOCKS = 24

# Locomotion-stuck recovery. The grid guarantees the planned path has
# body-radius clearance, but the trained PPO locomotion policy can still
# wedge against geometry it has never seen. If we command a motion
# primitive for ``_STUCK_TRIGGER_BLOCKS`` blocks and the body neither
# translates nor rotates, emit ``_STUCK_BACKOUT_BLOCKS`` of backward to
# break the wedge, then invalidate the plan so the next block replans
# from the freed pose. This is the *only* state machine that survived
# the rewrite — it handles locomotion failure, not planning failure.
_STUCK_TRIGGER_BLOCKS = 3
_STUCK_BACKOUT_BLOCKS = 4
_STUCK_MIN_DISPLACEMENT_M = 0.015
_STUCK_MIN_YAW_CHANGE_RAD = math.radians(1.0)
_STUCK_MIN_CMD_VX_MPS = 0.05
_STUCK_MIN_CMD_YAW_RATE_RADPS = 0.10


class RouteTeacher:
    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        name: str = "route_teacher",
        revisit_after_arrival: bool = False,
        require_landmark_visible: bool = True,
        landmark_fov_deg: float = 78.323,
        landmark_los_margin_m: float = 0.02,
        landmark_standoff_m: float = 0.85,
        landmark_claim_radius_m: float = 0.20,
        # Forward offset of the camera origin from the body center in the
        # body frame. Used by the arrival gate to check beacon visibility
        # from the camera (not the body), since the camera sits 0.326 m
        # ahead of the body on the Go2 and that offset materially widens
        # the apparent bearing to a beacon at the claim envelope.
        # Set to 0.0 to recover the legacy body-frame check.
        landmark_camera_forward_offset_m: float = 0.326,
        grid_cell_size_m: float = 0.05,
        grid_inflation_m: float = 0.20,
        deviation_replan_m: float = 0.60,
        path_waypoint_spacing_m: float = 0.30,
        approach_retry_blocks: int | None = None,
        approach_retry_progress_m: float | None = None,
        # Accepted but ignored — kept so existing callers / tests that pass
        # the old keyword arguments do not break. The new design has no
        # equivalent dial (the inflated grid + A* is parameter-free
        # apart from cell size and inflation).
        arrival_cell_radius: int | None = None,
        replan_every_blocks: int | None = None,
        min_goal_distance_hops: int | None = None,
        arrival_distance_m: float | None = None,
    ) -> None:
        self.name = name
        self._registry = registry
        self._revisit = bool(revisit_after_arrival)
        # Public read-only flag: revisit collectors keep re-targeting claimed
        # beacons after clearing them all, so the rollout must NOT treat their
        # all-claimed state as episode completion (that's their data signal).
        self.revisit_after_arrival = bool(revisit_after_arrival)
        self._require_landmark_visible = bool(require_landmark_visible)
        self._landmark_half_fov_rad = max(0.0, float(landmark_fov_deg)) * 0.5 * math.pi / 180.0
        self._landmark_los_margin_m = float(landmark_los_margin_m)
        self._landmark_standoff_m = max(0.0, float(landmark_standoff_m))
        self._landmark_claim_radius_m = max(0.0, float(landmark_claim_radius_m))
        self._landmark_camera_forward_offset_m = float(landmark_camera_forward_offset_m)
        self._grid_cell_size_m = float(grid_cell_size_m)
        self._grid_inflation_m = float(grid_inflation_m)
        self._deviation_replan_m = float(deviation_replan_m)
        self._path_waypoint_spacing_m = float(path_waypoint_spacing_m)
        self._replan_every_blocks = int(
            replan_every_blocks if replan_every_blocks is not None else _REPLAN_EVERY_BLOCKS
        )
        self._approach_retry_blocks = int(
            approach_retry_blocks
            if approach_retry_blocks is not None
            else _APPROACH_RETRY_BLOCKS
        )
        self._approach_retry_progress_m = float(
            approach_retry_progress_m
            if approach_retry_progress_m is not None
            else _APPROACH_RETRY_PROGRESS_M
        )

        # Per-env state. Sized to ``n_envs`` and reset on episode reset.
        n = int(n_envs)
        self._claimed: list[set[int]] = [set() for _ in range(n)]
        self._goal_cell: list[int] = [-1] * n
        self._goal_standoff_xy: list[tuple[float, float] | None] = [None] * n
        self._path: list[tuple[tuple[float, float], ...]] = [()] * n
        self._path_idx: list[int] = [0] * n
        self._blocks_since_plan: list[int] = [0] * n
        self._goal_history: list[list[int]] = [[] for _ in range(n)]
        self._unreachable_beacons: list[set[int]] = [set() for _ in range(n)]
        self._rejected_standoffs: list[dict[int, set[tuple[int, int]]]] = [
            {} for _ in range(n)
        ]
        self._approach_best_remaining_m: list[float | None] = [None] * n
        self._approach_blocks_since_progress: list[int] = [0] * n
        # Relaxed-claim telemetry: a claim is *relaxed* when it fires under
        # the wedge-family widened envelope but would not have fired under
        # the strict default radius+FOV. By construction this can only be
        # non-zero for families in _WEDGE_RELAXED_ARRIVAL_FAMILIES, so the
        # mass-datagen audit can use the per-family aggregate as a sanity
        # check on the relaxation scope. Events stay cheap (~bytes each)
        # so we keep them across episode resets — the rollout summary
        # consumes them once at run end.
        self._relaxed_claim_count: list[int] = [0] * n
        self._relaxed_claim_events: list[list[dict]] = [[] for _ in range(n)]
        # Locomotion-stuck FSM state (see _STUCK_* constants).
        self._last_xy: list[tuple[float, float] | None] = [None] * n
        self._last_yaw: list[float | None] = [None] * n
        self._stuck_blocks: list[int] = [0] * n
        self._backout_remaining: list[int] = [0] * n

        # Grids are cached by scene identity. Constructed once per scene
        # at the first ``on_block`` call so episode-reset latency stays low.
        self._grids: dict[int, InflatedOccupancyGrid] = {}
        self._spawn_restrictions: dict[int, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spawn_restriction_cells(self, scene) -> tuple[int, ...] | None:
        """Return the set of cell IDs that are in the canonical free component.

        If 'route_teacher' is active, we should only spawn in regions that
        can reach all beacons.
        """
        scene_id = id(scene)
        if scene_id in self._spawn_restrictions:
            return self._spawn_restrictions[scene_id]

        scene_graph = getattr(scene, "scene_graph", None)
        if scene_graph is None:
            return None

        cells = scene_graph.canonical_spawn_cells(
            cell_size_m=self._grid_cell_size_m,
            inflation_m=self._grid_inflation_m,
            standoff_m=self._landmark_standoff_m,
        )
        self._spawn_restrictions[scene_id] = cells
        return cells

    def spawn_planning_grid(self, scene) -> InflatedOccupancyGrid | None:
        """Expose the cached planning grid for spawn-yaw alignment.

        The rollout uses this to avoid rebuilding a second grid when the
        route teacher already owns one for the active scene.
        """

        return self._grid_for(scene)

    def on_episode_reset(self, env_idx: int) -> None:
        self._claimed[env_idx].clear()
        self._goal_cell[env_idx] = -1
        self._goal_standoff_xy[env_idx] = None
        self._path[env_idx] = ()
        self._path_idx[env_idx] = 0
        self._blocks_since_plan[env_idx] = 0
        self._unreachable_beacons[env_idx].clear()
        self._rejected_standoffs[env_idx].clear()
        self._reset_approach_progress(env_idx)
        self._last_xy[env_idx] = None
        self._last_yaw[env_idx] = None
        self._stuck_blocks[env_idx] = 0
        self._backout_remaining[env_idx] = 0
        # ``_goal_history`` is intentionally not cleared so revisit mode
        # still has prior goals to pick from after a reset.

    def visited_landmark_cells(self, env_idx: int) -> frozenset[int]:
        """Return landmark cells claimed by this env in the current episode."""

        return frozenset(self._claimed[int(env_idx)])

    def relaxed_claim_count(self, env_idx: int) -> int:
        """Return total relaxed claims recorded for this env across all episodes."""

        return int(self._relaxed_claim_count[int(env_idx)])

    def relaxed_claim_events(self, env_idx: int) -> tuple[dict, ...]:
        """Return a tuple of relaxed-claim events recorded for this env.

        Each event carries ``beacon_cell``, ``scene_id``, ``scene_family``,
        and ``env_idx`` so downstream audits can correlate the relaxation
        with rendered landmark visibility per frame.
        """

        return tuple(self._relaxed_claim_events[int(env_idx)])

    # ------------------------------------------------------------------
    # Per-block decision
    # ------------------------------------------------------------------

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,
        rng: np.random.Generator,
    ) -> BlockChoice:
        env_idx = observation.env_idx
        grid = self._grid_for(scene)

        # If the body is already crashed into a wall, reverse out before
        # doing anything else.
        if observation.clearance_to_walls_m <= 0.05:
            self._invalidate_plan(env_idx)
            self._stuck_blocks[env_idx] = 0
            self._last_xy[env_idx] = observation.base_xy_world
            self._last_yaw[env_idx] = observation.base_yaw_world
            return self._backward_block()

        # Locomotion-stuck recovery: a wedged body that the locomotion
        # policy can't free should bail out of the current plan rather
        # than burn the rest of the episode commanding the same
        # primitive.
        stuck_choice = self._step_stuck_fsm(env_idx, observation)
        if stuck_choice is not None:
            return stuck_choice

        # 1. Claim the current beacon if we've arrived at one.
        self._mark_arrival_if_present(env_idx, observation, scene)
        self._maybe_reject_stale_approach(env_idx, observation)

        # 2. Plan if we don't have a valid path to a pending beacon.
        if self._needs_replan(env_idx, observation):
            self._plan(env_idx, observation, scene, grid, rng)

        goal_cell = self._goal_cell[env_idx]
        if goal_cell < 0:
            return self._hold_block(route_target_id=-1)

        # 3. Pure-pursuit follower. Pick the farthest visible point on
        # the remaining path within the lookahead window; that becomes
        # the steering target.
        target_xy = self._lookahead_target(env_idx, observation, grid)
        if target_xy is None:
            # Path exhausted but arrival never fired (rare — happens when
            # physics overshoots the standoff). Steer to the standoff
            # anchor directly.
            target_xy = self._goal_standoff_xy[env_idx] or self._beacon_xy(goal_cell, scene)

        # If close to the standoff anchor, switch the steering target to
        # the beacon itself so the body finishes facing the beacon for
        # the FOV / LOS arrival check.
        standoff_xy = self._goal_standoff_xy[env_idx]
        near_beacon_standoff = False
        if standoff_xy is not None:
            dist_to_standoff = _distance(observation.base_xy_world, standoff_xy)
            if dist_to_standoff <= self._claim_radius_m(scene) + 0.10:
                target_xy = self._beacon_xy(goal_cell, scene)
                near_beacon_standoff = True

        bearing = math.atan2(
            target_xy[1] - observation.base_xy_world[1],
            target_xy[0] - observation.base_xy_world[0],
        )
        heading_err = wrap_angle_pi(bearing - observation.base_yaw_world)
        primitive_kwargs = {"heading_error_rad": heading_err}
        beacon_neighborhood_open = (
            observation.clearance_to_walls_m >= _NARROW_CORRIDOR_CLEARANCE_M
        )
        if near_beacon_standoff:
            if beacon_neighborhood_open:
                # Open beacon (room / open visual spot): arcing here circles
                # the beacon at standoff radius and never locks the FOV/LOS
                # claim pose — the cause of the orbiting failures. Yaw in
                # place to face the beacon and commit to the arrival check.
                primitive_kwargs["arc_tolerance_rad"] = _NARROW_CORRIDOR_ARC_TOLERANCE_RAD
            else:
                # Tight beacon pocket: in-place yaw risks wedging against
                # close geometry, so arc (turn-while-moving) to finish.
                primitive_kwargs["arc_tolerance_rad"] = _BEACON_APPROACH_ARC_TOLERANCE_RAD
        elif not beacon_neighborhood_open:
            # In a narrow corridor an arc (forward + yaw) sweeps the body
            # sideways into the wall — the camera ends up jammed against a
            # blank wall and the locomotion policy hugs it. Collapse the arc
            # band so any turn beyond the forward tolerance becomes an
            # in-place yaw, which rotates without lateral translation. The
            # stuck FSM still backs the body out if a yaw genuinely wedges.
            primitive_kwargs["arc_tolerance_rad"] = _NARROW_CORRIDOR_ARC_TOLERANCE_RAD
        primitive_name = primitive_toward_bearing(**primitive_kwargs)
        self._blocks_since_plan[env_idx] += 1
        return BlockChoice(
            requested_block=expand_primitive_to_block(self._registry, primitive_name),
            primitive_name=primitive_name,
            command_source=self.name,
            route_target_id=int(goal_cell),
            next_waypoint_id=-1,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _grid_for(self, scene) -> InflatedOccupancyGrid:
        key = id(scene)
        grid = self._grids.get(key)
        if grid is None:
            grid = InflatedOccupancyGrid(
                scene.manifest,
                cell_size_m=self._grid_cell_size_m,
                inflation_m=self._grid_inflation_m,
            )
            self._grids[key] = grid
        return grid

    def _needs_replan(self, env_idx: int, observation: EnvObservation) -> bool:
        if self._goal_cell[env_idx] < 0 or not self._path[env_idx]:
            return True
        if self._blocks_since_plan[env_idx] >= self._replan_every_blocks:
            return True
        # Replan if the robot has drifted far from every remaining path
        # waypoint — the path the controller is chasing is no longer the
        # right one.
        remaining = self._path[env_idx][self._path_idx[env_idx] :]
        if not remaining:
            return True
        min_dist = min(_distance(observation.base_xy_world, wp) for wp in remaining)
        if min_dist > self._deviation_replan_m:
            return True
        return False

    def _invalidate_plan(self, env_idx: int, *, reset_progress: bool = True) -> None:
        self._path[env_idx] = ()
        self._path_idx[env_idx] = 0
        if reset_progress:
            self._reset_approach_progress(env_idx)

    def _plan(
        self,
        env_idx: int,
        observation: EnvObservation,
        scene,
        grid: InflatedOccupancyGrid,
        rng: np.random.Generator,
    ) -> None:
        self._blocks_since_plan[env_idx] = 0
        start_xy = observation.base_xy_world

        # If the body has wandered far enough inside an inflated obstacle
        # that the grid can't snap a free start cell, don't try to plan
        # *and don't poison the unreachable cache* — the next block may
        # have a viable start once locomotion drift settles.
        snapped_start = grid.nearest_free(
            start_xy, max_radius_m=4.0 * grid.inflation_m
        )
        if snapped_start is None:
            self._clear_goal(env_idx)
            return

        # Sticky goal: if we already have a viable goal, just refresh
        # the path to it instead of re-shopping for the cheapest beacon
        # every replan. Mid-pursuit beacon-flipping wastes blocks turning
        # the body around and confuses the follower's path index.
        current_goal = self._goal_cell[env_idx]
        if current_goal >= 0 and current_goal not in self._claimed[env_idx]:
            beacon_xy = self._beacon_xy(current_goal, scene)
            if beacon_xy is not None:
                standoffs = self._available_standoffs(
                    env_idx,
                    current_goal,
                    grid,
                    beacon_xy,
                    scene,
                )
                sticky_best: tuple[float, tuple[float, float], GridPath] | None = None
                for standoff in standoffs:
                    path = grid.astar(snapped_start, standoff)
                    if path is None:
                        continue
                    cost = self._standoff_path_cost(path, beacon_xy)
                    cost += self._rejected_standoff_penalty(
                        env_idx,
                        current_goal,
                        standoff,
                    )
                    cost += self._standoff_corridor_penalty(grid, standoff, scene)
                    if sticky_best is None or cost < sticky_best[0]:
                        sticky_best = (cost, standoff, path)
                if sticky_best is not None:
                    _, standoff_xy, path = sticky_best
                    self._set_plan(
                        env_idx,
                        current_goal,
                        standoff_xy,
                        path,
                        observation,
                        append_history=False,
                    )
                    return
                if self._can_reuse_existing_path_after_failed_replan(
                    env_idx, observation
                ):
                    # A transient drift into inflated obstacle space can snap
                    # the replan start into the wrong component. Keep chasing
                    # the last viable path instead of poisoning the beacon.
                    self._blocks_since_plan[env_idx] = 0
                    self._reset_approach_progress(env_idx, observation)
                    return
                # Current goal no longer reachable — fall through and
                # consider it unreachable while picking a new beacon.
                self._unreachable_beacons[env_idx].add(int(current_goal))

        candidates = self._candidate_goals(env_idx, scene, rng)
        best: tuple[float, int, tuple[float, float], GridPath] | None = None
        for beacon_cell, beacon_xy in candidates:
            standoffs = self._available_standoffs(
                env_idx,
                beacon_cell,
                grid,
                beacon_xy,
                scene,
            )
            if not standoffs:
                # No safe standoff with LOS to the beacon — robot could
                # reach a standoff but never satisfy the arrival check,
                # so the beacon is effectively unreachable.
                self._unreachable_beacons[env_idx].add(beacon_cell)
                continue
            beacon_reachable = False
            for standoff in standoffs:
                path = grid.astar(snapped_start, standoff)
                if path is None:
                    continue
                beacon_reachable = True
                cost = self._standoff_path_cost(path, beacon_xy)
                cost += self._rejected_standoff_penalty(
                    env_idx,
                    beacon_cell,
                    standoff,
                )
                cost += self._standoff_corridor_penalty(grid, standoff, scene)
                if best is None or cost < best[0]:
                    best = (cost, beacon_cell, standoff, path)
            if not beacon_reachable:
                # All free standoffs of this beacon are in a different
                # connected component than our start cell. In a static
                # scene that fact doesn't change across the episode, so
                # cache it to skip the per-beacon A* work next replan.
                self._unreachable_beacons[env_idx].add(beacon_cell)

        if best is None:
            self._clear_goal(env_idx)
            return

        _, beacon_cell, standoff_xy, path = best
        self._set_plan(
            env_idx,
            beacon_cell,
            standoff_xy,
            path,
            observation,
            append_history=True,
        )

    def _standoff_corridor_penalty(
        self,
        grid: InflatedOccupancyGrid,
        standoff_xy: tuple[float, float],
        scene=None,
    ) -> float:
        """Penalize standoffs whose corridor is narrower than the safe width.

        The inflated grid encodes ``body_radius`` clearance only. A free
        cell can still sit in a corridor only marginally wider than the
        body (e.g. ~0.30 m grid-free corridor → ~6 cm trained-PPO
        clearance per side after the 0.40 m body width is laid in). The
        planner can pick such a standoff because it satisfies A*'s
        reachability check, but the locomotion policy wedges on
        approach. Score these poses with a width penalty so the planner
        prefers wider arrival corridors when alternatives exist; ties
        and wide-corridor scenes are unaffected.

        Width is approximated by cardinal-axis free-walk from the
        standoff cell. ``min(N+S, E+W)`` gives the narrower-axis
        corridor width; both axes meeting the safe threshold drops the
        penalty to zero.

        Scoped to families in ``_STANDOFF_CORRIDOR_PENALTY_FAMILIES``;
        any other scene returns zero so the planner's existing scoring
        is unchanged on families where the corpus sweep showed the
        cardinal-walk proxy regressing yield.
        """

        if scene is not None and self._scene_family(scene) not in _STANDOFF_CORRIDOR_PENALTY_FAMILIES:
            return 0.0
        min_width_m = corridor_width_m(
            grid, standoff_xy, max_cells=_STANDOFF_CORRIDOR_PROBE_MAX_CELLS
        )
        deficit_m = _STANDOFF_CORRIDOR_SAFE_WIDTH_M - min_width_m
        if deficit_m <= 0.0:
            return 0.0
        return _STANDOFF_CORRIDOR_PENALTY_CELLS_PER_CM * (deficit_m * 100.0)

    def _standoff_path_cost(
        self,
        path: GridPath,
        beacon_xy: tuple[float, float],
    ) -> float:
        """A* cost plus a final-heading penalty for beacon-facing arrivals."""

        waypoints = path.waypoints_xy
        if len(waypoints) < 2:
            return float(path.cost_cells)
        end = waypoints[-1]
        prev = None
        for candidate in reversed(waypoints[:-1]):
            if _distance(candidate, end) > 1e-6:
                prev = candidate
                break
        if prev is None:
            return float(path.cost_cells)
        approach_heading = math.atan2(end[1] - prev[1], end[0] - prev[0])
        beacon_heading = math.atan2(beacon_xy[1] - end[1], beacon_xy[0] - end[0])
        heading_err = abs(wrap_angle_pi(beacon_heading - approach_heading))
        return float(path.cost_cells) + (
            _STANDOFF_HEADING_PENALTY_CELLS * heading_err / math.pi
        )

    def _candidate_goals(
        self,
        env_idx: int,
        scene,
        rng: np.random.Generator,
    ) -> list[tuple[int, tuple[float, float]]]:
        """Return (beacon_cell, beacon_xy) pairs to consider for the next goal."""

        claimed = self._claimed[env_idx]
        unreachable = self._unreachable_beacons[env_idx]
        landmark_cells = getattr(scene, "landmark_cells", ())
        if self._revisit:
            pool: list[int] = []
            for history in self._goal_history:
                for cell in history:
                    if cell not in claimed and cell not in unreachable:
                        pool.append(int(cell))
            if pool:
                rng.shuffle(pool)
                out: list[tuple[int, tuple[float, float]]] = []
                seen: set[int] = set()
                for cell in pool:
                    if cell in seen:
                        continue
                    seen.add(cell)
                    xy = scene.landmark_xy_for_cell(int(cell))
                    if xy is not None:
                        out.append((int(cell), (float(xy[0]), float(xy[1]))))
                if out:
                    return out
        out = []
        for _name, cell in landmark_cells:
            if cell in claimed or cell in unreachable:
                continue
            xy = scene.landmark_xy_for_cell(int(cell))
            if xy is None:
                continue
            out.append((int(cell), (float(xy[0]), float(xy[1]))))
        return out

    # ------------------------------------------------------------------
    # Arrival
    # ------------------------------------------------------------------

    def _mark_arrival_if_present(
        self,
        env_idx: int,
        observation: EnvObservation,
        scene,
    ) -> None:
        goal_cell = self._goal_cell[env_idx]
        if goal_cell < 0:
            # Also pick up the case where the robot starts the episode
            # sitting in a beacon cell — claim it so we don't waste an
            # A* trying to plan back to where we already are.
            cur = int(observation.current_cell_id)
            if self._is_landmark_cell(cur, scene):
                strict_pass, relaxed_pass = self._evaluate_arrival_gates(
                    observation, cur, scene
                )
                if relaxed_pass:
                    if not strict_pass:
                        self._record_relaxed_claim(env_idx, cur, scene)
                    self._claimed[env_idx].add(cur)
            return
        strict_pass, relaxed_pass = self._evaluate_arrival_gates(
            observation, goal_cell, scene
        )
        if relaxed_pass:
            if not strict_pass:
                self._record_relaxed_claim(env_idx, goal_cell, scene)
            self._claimed[env_idx].add(goal_cell)
            self._clear_goal(env_idx)


    def _evaluate_arrival_gates(
        self,
        observation: EnvObservation,
        beacon_cell: int,
        scene,
    ) -> tuple[bool, bool]:
        """Return ``(strict_pass, relaxed_pass)`` for this beacon arrival.

        Strict = default radius + default FOV (family-agnostic). Relaxed =
        the family-aware envelope returned by ``_claim_radius_m`` /
        ``_landmark_half_fov_rad_for_scene``. For non-wedge families the
        two are identical, so any claim where ``relaxed_pass and not
        strict_pass`` is by construction a wedge-family relaxation.
        """

        beacon_xy = self._beacon_xy(beacon_cell, scene)
        if beacon_xy is None:
            return False, False
        dist = _distance(observation.base_xy_world, beacon_xy)
        strict_envelope = self._landmark_standoff_m + self._landmark_claim_radius_m
        relaxed_envelope = self._landmark_standoff_m + self._claim_radius_m(scene)
        if dist > relaxed_envelope:
            return False, False
        if not self._landmark_visible(observation, beacon_cell, scene):
            return False, False
        if dist <= strict_envelope and self._landmark_visible(
            observation, beacon_cell, scene, strict=True
        ):
            return True, True
        return False, True

    def _record_relaxed_claim(self, env_idx: int, beacon_cell: int, scene) -> None:
        self._relaxed_claim_count[env_idx] += 1
        self._relaxed_claim_events[env_idx].append(
            {
                "env_idx": int(env_idx),
                "beacon_cell": int(beacon_cell),
                "scene_family": self._scene_family(scene),
                "scene_id": str(getattr(getattr(scene, "manifest", None), "scene_id", "")),
            }
        )

    def _landmark_visible(
        self,
        observation: EnvObservation,
        beacon_cell: int,
        scene,
        *,
        strict: bool = False,
    ) -> bool:
        """Return True iff the beacon is visible *from the camera*, not the body.

        The camera is mounted ``_landmark_camera_forward_offset_m`` ahead of
        the body in the body frame. With a 0.326 m offset and a beacon at
        ~0.85 m, ignoring the offset can put a beacon ~15° wider in the
        camera frame than in the body frame — at the body 45° gate that
        was the historical default, ~53% of "achieved" events had the
        beacon outside the actual camera FOV, so the rendered training
        frame contained no beacon at all.

        When ``strict=True`` the family-aware FOV widening is bypassed so
        the relaxed-claim detector can decide whether a successful claim
        also passed the default gate.
        """

        if not self._require_landmark_visible:
            return True
        beacon_xy = self._beacon_xy(beacon_cell, scene)
        if beacon_xy is None:
            return True
        bx = float(observation.base_xy_world[0])
        by = float(observation.base_xy_world[1])
        yaw = float(observation.base_yaw_world)
        # Project the beacon into the camera frame: camera sits at
        # (base_xy + R(yaw) · (forward_offset, 0)). The camera's optical
        # axis is aligned with the body yaw (no extrinsic yaw jitter is
        # modelled at this gate; ±3° of jitter is well inside the safety
        # margin between the route-teacher FOV and the platform-manifest
        # camera FOV).
        cam_x = bx + self._landmark_camera_forward_offset_m * math.cos(yaw)
        cam_y = by + self._landmark_camera_forward_offset_m * math.sin(yaw)
        lx, ly = float(beacon_xy[0]), float(beacon_xy[1])
        dx, dy = lx - cam_x, ly - cam_y
        if math.hypot(dx, dy) <= 1e-3:
            return True
        bearing = math.atan2(dy, dx)
        heading_err = wrap_angle_pi(bearing - yaw)
        half_fov = (
            self._landmark_half_fov_rad
            if strict
            else self._landmark_half_fov_rad_for_scene(scene)
        )
        if abs(heading_err) > half_fov:
            return False
        los = getattr(scene, "has_line_of_sight", None)
        if los is None:
            return True
        return bool(
            los(
                (cam_x, cam_y),
                (lx, ly),
                margin_m=self._landmark_los_margin_m,
                exclude_landmark_xy=(lx, ly),
            )
        )

    def _claim_radius_m(self, scene) -> float:
        radius = self._landmark_claim_radius_m
        if self._scene_family(scene) in _WEDGE_RELAXED_ARRIVAL_FAMILIES:
            radius += _WEDGE_RELAXED_CLAIM_RADIUS_M
        return radius

    def _landmark_half_fov_rad_for_scene(self, scene) -> float:
        half_fov = self._landmark_half_fov_rad
        if self._scene_family(scene) in _WEDGE_RELAXED_ARRIVAL_FAMILIES:
            half_fov += _WEDGE_RELAXED_HALF_FOV_MARGIN_RAD
        return half_fov

    def _scene_family(self, scene) -> str:
        manifest = getattr(scene, "manifest", None)
        return str(getattr(manifest, "family", ""))

    # ------------------------------------------------------------------
    # Pure-pursuit follower
    # ------------------------------------------------------------------

    def _lookahead_target(
        self,
        env_idx: int,
        observation: EnvObservation,
        grid: InflatedOccupancyGrid,
    ) -> tuple[float, float] | None:
        path = self._path[env_idx]
        if not path:
            return None
        # Advance the path index to the waypoint *closest to the robot*.
        # Crude metric-distance pinning ("only advance when within 0.2 m
        # of the current waypoint") fails when controller drift pushes
        # the body laterally off the path — the index pins at 0 and the
        # lookahead starts re-pursuing the path origin. The closest-point
        # rule handles arbitrary drift as long as the body is anywhere
        # near the path.
        rx, ry = observation.base_xy_world
        idx = self._path_idx[env_idx]
        best_idx = idx
        best_dist = _distance((rx, ry), path[idx])
        for j in range(idx + 1, len(path)):
            d = _distance((rx, ry), path[j])
            if d < best_dist:
                best_dist = d
                best_idx = j
        self._path_idx[env_idx] = best_idx

        # Scan forward along the path within the lookahead window and
        # keep the farthest waypoint whose straight line from the robot
        # is grid-free. This is the standard pure-pursuit choice; it
        # smooths the path through long corridors and only forces a turn
        # when the next segment goes around a corner.
        chosen = path[best_idx]
        for wp in path[best_idx:]:
            if _distance((rx, ry), wp) > _LOOKAHEAD_M:
                break
            if not grid.has_free_line((rx, ry), wp):
                break
            chosen = wp
        return chosen

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_goal(self, env_idx: int) -> None:
        self._goal_cell[env_idx] = -1
        self._goal_standoff_xy[env_idx] = None
        self._invalidate_plan(env_idx)

    def _set_plan(
        self,
        env_idx: int,
        beacon_cell: int,
        standoff_xy: tuple[float, float],
        path: GridPath,
        observation: EnvObservation,
        *,
        append_history: bool,
    ) -> None:
        self._goal_cell[env_idx] = int(beacon_cell)
        self._goal_standoff_xy[env_idx] = standoff_xy
        self._path[env_idx] = self._sparsify_path(path.waypoints_xy)
        self._path_idx[env_idx] = 0
        self._reset_approach_progress(env_idx, observation)
        if append_history:
            self._goal_history[env_idx].append(int(beacon_cell))

    def _reset_approach_progress(
        self,
        env_idx: int,
        observation: EnvObservation | None = None,
    ) -> None:
        self._approach_blocks_since_progress[env_idx] = 0
        self._approach_best_remaining_m[env_idx] = (
            self._remaining_path_distance(env_idx, observation)
            if observation is not None
            else None
        )

    def _maybe_reject_stale_approach(
        self,
        env_idx: int,
        observation: EnvObservation,
    ) -> None:
        if self._approach_retry_blocks <= 0:
            return
        goal_cell = self._goal_cell[env_idx]
        standoff_xy = self._goal_standoff_xy[env_idx]
        if goal_cell < 0 or standoff_xy is None or not self._path[env_idx]:
            self._reset_approach_progress(env_idx)
            return

        remaining = self._remaining_path_distance(env_idx, observation)
        if remaining is None:
            self._reset_approach_progress(env_idx)
            return
        best = self._approach_best_remaining_m[env_idx]
        if best is None or remaining <= best - self._approach_retry_progress_m:
            self._approach_best_remaining_m[env_idx] = remaining
            self._approach_blocks_since_progress[env_idx] = 0
            return

        self._approach_blocks_since_progress[env_idx] += 1
        if self._approach_blocks_since_progress[env_idx] < self._approach_retry_blocks:
            return

        self._penalize_current_standoff(env_idx)
        self._clear_goal(env_idx)

    def _penalize_current_standoff(self, env_idx: int) -> None:
        goal_cell = self._goal_cell[env_idx]
        standoff_xy = self._goal_standoff_xy[env_idx]
        if goal_cell < 0 or standoff_xy is None:
            return
        self._rejected_standoffs[env_idx].setdefault(int(goal_cell), set()).add(
            self._standoff_key(standoff_xy)
        )

    def _can_reuse_existing_path_after_failed_replan(
        self,
        env_idx: int,
        observation: EnvObservation,
    ) -> bool:
        path = self._path[env_idx]
        if not path:
            return False
        max_reuse_dist = max(
            self._deviation_replan_m * _FAILED_REPLAN_PATH_REUSE_FACTOR,
            self._landmark_standoff_m,
        )
        remaining = path[self._path_idx[env_idx] :] or path
        min_dist = min(_distance(observation.base_xy_world, wp) for wp in remaining)
        return min_dist <= max_reuse_dist

    def _remaining_path_distance(
        self,
        env_idx: int,
        observation: EnvObservation,
    ) -> float | None:
        path = self._path[env_idx]
        if not path:
            return None
        rx, ry = observation.base_xy_world
        idx = max(0, min(self._path_idx[env_idx], len(path) - 1))
        best_idx = idx
        best_dist = _distance((rx, ry), path[idx])
        for j in range(idx + 1, len(path)):
            d = _distance((rx, ry), path[j])
            if d < best_dist:
                best_dist = d
                best_idx = j
        remaining = best_dist
        for j in range(best_idx, len(path) - 1):
            remaining += _distance(path[j], path[j + 1])
        return remaining

    def _available_standoffs(
        self,
        env_idx: int,
        beacon_cell: int,
        grid: InflatedOccupancyGrid,
        beacon_xy: tuple[float, float],
        scene,
    ) -> tuple[tuple[float, float], ...]:
        rejected = self._rejected_standoffs[env_idx].get(int(beacon_cell), set())
        standoffs = self._standoffs_with_los(grid, beacon_xy, scene)
        if not rejected:
            return standoffs
        filtered = tuple(
            standoff
            for standoff in standoffs
            if self._standoff_key(standoff) not in rejected
        )
        # Fall back to *all* standoffs when every candidate has been
        # rejected so the planner can still attempt the beacon. The
        # ``_rejected_standoff_penalty`` (1e6) keeps a rejected
        # standoff at the bottom of the cost ranking — so the planner
        # only re-uses one if no other beacon offers a cheaper route,
        # which lets a lucky controller-drift retry recover a wedge-
        # prone beacon without burning the episode hunting alternatives.
        return filtered or standoffs

    def _rejected_standoff_penalty(
        self,
        env_idx: int,
        beacon_cell: int,
        standoff_xy: tuple[float, float],
    ) -> float:
        rejected = self._rejected_standoffs[env_idx].get(int(beacon_cell), set())
        if self._standoff_key(standoff_xy) not in rejected:
            return 0.0
        return _REJECTED_STANDOFF_PENALTY_CELLS

    def _standoff_key(self, standoff_xy: tuple[float, float]) -> tuple[int, int]:
        scale = max(self._grid_cell_size_m, 1e-6)
        return (
            int(round(float(standoff_xy[0]) / scale)),
            int(round(float(standoff_xy[1]) / scale)),
        )

    def _is_landmark_cell(self, cell: int, scene) -> bool:
        return int(cell) in {int(c) for _name, c in getattr(scene, "landmark_cells", ())}

    def _beacon_xy(self, beacon_cell: int, scene) -> tuple[float, float] | None:
        getter = getattr(scene, "landmark_xy_for_cell", None)
        if getter is None:
            return None
        xy = getter(int(beacon_cell))
        if xy is None:
            return None
        return float(xy[0]), float(xy[1])

    def _hold_block(self, *, route_target_id: int) -> BlockChoice:
        return BlockChoice(
            requested_block=expand_primitive_to_block(self._registry, "hold"),
            primitive_name="hold",
            command_source=self.name,
            route_target_id=int(route_target_id),
            next_waypoint_id=-1,
        )

    def _standoffs_with_los(
        self,
        grid: InflatedOccupancyGrid,
        beacon_xy: tuple[float, float],
        scene,
    ) -> tuple[tuple[float, float], ...]:
        """Free standoffs that also have scene-level LOS to the beacon.

        A standoff that's free on the grid but separated from the beacon
        by a wall is useless: the robot can drive there but never satisfy
        the FOV+LOS arrival check, so the beacon would never be claimed.
        Filter those out at planning time.
        """

        candidates = safe_standoff_xys(
            grid,
            beacon_xy,
            standoff_m=self._landmark_standoff_m,
            n_candidates=self._standoff_candidate_count(scene),
        )
        los = getattr(scene, "has_line_of_sight", None)
        if los is None:
            return candidates
        bx, by = float(beacon_xy[0]), float(beacon_xy[1])
        out: list[tuple[float, float]] = []
        for sx, sy in candidates:
            if los(
                (float(sx), float(sy)),
                (bx, by),
                margin_m=self._landmark_los_margin_m,
                exclude_landmark_xy=(bx, by),
            ):
                out.append((float(sx), float(sy)))
        return tuple(out)

    def _standoff_candidate_count(self, scene) -> int:
        if self._scene_family(scene) in _DENSE_STANDOFF_FAMILIES:
            return _LOOP_ALIAS_STANDOFF_CANDIDATES
        return _STANDOFF_CANDIDATES

    def _sparsify_path(
        self,
        waypoints: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        """Down-sample a 5-cm A* path to ~``path_waypoint_spacing_m`` spacing.

        The raw A* path has one waypoint per grid cell (5 cm). Following
        every one of those makes the pure-pursuit lookahead overly
        sensitive to single-cell deviations and amplifies small
        controller drift into oscillation. Keeping waypoints roughly
        every 30 cm gives the follower longer commit segments that match
        the body's actual cornering radius. The final waypoint (= goal
        standoff) is always retained.
        """

        if len(waypoints) <= 2:
            return waypoints
        spacing = float(self._path_waypoint_spacing_m)
        if spacing <= 0.0:
            return waypoints
        kept: list[tuple[float, float]] = [waypoints[0]]
        last = waypoints[0]
        for wp in waypoints[1:-1]:
            if _distance(last, wp) >= spacing:
                kept.append(wp)
                last = wp
        kept.append(waypoints[-1])
        return tuple(kept)

    def _backward_block(self) -> BlockChoice:
        return BlockChoice(
            requested_block=expand_primitive_to_block(self._registry, "backward"),
            primitive_name="backward",
            command_source=self.name,
        )

    def _step_stuck_fsm(
        self,
        env_idx: int,
        observation: EnvObservation,
    ) -> BlockChoice | None:
        """Return a backout block if the body is wedged; else update state.

        Tracks how many consecutive blocks commanded motion without
        producing displacement or rotation. ``_STUCK_TRIGGER_BLOCKS``
        stuck blocks in a row → ``_STUCK_BACKOUT_BLOCKS`` of backward,
        then invalidate the plan so the next block replans from the
        freed pose.
        """

        cur_xy = observation.base_xy_world
        cur_yaw = observation.base_yaw_world
        prev_xy = self._last_xy[env_idx]
        prev_yaw = self._last_yaw[env_idx]
        self._last_xy[env_idx] = cur_xy
        self._last_yaw[env_idx] = cur_yaw

        if self._backout_remaining[env_idx] > 0:
            self._backout_remaining[env_idx] -= 1
            if self._backout_remaining[env_idx] == 0:
                self._invalidate_plan(env_idx)
                self._stuck_blocks[env_idx] = 0
            return self._backward_block()

        if prev_xy is None or prev_yaw is None:
            return None
        last_vx = abs(float(observation.last_executed_cmd[0]))
        last_yaw_rate = abs(float(observation.last_executed_cmd[2]))
        commanded_motion = (
            last_vx > _STUCK_MIN_CMD_VX_MPS
            or last_yaw_rate > _STUCK_MIN_CMD_YAW_RATE_RADPS
        )
        displacement = math.hypot(cur_xy[0] - prev_xy[0], cur_xy[1] - prev_xy[1])
        yaw_change = abs(wrap_angle_pi(cur_yaw - prev_yaw))
        moved = (
            displacement >= _STUCK_MIN_DISPLACEMENT_M
            or yaw_change >= _STUCK_MIN_YAW_CHANGE_RAD
        )
        if commanded_motion and not moved:
            self._stuck_blocks[env_idx] += 1
        else:
            self._stuck_blocks[env_idx] = 0

        if self._stuck_blocks[env_idx] >= _STUCK_TRIGGER_BLOCKS:
            self._penalize_current_standoff(env_idx)
            self._backout_remaining[env_idx] = _STUCK_BACKOUT_BLOCKS - 1
            self._stuck_blocks[env_idx] = 0
            return self._backward_block()
        return None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
