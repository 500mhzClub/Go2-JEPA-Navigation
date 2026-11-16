"""Privileged frontier teacher (data spec §13 — 20% share).

Drives the agent toward the *least-visited* reachable cell, biased to cells
that are farther away on the graph. Visit counts are kept per env so the
frontier signal stays coherent within an episode and bleeds gently across
episodes for the same env (useful for the loop-closure curriculum).

Reuses the route-teacher's bearing-to-primitive mapping; the only behavioural
difference is goal selection.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from lewm_genesis.collectors.base import (
    BlockChoice,
    EnvObservation,
    primitive_toward_bearing,
    wrap_angle_pi,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry, expand_primitive_to_block


class FrontierTeacher:
    name = "frontier"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        retarget_after_arrival: bool = True,
        bias_distance_weight: float = 0.2,
    ) -> None:
        self._registry = registry
        self._goal: list[int] = [-1] * int(n_envs)
        self._visit_counts: list[defaultdict[int, int]] = [
            defaultdict(int) for _ in range(int(n_envs))
        ]
        self._retarget = bool(retarget_after_arrival)
        self._bias = float(bias_distance_weight)

    def on_episode_reset(self, env_idx: int) -> None:
        # Reset goal but keep visit counts: revisiting the same cell across
        # episodes is exactly what makes a loop closure useful.
        self._goal[env_idx] = -1

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,
        rng: np.random.Generator,
    ) -> BlockChoice:
        env_idx = observation.env_idx
        self._visit_counts[env_idx][int(observation.current_cell_id)] += 1
        goal = self._goal[env_idx]
        blocked = getattr(scene, "nav_blocked_cells", frozenset())
        if (
            goal < 0
            or goal >= scene.n_nodes
            or scene.bfs_distance(observation.current_cell_id, goal, transit_blocked=blocked) is None
            or (self._retarget and observation.current_cell_id == goal)
        ):
            goal = self._pick_goal(env_idx, observation.current_cell_id, scene, rng)
            self._goal[env_idx] = goal

        if goal < 0:
            return BlockChoice(
                requested_block=expand_primitive_to_block(self._registry, "hold"),
                primitive_name="hold",
                command_source=self.name,
            )

        waypoint = scene.next_waypoint(observation.current_cell_id, goal, transit_blocked=blocked)
        if waypoint is None:
            self._goal[env_idx] = -1
            return BlockChoice(
                requested_block=expand_primitive_to_block(self._registry, "hold"),
                primitive_name="hold",
                command_source=self.name,
                route_target_id=int(goal),
            )

        wp_xy = scene.cell_center(waypoint)
        bearing = math.atan2(
            wp_xy[1] - observation.base_xy_world[1],
            wp_xy[0] - observation.base_xy_world[0],
        )
        heading_err = wrap_angle_pi(bearing - observation.base_yaw_world)
        primitive_name = primitive_toward_bearing(heading_error_rad=heading_err)
        return BlockChoice(
            requested_block=expand_primitive_to_block(self._registry, primitive_name),
            primitive_name=primitive_name,
            command_source=self.name,
            route_target_id=int(goal),
            next_waypoint_id=int(waypoint),
        )

    # ------------------------------------------------------------------

    def _pick_goal(
        self,
        env_idx: int,
        current_cell: int,
        scene,
        rng: np.random.Generator,
    ) -> int:
        blocked = getattr(scene, "nav_blocked_cells", frozenset())
        reachable = [c for c in scene.reachable_cells(current_cell, transit_blocked=blocked) if c != current_cell]
        if not reachable:
            return -1
        visits = self._visit_counts[env_idx]
        # Score = -visits + bias * graph_distance. Higher = better.
        scored = []
        for cell in reachable:
            dist = scene.bfs_distance(current_cell, cell, transit_blocked=blocked) or 0
            score = -float(visits.get(cell, 0)) + self._bias * float(dist)
            scored.append((score, int(cell)))
        scored.sort(reverse=True)
        # Pick uniformly among the top-3 ties so the frontier search doesn't
        # become deterministic when many cells are unvisited at start.
        top = scored[: max(1, min(3, len(scored)))]
        return int(rng.choice([c for _s, c in top]))
