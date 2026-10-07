"""Collector-policy interface and per-episode scheduler.

A *collector policy* picks the requested command block for one env at one
block tick. Each policy is independent and stateless across scenes (but may
carry per-env state across blocks within the same episode).

The :class:`EpisodeScheduler` draws a collector per env per episode from the
weighted share table in ``docs/fresh_retrain_data_spec.md`` §13, so each
episode has a coherent intent (route teacher all the way to the goal, not
route teacher for one block then OU noise the next).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # avoid pulling lewm_worlds into every import path
    from go2nav.corpus.graph import SceneGraph


@dataclass(frozen=True)
class EnvObservation:
    """Privileged per-env state observed at the start of a command block.

    Used by collector policies to pick the next primitive. Reflects the data
    spec §3.4 boundary: every field here is privileged and may not appear in
    the deployed model's input — only the chosen command block does.
    """

    env_idx: int
    base_xy_world: tuple[float, float]
    base_yaw_world: float
    current_cell_id: int
    nearest_cell_distance_m: float
    clearance_to_walls_m: float
    last_executed_cmd: tuple[float, float, float]
    episode_id: int
    episode_step: int
    block_idx_in_episode: int


@dataclass(frozen=True)
class BlockChoice:
    """A collector's chosen command block for one env at one tick."""

    requested_block: np.ndarray  # (T, 3) float32
    primitive_name: str
    command_source: str
    route_target_id: int = -1
    next_waypoint_id: int = -1


class CollectorPolicy(Protocol):
    """Per-env state-conditioned primitive selector.

    Implementations may keep per-env state across blocks (see
    ``on_episode_reset`` and ``on_block``). They must be deterministic given
    the same ``rng``, observation, and prior on-episode-reset / on-block
    calls — repeatability is part of the data contract.
    """

    name: str

    def on_episode_reset(self, env_idx: int) -> None:
        """Called once per env when a new episode begins (cell, yaw, sim time reset)."""

    def on_block(self, *, observation: EnvObservation, scene: "SceneGraph", rng: np.random.Generator) -> BlockChoice:
        """Return the requested command block for this env at this block tick."""




# ---------------------------------------------------------------------------
# Episode scheduler
# ---------------------------------------------------------------------------


class EpisodeScheduler:
    """Assign one collector per env, redrawn each time the env resets.

    Weights default to the §13 mix but are validated against the registered
    policies — missing policies are dropped silently with their share
    redistributed proportionally so a smoke run with a smaller bench still
    works without hand-editing the table.
    """

    def __init__(
        self,
        *,
        policies: dict[str, CollectorPolicy],
        shares: dict[str, float],
        rng: np.random.Generator,
        n_envs: int,
    ) -> None:
        if not policies:
            raise ValueError("EpisodeScheduler requires at least one policy")
        keep = {name: float(shares.get(name, 0.0)) for name in policies}
        total = sum(keep.values())
        if total <= 0.0:
            # Fall back to uniform if no overlap between policies and shares.
            uniform = 1.0 / len(policies)
            keep = {name: uniform for name in policies}
            total = 1.0
        self._policy_names: list[str] = list(policies.keys())
        self._policies = dict(policies)
        self._cdf: list[float] = []
        running = 0.0
        for name in self._policy_names:
            running += keep[name] / total
            self._cdf.append(running)
        # Make sure the final entry is exactly 1.0 so bisect lookups don't
        # ever miss the last bucket because of float drift.
        self._cdf[-1] = 1.0
        self._rng = rng
        self._assignment: list[str] = [self._draw() for _ in range(int(n_envs))]
        # Tracks how many episodes each policy was assigned to per env so
        # the rollout summary can report the realized mix.
        self._assignment_counts: dict[str, int] = {name: 0 for name in self._policy_names}
        for name in self._assignment:
            self._assignment_counts[name] += 1


    def policy_for(self, env_idx: int) -> CollectorPolicy:
        return self._policies[self._assignment[int(env_idx)]]


    def on_episode_reset(self, env_idx: int) -> str:
        """Redraw the policy for this env and notify it. Returns the new name."""

        name = self._draw()
        self._assignment[int(env_idx)] = name
        self._assignment_counts[name] += 1
        self._policies[name].on_episode_reset(int(env_idx))
        return name

    def realized_mix(self) -> dict[str, int]:
        return dict(self._assignment_counts)

    # ------------------------------------------------------------------

    def _draw(self) -> str:
        sample = float(self._rng.random())
        idx = bisect.bisect_left(self._cdf, sample)
        if idx >= len(self._policy_names):
            idx = len(self._policy_names) - 1
        return self._policy_names[idx]


# ---------------------------------------------------------------------------
# Primitive-selection helpers (shared by route teacher / frontier teacher)
# ---------------------------------------------------------------------------


def wrap_angle_pi(angle: float) -> float:
    return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)


def primitive_toward_bearing(
    *,
    heading_error_rad: float,
    yaw_tolerance_rad: float = math.pi / 6,
    arc_tolerance_rad: float = math.pi / 2,
) -> str:
    """Pick a velocity primitive that steers toward a desired bearing.

    | |error| | primitive       |
    | ------- | --------------- |
    | <= π/6  | forward_medium  |
    | <= π/2  | arc_left/right  |
    | >  π/2  | yaw_left/right  |

    Defaults bias hard toward translation: arc covers any turn up to 90°
    (so the body keeps moving through corridor corners) and yaw-in-place
    only kicks in for near-reversals. With the previous π/8 / π/3 bands a
    typical 90° junction turn produced ~7 blocks of pure yaw before the
    robot moved forward again, which dominated long maze traversals
    (env 2 in the route_teacher profiler spent 42% of its blocks yawing).
    """

    if abs(heading_error_rad) <= yaw_tolerance_rad:
        return "forward_medium"
    if abs(heading_error_rad) <= arc_tolerance_rad:
        return "arc_left" if heading_error_rad > 0.0 else "arc_right"
    return "yaw_left" if heading_error_rad > 0.0 else "yaw_right"
