"""OU-noise exploration collector (data spec §13 — 10% share).

An Ornstein–Uhlenbeck process in command space, snapped to the nearest
trainable velocity primitive. This gives the LeWM encoder local-dynamics
diversity beyond the discrete primitive bank — the predictor sees command
sequences that drift smoothly through the action space rather than always
hitting one of a handful of clean primitive corners.

Snapping is necessary because (a) the planner's primitive bank is discrete
and (b) the rest of the pipeline (CommandBlock msg, executed-vs-requested
audit, primitive support histogram) is keyed off ``primitive_name``.
"""

from __future__ import annotations

import numpy as np

from lewm_genesis.collectors.base import BlockChoice, EnvObservation
from lewm_genesis.lewm_contract import PrimitiveRegistry, expand_primitive_to_block


class OUExploration:
    name = "ou_noise"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        theta: float = 0.15,
        sigma: float = 0.20,
        dt: float = 1.0,
        allowed_primitives: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._theta = float(theta)
        self._sigma = float(sigma)
        self._dt = float(dt)
        names = list(allowed_primitives or registry.trainable_velocity_names())
        # Pre-tabulate the (vx, vy, yaw_rate) of each primitive so the snap
        # step is a constant-time nearest-neighbour search.
        self._primitive_names = names
        self._primitive_vectors = np.zeros((len(names), 3), dtype=np.float32)
        for i, name in enumerate(names):
            spec = registry.get(name)
            cmd = spec.get("command", {})
            self._primitive_vectors[i] = (
                float(cmd.get("vx_body_mps", 0.0)),
                float(cmd.get("vy_body_mps", 0.0)),
                float(cmd.get("yaw_rate_radps", 0.0)),
            )
        # Per-env OU state, initialised at zero (the equilibrium mean).
        self._state = np.zeros((int(n_envs), 3), dtype=np.float32)

    def on_episode_reset(self, env_idx: int) -> None:
        self._state[env_idx] = 0.0

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,  # unused
        rng: np.random.Generator,
    ) -> BlockChoice:
        env_idx = observation.env_idx
        # OU update: x_{t+1} = x_t - theta * x_t * dt + sigma * sqrt(dt) * N(0, I)
        noise = rng.standard_normal(size=3).astype(np.float32)
        self._state[env_idx] += (
            -self._theta * self._state[env_idx] * self._dt
            + self._sigma * np.sqrt(self._dt) * noise
        )
        # Snap to nearest primitive by Euclidean distance in (vx, vy, yaw_rate).
        target = self._state[env_idx]
        diffs = self._primitive_vectors - target
        sq = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(sq))
        primitive_name = self._primitive_names[idx]
        block = expand_primitive_to_block(self._registry, primitive_name)
        return BlockChoice(
            requested_block=block,
            primitive_name=primitive_name,
            command_source=self.name,
        )
