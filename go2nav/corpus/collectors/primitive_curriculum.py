"""Random primitive sampler — the existing collector wrapped in the new API.

Each env independently samples one trainable velocity primitive per block
from the registry. Provides the 20% "primitive/scripted command curriculum"
slice in data spec §13.
"""

from __future__ import annotations

import numpy as np

from go2nav.corpus.collectors.base import BlockChoice, EnvObservation
from go2nav.corpus.contract import PrimitiveRegistry, expand_primitive_to_block


class PrimitiveCurriculum:
    name = "primitive_curriculum"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        n_envs: int,
        allowed_primitives: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._allowed = list(
            allowed_primitives or registry.trainable_velocity_names()
        )
        if not self._allowed:
            raise ValueError("no trainable velocity primitives available")


    def on_episode_reset(self, env_idx: int) -> None:
        """No per-env state to reset."""

    def on_block(
        self,
        *,
        observation: EnvObservation,
        scene,  # SceneGraph; unused
        rng: np.random.Generator,
    ) -> BlockChoice:
        primitive_name = str(rng.choice(self._allowed))
        block = expand_primitive_to_block(self._registry, primitive_name)
        return BlockChoice(
            requested_block=block,
            primitive_name=primitive_name,
            command_source=self.name,
        )
