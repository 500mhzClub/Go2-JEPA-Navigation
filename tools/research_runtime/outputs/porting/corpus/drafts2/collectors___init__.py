"""Per-env collector policies that drive the §13 command-source mix."""

from lewm_genesis.collectors.base import (
    BlockChoice,
    CollectorPolicy,
    EnvObservation,
    EpisodeScheduler,
)
from lewm_genesis.collectors.frontier import FrontierTeacher
from lewm_genesis.collectors.ou_noise import OUExploration
from lewm_genesis.collectors.primitive_curriculum import PrimitiveCurriculum
from lewm_genesis.collectors.recovery import RecoveryCurriculum
from lewm_genesis.collectors.route_teacher import RouteTeacher

# Default share table from data-spec §13. Sum is 1.0.
DEFAULT_COLLECTION_MIX: dict[str, float] = {
    "route_teacher": 0.30,
    "frontier": 0.20,
    "primitive_curriculum": 0.20,
    "ou_noise": 0.10,
    "recovery": 0.10,
    "loop_revisit": 0.10,
}


def build_default_policies(registry, *, n_envs: int) -> dict[str, CollectorPolicy]:
    """Construct one instance of every collector available out of the box.

    ``loop_revisit`` is currently fulfilled by ``RouteTeacher`` configured to
    re-target previously visited goal cells; see ``RouteTeacher``'s
    ``revisit_after_arrival`` flag.
    """

    return {
        "route_teacher": RouteTeacher(registry, n_envs=n_envs),
        "loop_revisit": RouteTeacher(
            registry,
            n_envs=n_envs,
            revisit_after_arrival=True,
            name="loop_revisit",
        ),
        "frontier": FrontierTeacher(registry, n_envs=n_envs),
        "primitive_curriculum": PrimitiveCurriculum(registry, n_envs=n_envs),
        "ou_noise": OUExploration(registry, n_envs=n_envs),
        "recovery": RecoveryCurriculum(registry, n_envs=n_envs),
    }


__all__ = [
    "BlockChoice",
    "CollectorPolicy",
    "DEFAULT_COLLECTION_MIX",
    "EnvObservation",
    "EpisodeScheduler",
    "FrontierTeacher",
    "OUExploration",
    "PrimitiveCurriculum",
    "RecoveryCurriculum",
    "RouteTeacher",
    "build_default_policies",
]
