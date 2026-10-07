"""Deterministic corpus planning with split- and family-disjoint seeds.

The planner converts a desired per-split, per-family scene count into a flat,
reproducible list of ``(split, family, scene_seed)`` assignments. Seeds are
derived by hashing ``(plan_seed, split, family, index, salt)``, so two splits
never share a topology seed even if their family counts overlap. ``salt`` is
0 by default; when scene-reachability validation is enabled, the planner
increments the salt until the scene passes ``audit_scene_reachability``,
keeping plan generation deterministic without inheriting unusable seeds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

from go2nav.corpus.families import build_family_manifest, registered_families
from go2nav.corpus.manifest import stable_scene_id


# Splits supported out of the box. Callers may extend ``totals`` with custom
# splits — the planner is split-name agnostic.
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test_id", "test_hard")


# Shares (data spec §8) used by ``smoke_corpus_plan`` and
# ``standard_corpus_plan`` below. Train shares match the spec table verbatim;
# hard test deliberately reweights toward large/alias families.
TRAIN_SHARES: dict[str, float] = {
    "open_obstacle_field": 0.10,
    "local_composite_motifs": 0.15,
    "small_enclosed_maze": 0.15,
    "medium_enclosed_maze": 0.25,
    "large_enclosed_maze": 0.15,
    "loop_alias_stress": 0.10,
    "rough_local_dynamics": 0.05,
    "visual_sensor_stress": 0.05,
}

HARD_TEST_SHARES: dict[str, float] = {
    "open_obstacle_field": 0.00,
    "local_composite_motifs": 0.00,
    "small_enclosed_maze": 0.00,
    "medium_enclosed_maze": 0.20,
    "large_enclosed_maze": 0.30,
    "loop_alias_stress": 0.30,
    "rough_local_dynamics": 0.10,
    "visual_sensor_stress": 0.10,
}


@dataclass(frozen=True)
class SceneAssignment:
    split: str
    family: str
    scene_index: int
    scene_seed: int
    scene_id: str
    # Salt offset used when scene-reachability validation rerolled a
    # fragmented seed. 0 when the deterministic base seed produced a valid
    # scene (the common case). Recorded so corpus reproducibility is exact
    # and so audits can spot families with high reroll rates.
    scene_seed_salt: int = 0


@dataclass(frozen=True)
class CorpusPlan:
    plan_seed: int
    plan_version: str
    splits: tuple[str, ...]
    families: tuple[str, ...]
    totals: dict[str, dict[str, int]]
    assignments: tuple[SceneAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_seed": self.plan_seed,
            "plan_version": self.plan_version,
            "splits": list(self.splits),
            "families": list(self.families),
            "totals": self.totals,
            "assignments": [asdict(a) for a in self.assignments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusPlan":
        return cls(plan_seed=int(payload["plan_seed"]), plan_version=str(payload["plan_version"]),
                   splits=tuple(payload["splits"]), families=tuple(payload["families"]), totals=payload["totals"],
                   assignments=tuple(SceneAssignment(**a) for a in payload["assignments"]))



PLAN_VERSION = "1"

# Max reroll attempts per (split, family, index) slot when validation is on.
# Surveys show ≤26% fragmentation rate across families, so 32 retries leaves
# plenty of headroom (≈0.74^32 ≈ 1e-4 chance of giving up legitimately).
_MAX_VALIDATION_SALT = 32


def plan_corpus(
    plan_seed: int,
    totals: dict[str, dict[str, int]],
    *,
    validate: bool = False,
    validator: Callable[[Any], bool] | None = None,
) -> CorpusPlan:
    """Return a deterministic :class:`CorpusPlan`.

    ``totals`` maps ``split -> {family -> count}``. Splits and families are
    validated against the registered set so a typo fails loudly instead of
    silently producing an empty plan.

    When ``validate=True``, each slot's deterministic seed is built into a
    manifest and audited for route-teacher reachability (every beacon must
    sit in the same free component, with at least one spawn candidate in
    that component). Failing seeds are rerolled by incrementing the salt
    and recorded via ``SceneAssignment.scene_seed_salt`` so the corpus
    layout stays reproducible. ``validator`` overrides the default
    reachability gate (useful for tests).
    """

    known_families = set(registered_families())
    assignments: list[SceneAssignment] = []
    seen_seeds: set[int] = set()

    splits_present: list[str] = []
    families_present: set[str] = set()

    scene_ok = validator or (_default_scene_validator if validate else None)

    for split, family_counts in totals.items():
        if not isinstance(family_counts, dict):
            raise TypeError(f"totals[{split!r}] must be a dict of family->count")
        splits_present.append(split)
        for family, count in family_counts.items():
            if family not in known_families:
                raise ValueError(
                    f"unknown family '{family}' in totals[{split!r}]; "
                    f"registered families: {sorted(known_families)}"
                )
            if int(count) < 0:
                raise ValueError(f"negative count for totals[{split!r}][{family!r}]")
            families_present.add(family)
            for index in range(int(count)):
                seed, salt = _pick_valid_seed(
                    plan_seed=plan_seed,
                    split=split,
                    family=family,
                    index=index,
                    seen_seeds=seen_seeds,
                    validate=scene_ok,
                )
                seen_seeds.add(seed)
                assignments.append(
                    SceneAssignment(
                        split=split,
                        family=family,
                        scene_index=index,
                        scene_seed=seed,
                        scene_id=stable_scene_id(family, seed),
                        scene_seed_salt=salt,
                    )
                )

    return CorpusPlan(
        plan_seed=int(plan_seed),
        plan_version=PLAN_VERSION,
        splits=tuple(splits_present),
        families=tuple(sorted(families_present)),
        totals={s: dict(f) for s, f in totals.items()},
        assignments=tuple(assignments),
    )


def _pick_valid_seed(
    *,
    plan_seed: int,
    split: str,
    family: str,
    index: int,
    seen_seeds: set[int],
    validate: Callable[[Any], bool] | None,
) -> tuple[int, int]:
    """Return ``(seed, salt)`` for one plan slot, retrying on collisions and
    (if ``validate`` is provided) on reachability failures."""

    for salt in range(_MAX_VALIDATION_SALT + 1):
        seed = _scene_seed(
            plan_seed=plan_seed, split=split, family=family, index=index, salt=salt
        )
        if seed in seen_seeds:
            continue
        if validate is None:
            return seed, salt
        manifest = build_family_manifest(
            scene_seed=seed, family=family, split=split, difficulty_tier=None
        )
        if validate(manifest):
            return seed, salt
    raise RuntimeError(
        f"could not find a valid seed for split={split!r} family={family!r} "
        f"index={index} after {_MAX_VALIDATION_SALT} reroll attempts — the "
        f"family builder may be producing fragmented scenes too often; "
        f"inspect with audit_scene_reachability and fix the topology"
    )


def _default_scene_validator(manifest: Any) -> bool:
    # Imported lazily so plan_corpus stays usable in environments without
    # numpy (the unrestricted path has no scene-validation dependency).
    from go2nav.corpus.validation import audit_scene_reachability

    report = audit_scene_reachability(
        manifest,
        cell_size_m=0.05,
        inflation_m=0.20,
        standoff_m=0.85,
        spawn_clearance_floor_m=0.20,
        # Require each beacon to expose at least one navigable-width LOS
        # standoff in the canonical component. Without this, alcove
        # beacons (notably in local_composite_motifs, ~50% of that
        # family's beacons in the 2026-05-19 scan) pass the grid-
        # reachability gate but cap route-teacher yield because the only
        # sightline is through a corridor the trained PPO can't traverse.
        # Reroll feasibility was confirmed: 0/30 local_composite slots
        # exhaust the salt budget under this criterion (mean salt 3.2).
        min_navigable_standoffs_per_beacon=1,
    )
    return bool(report.is_valid)




def standard_corpus_plan(
    *,
    plan_seed: int = 1,
    train_scenes: int = 200,
    val_scenes: int = 50,
    test_id_scenes: int = 50,
    test_hard_scenes: int = 50,
    validate: bool = True,
) -> CorpusPlan:
    """Spec-aligned plan with configurable per-split totals.

    Counts are distributed across families using :data:`TRAIN_SHARES` for
    train/val/test_id and :data:`HARD_TEST_SHARES` for the hard test split.
    Fractional allocations are rounded with leftovers given to the family
    holding the largest share so that the totals add up exactly.
    """

    totals: dict[str, dict[str, int]] = {
        "train": _allocate(train_scenes, TRAIN_SHARES),
        "val": _allocate(val_scenes, TRAIN_SHARES),
        "test_id": _allocate(test_id_scenes, TRAIN_SHARES),
        "test_hard": _allocate(test_hard_scenes, HARD_TEST_SHARES),
    }
    return plan_corpus(plan_seed=plan_seed, totals=totals, validate=validate)


def plan_sha256(plan: CorpusPlan) -> str:
    payload = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scene_seed(
    *, plan_seed: int, split: str, family: str, index: int, salt: int = 0
) -> int:
    """Deterministic per-slot seed.

    ``salt=0`` reproduces the legacy seed exactly, so existing plans built
    without validation continue to map to the same scenes. Validation-aware
    planning increments the salt for each reroll attempt.
    """

    if salt == 0:
        key = f"{plan_seed}:{split}:{family}:{index}".encode("utf-8")
    else:
        key = f"{plan_seed}:{split}:{family}:{index}:{salt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {family: 0 for family in shares}
    raw = {family: total * weight for family, weight in shares.items()}
    base = {family: int(value) for family, value in raw.items()}
    remainder = total - sum(base.values())
    if remainder:
        leftover = sorted(
            shares.items(),
            key=lambda item: (raw[item[0]] - base[item[0]], item[1]),
            reverse=True,
        )
        for family, _ in leftover[:remainder]:
            base[family] += 1
    return base
