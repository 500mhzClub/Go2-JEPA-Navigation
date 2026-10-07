"""Per-scene Genesis bulk rollout loop.

Drives one ``SceneBuild`` with the locomotion policy for ``n_blocks`` command
blocks in parallel environments, each steered by a data collector
(`go2nav.corpus.collectors`), and writes every per-tick observation as
messages (`go2nav.corpus.messages`).

Nesting (taken from the platform manifest):

- physics step ``physics_dt_s`` (0.002 s)
- policy step ``policy_dt_s`` (0.02 s) → 10 physics steps per policy step
- command tick ``command_dt_s`` (0.10 s) → 5 policy steps per command tick
- command block ``action_block_size`` ticks (5) → one block per 0.5 s

Data is emitted at command-tick cadence (10 Hz); camera frames are rendered
afterwards from the recorded poses (`go2nav.corpus.render`). The command
request is emitted at block start; the executed-block record at block end with
the full requested/executed arrays.

Genesis is required at run time (imported lazily by `go2nav.corpus.world`).
"""

from __future__ import annotations

import hashlib
import math
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from go2nav.corpus import messages
from go2nav.corpus.contract import (
    CommandBlockRecord,
    EpisodeState,
    PrimitiveRegistry,
    SafetyLimits,
    apply_safety_limits_batch,
    base_state_from_genesis,
    foot_contacts_record,
    make_episode_states,
    reconstruct_executed_block,
    rotate_body_to_world,
)
from go2nav.corpus.packs import ScenePack
from go2nav.corpus.world import SceneBuild


def _summarize_marginal_isotropy(
    *,
    block_counts: dict[str, int],
    cell_yaw_counts: dict[str, dict[int, np.ndarray]],
    primitive_counts: dict[str, "Counter[str]"],
    yaw_bins: int,
) -> dict[str, dict[str, Any]]:
    """Per-collector marginal-isotropy stats for the per-scene summary.

    Whether each collector's marginal over (cell, heading) and over actions
    is close to isotropic is auditable from three numbers, without the raw
    trajectories: cell x yaw-bin coverage, mean per-cell yaw entropy, and
    primitive entropy.

    Cell-yaw counts are keyed (source -> cell_id -> 1-D histogram of
    length ``yaw_bins``). Primitive counts are keyed (source ->
    Counter[name]). Returns one dict per source.
    """

    result: dict[str, dict[str, Any]] = {}
    max_entropy = float(np.log(max(int(yaw_bins), 1)))
    for source, block_count in block_counts.items():
        cells = cell_yaw_counts.get(source, {})
        primitives = primitive_counts.get(source, Counter())
        distinct_cells = len(cells)
        distinct_cell_yaw = sum(int((arr > 0).sum()) for arr in cells.values())
        per_cell_entropies: list[float] = []
        for arr in cells.values():
            total = float(arr.sum())
            if total <= 0.0:
                continue
            probs = arr.astype(np.float64) / total
            nz = probs[probs > 0.0]
            per_cell_entropies.append(float(-(nz * np.log(nz)).sum()))
        mean_yaw_entropy = (
            float(np.mean(per_cell_entropies)) if per_cell_entropies else 0.0
        )
        prim_total = float(sum(primitives.values()))
        if prim_total > 0.0:
            probs = np.array(list(primitives.values()), dtype=np.float64) / prim_total
            nz = probs[probs > 0.0]
            prim_entropy = float(-(nz * np.log(nz)).sum())
        else:
            prim_entropy = 0.0
        result[source] = {
            "blocks": int(block_count),
            "distinct_cells": int(distinct_cells),
            "distinct_cell_yaw_bins": int(distinct_cell_yaw),
            "mean_per_cell_yaw_entropy_nats": mean_yaw_entropy,
            "max_per_cell_yaw_entropy_nats": max_entropy,
            "primitive_entropy_nats": prim_entropy,
            "primitive_counts": dict(primitives),
        }
    return result


# Default Go2 stance qpos for the 12 leg DOFs, ordered to match Genesis's
# joint-type grouping: [hip_FL, hip_FR, hip_RL, hip_RR,
#                       thigh_FL, thigh_FR, thigh_RL, thigh_RR,
#                       calf_FL, calf_FR, calf_RL, calf_RR].
DEFAULT_GO2_STANCE_RAD = np.array(
    [
        0.0, 0.0, 0.0, 0.0,
        0.9, 0.9, 0.9, 0.9,
        -1.8, -1.8, -1.8, -1.8,
    ],
    dtype=np.float32,
)

# Go2 entity-local DOF indices for the 12 leg joints (skipping the 6-DOF free
# root joint). Built scenes may shift these globally when other entities are
# inserted first; RolloutRunner resolves concrete indices from joint names.
DEFAULT_GO2_LEG_DOF_INDICES: tuple[int, ...] = tuple(range(6, 18))

DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER: tuple[str, ...] = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

# Upstream Genesis Go2 policy joint order maps to Genesis DOF starts in this
# order for the bundled Go2 URDF. Rollout application remains in sorted DOF
# order via DEFAULT_GO2_LEG_DOF_INDICES.
GENESIS_GO2_POLICY_DOF_INDICES: tuple[int, ...] = (
    7, 11, 15,  # FR hip/thigh/calf
    6, 10, 14,  # FL hip/thigh/calf
    9, 13, 17,  # RR hip/thigh/calf
    8, 12, 16,  # RL hip/thigh/calf
)


@dataclass(frozen=True)
class RolloutConfig:
    """Per-rollout knobs that aren't fixed by the platform manifest."""

    n_blocks: int = 200
    fall_z_threshold_m: float = 0.15
    # Reset if the body is tipped past this for `tip_recovery_blocks` consecutive
    # blocks. Catches "stuck on side" cases where the body rolls onto a hip
    # but the base center stays above fall_z_threshold_m (seen in
    # loop_alias_stress env 7: 86° tip, z=0.153 m, never reset, 38 s of
    # corrupted training data).
    tip_threshold_rad: float = math.radians(60.0)
    tip_recovery_blocks: int = 1
    # Privileged collection-time guard: if any non-recovery collector drives
    # the base this close to a wall, hand off to the recovery curriculum for
    # a short backout/pivot sequence. This is a data-generation label, not a
    # deployed-model input. The default sits just above the Go2 footprint
    # half-width (~0.15 m) so it only fires when the body is genuinely
    # grazing a wall — earlier 0.55 m exceeded every maze corridor
    # half-width (0.25–0.46 m), so collectors got handed off to recovery
    # on every block and reversed instead of exploring.
    recovery_interlock_clearance_m: float = 0.20
    recovery_interlock_blocks: int = 16
    # Spawn-time wall-clearance floor enforced for *every* reset, whether the
    # pose is sampled (``randomize_spawn_pose=True``) or read from the scene
    # manifest. Sits above the Go2 footprint half-diagonal (~0.36 m) and below
    # the runtime recovery interlock so a fresh episode never starts already
    # wedged between walls (data spec §16). If the manifest spawn fails the
    # floor, the rollout falls back to ``SceneGraph.sample_spawn_pose`` and
    # logs the override so fixed-spawn modes (e.g. long_route_teacher) fail
    # loudly instead of producing stuck-robot frames.
    spawn_clearance_floor_m: float = 0.45
    out_of_bounds_pad_m: float = 0.5
    seed: int = 0
    leg_dof_indices: tuple[int, ...] = DEFAULT_GO2_LEG_DOF_INDICES
    stance_qpos_rad: tuple[float, ...] = tuple(DEFAULT_GO2_STANCE_RAD.tolist())
    backend_id: str = "genesis_tier_a"
    log_progress_every_blocks: int = 50
    foot_contact_source: str = "genesis"
    # Collection-policy controls. ``collector_mix`` is a name->share map; the
    # rollout looks each name up in the policy registry built from
    # :func:`go2nav.corpus.collectors.build_default_policies`. None means
    # "use the §13 default mix". An empty dict forces the legacy uniform
    # random-primitive sampler.
    collector_mix: dict[str, float] | None = None
    # Per-episode spawn-pose randomization. When True (default), every reset
    # samples a random reachable cell and random yaw; when False the rollout
    # falls back to the manifest spawn.
    randomize_spawn_pose: bool = True


_SPAWN_YAW_MIN_FREE_RAY_M = 0.60

# Minimum blocks an env must spend in an episode before a route-completion
# reset can fire — guards against reset thrash if a scene's beacons happen to
# sit right next to the spawn (claimed in the first block or two).
_MIN_BLOCKS_BEFORE_COMPLETE_RESET = 2

# Adaptive spawn-floor relaxation. The configured ``spawn_clearance_floor_m``
# (0.45) was tuned for the wide-celled legacy mazes. Rooms-and-corridors
# mazes have ~0.28 m corridor clearance, so at 0.45 only the few
# high-clearance room-interior cells qualify — collapsing every env onto a
# handful of identical spawns (the deterministic route teacher then traces
# identical trajectories) and parking spawns on top of beacons instead of in
# the corridors. To keep spawns diverse *and* corridor-biased, relax the
# floor in steps until at least ``_MIN_SPAWN_CANDIDATES`` cells qualify,
# never going below ``_SPAWN_FLOOR_HARD_MIN_M`` (the grid inflation). A
# generous minimum is what pulls the floor below the room-cell clearance
# plateau so corridor cells re-enter the candidate set.
_MIN_SPAWN_CANDIDATES = 24
_SPAWN_FLOOR_HARD_MIN_M = 0.20
_SPAWN_FLOOR_RELAX_STEP_M = 0.05


def _relax_spawn_floor(
    scene_graph: Any,
    candidate_cells: list[int],
    configured_floor_m: float,
    *,
    min_candidates: int = _MIN_SPAWN_CANDIDATES,
    hard_min_m: float = _SPAWN_FLOOR_HARD_MIN_M,
    step_m: float = _SPAWN_FLOOR_RELAX_STEP_M,
) -> tuple[tuple[int, ...], float]:
    """Return ``(passing_cells, effective_floor)`` after adaptive relaxation.

    Walks the floor down from ``configured_floor_m`` to ``hard_min_m`` in
    ``step_m`` steps and stops at the highest floor that admits at least
    ``min_candidates`` cells. If even the hard minimum cannot reach the
    target (small scene), returns whatever passes the hard minimum — or,
    failing that, all candidate cells.
    """

    clearances = {
        cid: float(scene_graph.clearance_to_walls(scene_graph.cell_center(int(cid))))
        for cid in candidate_cells
        if 0 <= int(cid) < scene_graph.n_nodes
    }
    floor = float(configured_floor_m)
    best_passing: tuple[int, ...] = ()
    while floor >= hard_min_m - 1e-9:
        passing = tuple(c for c, cl in clearances.items() if cl >= floor)
        if len(passing) >= int(min_candidates):
            return passing, floor
        best_passing = passing  # remember the most permissive non-empty set
        floor -= step_m
    passing_hard = tuple(c for c, cl in clearances.items() if cl >= hard_min_m)
    if passing_hard:
        return passing_hard, hard_min_m
    return (best_passing or tuple(clearances.keys())), hard_min_m


def _wxyz_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _align_spawn_yaw(
    xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    planning_grid: Any | None,
    min_free_ray_m: float,
) -> tuple[tuple[float, float, float, float], float]:
    """Snap the spawn yaw to the longest free ray, if the original yaw is wedged.

    Returns ``(new_quat_wxyz, chosen_free_ray_m)``. When ``planning_grid`` is
    ``None`` or the original yaw already has more than ``min_free_ray_m`` of
    free space ahead, returns the input quat unchanged.
    """

    if planning_grid is None:
        return quat_wxyz, float("inf")
    from go2nav.corpus.grid import best_free_yaw

    current_yaw = _yaw_from_quat_wxyz(
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    cos_y = math.cos(current_yaw)
    sin_y = math.sin(current_yaw)
    bx, by = float(xyz[0]), float(xyz[1])
    step = 0.25 * planning_grid.cell_size_m
    max_steps = max(1, int(math.ceil(float(min_free_ray_m) / step)))
    current_free = 0.0
    for s in range(1, max_steps + 1):
        d = s * step
        if not planning_grid.is_free((bx + cos_y * d, by + sin_y * d)):
            break
        current_free = d
    if current_free >= float(min_free_ray_m):
        return quat_wxyz, current_free

    best_yaw, best_free = best_free_yaw(
        planning_grid, (bx, by), n_rays=24, max_m=max(2.0, float(min_free_ray_m) * 2.0)
    )
    if best_free <= current_free:
        return quat_wxyz, current_free
    return _wxyz_from_yaw(best_yaw), best_free


def _select_spawn_pose(
    *,
    scene_graph: Any | None,
    manifest_xyz: tuple[float, float, float],
    manifest_quat_wxyz: tuple[float, float, float, float],
    spawn_z_m: float,
    randomize: bool,
    clearance_floor_m: float,
    rng: random.Random,
    restrict_to_cells: tuple[int, ...] | None = None,
    planning_grid: Any | None = None,
    yaw_align_min_free_ray_m: float = _SPAWN_YAW_MIN_FREE_RAY_M,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    int,
    str,
]:
    """Resolve a spawn pose that satisfies the wall-clearance floor.

    Returns ``(xyz, quat_wxyz, cell_id, source)`` where ``source`` is one of:

    * ``"randomized"``  — pose sampled from the scene graph.
    * ``"manifest"``    — manifest spawn passed the floor (or no graph).
    * ``"safety_fallback"`` — manifest spawn failed the floor; rolled to a
      sampled pose so the episode does not start wedged.

    When ``planning_grid`` is provided, the yaw is re-aligned to the longest
    free ray from the resolved xy if the original yaw has less than
    ``yaw_align_min_free_ray_m`` of free space ahead — this prevents the
    common "spawn cell is fine but the robot is facing into a wall" failure
    in narrow corridors where the manifest spawn quat (always identity)
    points perpendicular to the corridor.

    Scenes without a graph (legacy ScenePack) bypass the gate and return the
    manifest pose unchanged — the caller is expected to validate clearance
    elsewhere in that path.
    """

    manifest_xyz_t = (float(manifest_xyz[0]), float(manifest_xyz[1]), float(manifest_xyz[2]))
    manifest_quat_t = (
        float(manifest_quat_wxyz[0]),
        float(manifest_quat_wxyz[1]),
        float(manifest_quat_wxyz[2]),
        float(manifest_quat_wxyz[3]),
    )

    if scene_graph is None:
        return manifest_xyz_t, manifest_quat_t, -1, "manifest"

    # Resolve the spawn-candidate set with adaptive floor relaxation. The
    # configured floor is relaxed per-scene until at least
    # ``_MIN_SPAWN_CANDIDATES`` cells qualify (down to the grid inflation),
    # so narrow-corridor room mazes don't collapse onto the handful of
    # high-clearance room-interior cells (which destroys per-env spawn
    # diversity and parks spawns on beacons). See ``_relax_spawn_floor``.
    base_cells = (
        [int(c) for c in restrict_to_cells]
        if restrict_to_cells
        else list(range(scene_graph.n_nodes))
    )
    passing, effective_floor = _relax_spawn_floor(
        scene_graph, base_cells, float(clearance_floor_m)
    )
    if restrict_to_cells:
        restrict_to_cells = passing if passing else tuple(base_cells)

    if randomize:
        xyz, wxyz, cell_id = scene_graph.sample_spawn_pose(
            rng,
            clearance_floor_m=effective_floor,
            spawn_z_m=float(spawn_z_m),
            restrict_to_cells=restrict_to_cells,
        )
        xyz_t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        quat_t = (float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3]))
        quat_t, _free = _align_spawn_yaw(
            xyz_t, quat_t, planning_grid, yaw_align_min_free_ray_m
        )
        return xyz_t, quat_t, int(cell_id), "randomized"

    manifest_cell_id = int(scene_graph.locate(manifest_xyz_t[:2]).cell_id)
    manifest_in_restriction = (
        restrict_to_cells is not None and manifest_cell_id in restrict_to_cells
    )
    manifest_clearance = float(
        scene_graph.clearance_to_walls((manifest_xyz_t[0], manifest_xyz_t[1]))
    )
    if manifest_in_restriction or manifest_clearance >= float(clearance_floor_m):
        if restrict_to_cells is not None and not manifest_in_restriction:
            pass  # fall through to safety_fallback
        else:
            aligned_quat, _free = _align_spawn_yaw(
                manifest_xyz_t, manifest_quat_t, planning_grid, yaw_align_min_free_ray_m
            )
            return manifest_xyz_t, aligned_quat, -1, "manifest"

    xyz, wxyz, cell_id = scene_graph.sample_spawn_pose(
        rng,
        clearance_floor_m=effective_floor,
        spawn_z_m=float(spawn_z_m),
        restrict_to_cells=restrict_to_cells,
    )
    xyz_t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    quat_t = (float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3]))
    quat_t, _free = _align_spawn_yaw(
        xyz_t, quat_t, planning_grid, yaw_align_min_free_ray_m
    )
    return xyz_t, quat_t, int(cell_id), "safety_fallback"


def _is_route_like_policy(policy: Any) -> bool:
    """Return True for policies that own route progress and local recovery."""

    return callable(getattr(policy, "visited_landmark_cells", None))


def _iter_route_like_landmark_cells(
    collectors: dict[str, Any],
    env_idx: int,
) -> Iterable[int]:
    """Yield landmark cells claimed by route-like collectors for one env."""

    for collector in collectors.values():
        visited_cells = getattr(collector, "visited_landmark_cells", None)
        if not callable(visited_cells):
            continue
        for cell in visited_cells(int(env_idx)):
            yield int(cell)


@dataclass
class GenesisGo2PPOPolicy:
    """RSL-RL Go2 PPO policy adapter for the Genesis bulk rollout loop.

    The trained actor consumes the upstream Genesis Go2 observation layout:
    body angular velocity, projected gravity, scaled command, joint position
    error, joint velocity, and previous action. It emits 12 position-offset
    actions in the upstream policy joint-name order. This adapter returns
    absolute joint targets in the rollout DOF order expected by
    :meth:`RolloutRunner._apply_joint_targets`.
    """

    checkpoint_path: str | Path
    cfg_path: str | Path | None = None
    rollout_leg_dof_indices: tuple[int, ...] = DEFAULT_GO2_LEG_DOF_INDICES
    policy_dof_indices: tuple[int, ...] = GENESIS_GO2_POLICY_DOF_INDICES
    device: str = "cpu"
    simulate_action_latency: bool = True
    deduplicate_exact_observation_rows: bool = False

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path)
        self.cfg_path = Path(self.cfg_path) if self.cfg_path is not None else self.checkpoint_path.parent / "cfgs.pkl"
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"missing PPO checkpoint: {self.checkpoint_path}")
        if not self.cfg_path.is_file():
            raise FileNotFoundError(f"missing PPO cfgs.pkl: {self.cfg_path}")

        with self.cfg_path.open("rb") as f:
            env_cfg, obs_cfg, _reward_cfg, _command_cfg, train_cfg = pickle.load(f)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.train_cfg = train_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.command_scale = np.array(
            [
                float(self.obs_scales["lin_vel"]),
                float(self.obs_scales["lin_vel"]),
                float(self.obs_scales["ang_vel"]),
            ],
            dtype=np.float32,
        )
        self.action_scale = float(env_cfg["action_scale"])
        self.default_dof_pos_policy = np.array(
            [float(env_cfg["default_joint_angles"][name]) for name in env_cfg["joint_names"]],
            dtype=np.float32,
        )

        # For the Genesis-bundled Go2 URDF, these policy joint names map to
        # DOF starts 6..17. Keep the mapping explicit so a future URDF change
        # fails loudly instead of silently permuting actions.
        self.policy_joint_names = tuple(str(name) for name in env_cfg["joint_names"])
        if len(self.policy_dof_indices) != len(self.policy_joint_names):
            raise ValueError(
                f"policy_dof_indices has {len(self.policy_dof_indices)} entries; "
                f"expected {len(self.policy_joint_names)}"
            )
        self._rollout_from_policy = np.array(
            [self.policy_dof_indices.index(int(dof_idx)) for dof_idx in self.rollout_leg_dof_indices],
            dtype=np.int64,
        )
        self._policy_from_rollout = np.array(
            [self.rollout_leg_dof_indices.index(int(dof_idx)) for dof_idx in self.policy_dof_indices],
            dtype=np.int64,
        )
        self.reset_stance_rad = self.default_dof_pos_policy[self._rollout_from_policy].astype(
            np.float32,
            copy=True,
        )

        self._device = self.device
        self._policy = self._load_policy()
        if self.deduplicate_exact_observation_rows and bool(
            getattr(self._policy, "is_recurrent", False)
        ):
            raise ValueError(
                "exact observation-row deduplication requires a non-recurrent policy"
            )
        self._last_actions: np.ndarray | None = None

    @classmethod
    def from_platform_manifest(
        cls,
        platform_manifest: dict[str, Any],
        workspace_root: str | Path,
        **kwargs: Any,
    ) -> "GenesisGo2PPOPolicy":
        """Create the adapter from ``locomotion.policy_artifact`` metadata."""

        artifact = platform_manifest.get("locomotion", {}).get("policy_artifact", {})
        checkpoint = artifact.get("path")
        if not checkpoint:
            raise ValueError("platform manifest missing locomotion.policy_artifact.path")
        root = Path(workspace_root)
        cfg_path = artifact.get("cfg_path")
        return cls(
            checkpoint_path=_resolve_manifest_path(root, checkpoint),
            cfg_path=_resolve_manifest_path(root, cfg_path) if cfg_path else None,
            **kwargs,
        )


    def _load_policy(self):
        try:
            import torch
            from rsl_rl.runners import OnPolicyRunner
            from tensordict import TensorDict
        except ImportError as exc:  # pragma: no cover - depends on training venv
            raise RuntimeError(
                "GenesisGo2PPOPolicy requires torch, tensordict, and rsl-rl-lib. "
                "Use the Genesis ROCm training venv."
            ) from exc

        class _DummyEnv:
            def __init__(self, env_cfg: dict[str, Any], device: str) -> None:
                self.num_envs = 1
                self.num_actions = int(env_cfg["num_actions"])
                self.max_episode_length = 1000
                self.episode_length_buf = torch.zeros((1,), dtype=torch.int32, device=device)
                self.device = device
                self.cfg = env_cfg

            def get_observations(self):
                return TensorDict(
                    {"policy": torch.zeros((1, 45), dtype=torch.float32, device=self.device)},
                    batch_size=[1],
                )


        runner = OnPolicyRunner(_DummyEnv(self.env_cfg, self._device), self.train_cfg, log_dir=None, device=self._device)
        runner.load(str(self.checkpoint_path), map_location=self._device)
        policy = runner.get_inference_policy(device=self._device)
        policy.eval()
        return policy

    def reset(self, envs_idx: list[int] | None = None) -> None:
        """Clear action-history latency state after rollout resets."""

        if self._last_actions is None:
            return
        if envs_idx is None:
            self._last_actions.fill(0.0)
            return
        if envs_idx:
            self._last_actions[np.asarray(envs_idx, dtype=np.int64)] = 0.0

    def validate_rollout_robot(self, robot: Any) -> None:
        """Verify the built Genesis robot matches the policy joint contract."""

        joints = getattr(robot, "joints", None)
        if joints is None:
            return
        joint_by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
        missing = [name for name in self.policy_joint_names if name not in joint_by_name]
        if missing:
            raise ValueError(
                "Genesis Go2 PPO policy requires joints missing from the rollout robot: "
                + ", ".join(missing)
            )
        actual_policy_dofs = tuple(
            _entity_local_dof_index(robot, joint_by_name[name]) for name in self.policy_joint_names
        )
        expected_policy_dofs = tuple(int(idx) for idx in self.policy_dof_indices)
        if actual_policy_dofs != expected_policy_dofs:
            pairs = ", ".join(
                f"{name}: expected {expected}, actual {actual}"
                for name, expected, actual in zip(
                    self.policy_joint_names,
                    expected_policy_dofs,
                    actual_policy_dofs,
                )
                if expected != actual
            )
            raise ValueError(
                "Genesis Go2 PPO policy joint mapping does not match rollout robot "
                f"({pairs})"
            )

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        try:
            import torch
            from tensordict import TensorDict
        except ImportError as exc:  # pragma: no cover - depends on training venv
            raise RuntimeError(
                "GenesisGo2PPOPolicy requires torch and tensordict at inference time."
            ) from exc

        n_envs = int(observation["command"].shape[0])
        if self._last_actions is None or self._last_actions.shape[0] != n_envs:
            self._last_actions = np.zeros((n_envs, len(self.policy_joint_names)), dtype=np.float32)

        obs_np = self._build_policy_observation(observation)
        if self.deduplicate_exact_observation_rows:
            # Counterfactual lanes with byte-identical histories must receive
            # byte-identical policy outputs.  Evaluate each distinct row as a
            # one-row batch so CPU kernel rounding cannot depend on lane index.
            unique_rows: list[np.ndarray] = []
            inverse: list[int] = []
            index_by_bytes: dict[bytes, int] = {}
            for row in np.ascontiguousarray(obs_np, dtype=np.float32):
                key = row.tobytes(order="C")
                unique_index = index_by_bytes.get(key)
                if unique_index is None:
                    unique_index = len(unique_rows)
                    index_by_bytes[key] = unique_index
                    unique_rows.append(row.copy())
                inverse.append(unique_index)
            unique_actions: list[np.ndarray] = []
            with torch.no_grad():
                for row in unique_rows:
                    row_tensor = torch.as_tensor(
                        row[None, :], dtype=torch.float32, device=self._device
                    )
                    row_dict = TensorDict({"policy": row_tensor}, batch_size=[1])
                    row_actions = self._policy(
                        row_dict, stochastic_output=False
                    ).detach().cpu().numpy().astype(np.float32, copy=False)
                    if row_actions.shape != (1, len(self.policy_joint_names)):
                        raise ValueError(
                            "PPO policy returned a malformed one-row action"
                        )
                    unique_actions.append(row_actions[0].copy())
            actions = np.stack(
                [unique_actions[index] for index in inverse], axis=0
            ).astype(np.float32, copy=False)
        else:
            obs_tensor = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self._device
            )
            obs_dict = TensorDict({"policy": obs_tensor}, batch_size=[n_envs])
            with torch.no_grad():
                actions = self._policy(
                    obs_dict, stochastic_output=False
                ).detach().cpu().numpy().astype(np.float32, copy=False)
        if actions.shape != self._last_actions.shape:
            raise ValueError(f"PPO policy returned {actions.shape}; expected {self._last_actions.shape}")

        exec_actions = self._last_actions if self.simulate_action_latency else actions
        target_policy_order = exec_actions * self.action_scale + self.default_dof_pos_policy
        self._last_actions = actions.copy()
        return target_policy_order[:, self._rollout_from_policy].astype(np.float32, copy=False)

    def _build_policy_observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        quat_xyzw = np.asarray(observation["base_quat_xyzw"], dtype=np.float32)
        ang_body = _rotate_world_to_body(
            np.asarray(observation["base_ang_vel_world"], dtype=np.float32),
            quat_xyzw,
        )
        gravity_world = np.zeros_like(ang_body, dtype=np.float32)
        gravity_world[:, 2] = -1.0
        projected_gravity = _rotate_world_to_body(gravity_world, quat_xyzw)
        joint_pos = np.asarray(observation["joint_pos"], dtype=np.float32)[:, self._policy_from_rollout]
        joint_vel = np.asarray(observation["joint_vel"], dtype=np.float32)[:, self._policy_from_rollout]
        command = np.asarray(observation["command"], dtype=np.float32) * self.command_scale
        return np.concatenate(
            (
                ang_body * float(self.obs_scales["ang_vel"]),
                projected_gravity,
                command,
                (joint_pos - self.default_dof_pos_policy) * float(self.obs_scales["dof_pos"]),
                joint_vel * float(self.obs_scales["dof_vel"]),
                self._last_actions,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)


@dataclass
class _BlockTrajectory:
    """Per-block requested-vs-executed (n_envs, T, 3) arrays."""

    requested: np.ndarray
    executed: np.ndarray
    clipped: np.ndarray  # (n_envs,) bool


def stable_scene_number(scene_id: str) -> int:
    """The scene's 31-bit number in the recorded messages. (The May recordings used Python's per-process string
    hash, which no rerun can reproduce; this is a fixed SHA-256-based number instead.)"""
    return int.from_bytes(hashlib.sha256(scene_id.encode()).digest()[:4], 'big') & 0x7FFF_FFFF


class RolloutRunner:
    """Drives one scene's full rollout."""

    def __init__(
        self,
        build: SceneBuild,
        policy: "GenesisGo2PPOPolicy",
        registry: PrimitiveRegistry,
        safety_limits: SafetyLimits,
        *,
        config: RolloutConfig = RolloutConfig(),
    ) -> None:
        self.build = build
        self.policy = policy
        self.registry = registry
        self.safety = safety_limits
        self.config = config
        self.pack: ScenePack = build.pack
        self.timing = build.pack.timing
        self.n_envs = int(build.n_envs)

        self._physics_steps_per_policy = self.timing.policy_decimation
        self._policy_steps_per_command_tick = self.timing.command_ticks_per_block
        self._command_dt_ns = int(round(self.timing.command_dt_s * 1e9))
        self._policy_dt_ns = int(round(self.timing.policy_dt_s * 1e9))
        self._block_size = int(registry.block_size)
        self._leg_dof_idx = _resolve_rollout_leg_dof_indices(build.robot, config.leg_dof_indices)
        self._stance = np.array(config.stance_qpos_rad, dtype=np.float32)
        if config.foot_contact_source not in {"genesis", "zero"}:
            raise ValueError(
                "foot_contact_source must be 'genesis' or 'zero'; "
                f"got {config.foot_contact_source!r}"
            )
        validate_robot = getattr(self.policy, "validate_rollout_robot", None)
        if callable(validate_robot):
            validate_robot(build.robot)

        scene_id_int = stable_scene_number(self.pack.scene_id)
        self.episode_states: list[EpisodeState] = make_episode_states(
            self.n_envs,
            scene_id=scene_id_int,
            scene_family=self.pack.family,
            split=self.pack.split,
            manifest_sha256=self.pack.manifest_sha256,
        )
        self._last_executed = np.zeros((self.n_envs, 3), dtype=np.float32)
        self._sim_time_ns = 0
        self._sequence_id_counter = 0
        self._rng = np.random.default_rng(int(config.seed))
        # Per-env spawn pose tracking. Populated by ``_reset_robot_to_spawn``
        # so the reset-event log records the actual cell + yaw the env was
        # respawned at (data spec §16: reset integrity).
        self._spawn_xyz_per_env = np.tile(
            np.asarray(self.pack.robot.spawn_xyz_m, dtype=np.float32),
            (self.n_envs, 1),
        )
        self._spawn_quat_wxyz_per_env = np.tile(
            np.asarray(self.pack.robot.spawn_quat_wxyz, dtype=np.float32),
            (self.n_envs, 1),
        )
        # Episode-scoped block counter for collector observations.
        self._blocks_in_episode = np.zeros(self.n_envs, dtype=np.int64)
        # Consecutive-block tip counter used by the tipped-body reset trigger.
        self._consecutive_tipped_blocks = np.zeros(self.n_envs, dtype=np.int64)
        self._recovery_interlock_blocks_remaining = np.zeros(self.n_envs, dtype=np.int64)
        # Per-env planner-quality metrics surfaced via run() — used by the
        # physics-only profiler so we can tune teacher logic without
        # re-rendering.
        self._per_env_path_length_m = np.zeros(self.n_envs, dtype=np.float64)
        self._per_env_last_xy: list[tuple[float, float] | None] = [None] * self.n_envs
        self._per_env_cells_visited: list[set[int]] = [set() for _ in range(self.n_envs)]
        self._per_env_primitive_counts: list[dict[str, int]] = [
            {} for _ in range(self.n_envs)
        ]
        self._per_env_source_counts: list[dict[str, int]] = [
            {} for _ in range(self.n_envs)
        ]
        self._per_env_goals_targeted: list[set[int]] = [set() for _ in range(self.n_envs)]
        self._per_env_target_changes: np.ndarray = np.zeros(self.n_envs, dtype=np.int64)
        self._per_env_prev_target: list[int] = [-1] * self.n_envs
        self._per_env_recovery_handoffs: np.ndarray = np.zeros(self.n_envs, dtype=np.int64)
        self._per_env_achieved_landmarks: list[set[str]] = [set() for _ in range(self.n_envs)]

        # Per-collector marginal-isotropy counters surfaced via run() for the
        # data-isotropy audit (`_summarize_marginal_isotropy`).
        self._isotropy_yaw_bins: int = 12
        self._isotropy_block_counts: dict[str, int] = {}
        self._isotropy_cell_yaw_counts: dict[str, dict[int, np.ndarray]] = {}
        self._isotropy_primitive_counts: dict[str, Counter[str]] = {}

        # Build the collector bench + scheduler. ``scene_graph`` may be None
        # on legacy ScenePack instances built in tests; in that case fall
        # back to the legacy uniform random sampler.
        from go2nav.corpus.collectors import (
            DEFAULT_COLLECTION_MIX,
            EpisodeScheduler,
            build_default_policies,
        )

        self._scene_graph = getattr(self.pack, "scene_graph", None)
        self._landmark_cell_to_id: dict[int, str] = {
            int(cell): str(name)
            for name, cell in getattr(self._scene_graph, "landmark_cells", ())
        }
        self._collectors = build_default_policies(self.registry, n_envs=self.n_envs)
        if config.collector_mix is not None and not config.collector_mix:
            # Empty dict => uniform primitive sampling (the old behaviour).
            mix = {"primitive_curriculum": 1.0}
        else:
            mix = dict(config.collector_mix or DEFAULT_COLLECTION_MIX)
        # If the scene has no graph available, route/frontier/recovery cannot
        # run, so collapse the mix to policies that don't need it. This keeps
        # legacy paths working without forcing every caller to populate a
        # SceneGraph.
        if self._scene_graph is None:
            graph_free = {"primitive_curriculum", "ou_noise"}
            mix = {k: v for k, v in mix.items() if k in graph_free}
            if not mix:
                mix = {"primitive_curriculum": 1.0}
        self._scheduler = EpisodeScheduler(
            policies=self._collectors,
            shares=mix,
            rng=self._rng,
            n_envs=self.n_envs,
        )
        self._spawn_rng = random.Random(int(config.seed) * 0x9E37 ^ 0xA5A5A5A5)
        # Planning grid used to align the spawn yaw with the longest free
        # ray from the spawn xy. Built lazily on first reset; reused across
        # episodes for the same scene.
        self._spawn_planning_grid: Any | None = None
        # Count of route episodes a non-revisit teacher finished (claimed
        # every beacon) and was respawned for, per env. Reported in the
        # summary so the data pipeline can see episodes-per-scene.
        self._per_env_route_completions = np.zeros(self.n_envs, dtype=np.int64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def run(self, writer: Any) -> dict[str, Any]:
        """Run the configured number of command blocks, streaming to ``writer``.

        Returns a stats dict for the caller's summary aggregation.
        """

        self._reset_robot_to_spawn(envs_idx=None)
        # All envs draw their first collector assignment via the scheduler so
        # the §13 mix is honored from block 0.
        for env_idx in range(self.n_envs):
            self._scheduler.on_episode_reset(env_idx)
            self._blocks_in_episode[env_idx] = 0
        self._emit_initial_reset_events(writer)
        wall_start = time.time()

        for block_idx in range(self.config.n_blocks):
            requested_block, choices = self._collect_block()
            block = self._clip_block(requested_block)
            self._emit_command_block_request(requested_block, choices, writer)

            for tick_idx in range(self._block_size):
                target_cmd = block.executed[:, tick_idx]
                self._step_command_tick(target_cmd)
                self._emit_per_tick_records(writer)
                writer.write_clock(self._sim_time_ns)

            self._emit_executed_command_block(block, choices, writer)
            self._last_executed = block.executed[:, -1, :]
            self._blocks_in_episode += 1
            self._check_and_reset_fallen_envs(writer)
            self._check_and_reset_completed_envs(writer)

            if (
                self.config.log_progress_every_blocks
                and (block_idx + 1) % self.config.log_progress_every_blocks == 0
            ):
                elapsed = time.time() - wall_start
                print(
                    f"[rollout {self.pack.scene_id}] block {block_idx + 1}/{self.config.n_blocks} "
                    f"sim_time={self._sim_time_ns / 1e9:.2f}s wall={elapsed:.1f}s"
                )

        wall_total = time.time() - wall_start
        per_env_metrics = []
        available_landmarks = set(self._landmark_cell_to_id.values())
        for env_idx in range(self.n_envs):
            achieved_landmarks = set(self._per_env_achieved_landmarks[env_idx])
            relaxed_count, relaxed_events = self._aggregate_relaxed_claims(env_idx)
            per_env_metrics.append(
                {
                    "env_idx": env_idx,
                    "path_length_m": float(self._per_env_path_length_m[env_idx]),
                    "cells_visited": len(self._per_env_cells_visited[env_idx]),
                    "unique_goals_targeted": len(self._per_env_goals_targeted[env_idx]),
                    "goal_target_changes": int(self._per_env_target_changes[env_idx]),
                    "recovery_interlock_handoffs": int(
                        self._per_env_recovery_handoffs[env_idx]
                    ),
                    "route_completions": int(self._per_env_route_completions[env_idx]),
                    "beacons_available": len(available_landmarks),
                    "beacons_achieved": len(achieved_landmarks),
                    "beacon_claim_ratio": (
                        len(achieved_landmarks) / len(available_landmarks)
                        if available_landmarks
                        else 1.0
                    ),
                    "achieved_landmark_ids": sorted(achieved_landmarks),
                    "unclaimed_landmark_ids": sorted(
                        available_landmarks - achieved_landmarks
                    ),
                    "relaxed_claim_count": relaxed_count,
                    "relaxed_claim_events": relaxed_events,
                    "primitive_counts": dict(self._per_env_primitive_counts[env_idx]),
                    "command_source_counts": dict(
                        self._per_env_source_counts[env_idx]
                    ),
                }
            )
        return {
            "scene_id": self.pack.scene_id,
            "n_envs": self.n_envs,
            "n_blocks": self.config.n_blocks,
            "command_ticks": self.config.n_blocks * self._block_size,
            "wall_seconds": wall_total,
            "final_sim_time_s": self._sim_time_ns / 1e9,
            "collector_mix_realized": self._scheduler.realized_mix(),
            "marginal_isotropy": _summarize_marginal_isotropy(
                block_counts=self._isotropy_block_counts,
                cell_yaw_counts=self._isotropy_cell_yaw_counts,
                primitive_counts=self._isotropy_primitive_counts,
                yaw_bins=self._isotropy_yaw_bins,
            ),
            "per_env_metrics": per_env_metrics,
        }

    def _aggregate_relaxed_claims(
        self, env_idx: int
    ) -> tuple[int, list[dict[str, Any]]]:
        """Pull relaxed-claim telemetry from any route-like collector.

        Returns total count and the event list. Collectors that don't
        expose the accessors (anything except route_teacher) contribute
        zero, so this stays a no-op for non-route mixes.
        """

        total = 0
        events: list[dict[str, Any]] = []
        for collector in self._collectors.values():
            count_fn = getattr(collector, "relaxed_claim_count", None)
            events_fn = getattr(collector, "relaxed_claim_events", None)
            if not callable(count_fn) or not callable(events_fn):
                continue
            total += int(count_fn(env_idx))
            events.extend(dict(event) for event in events_fn(env_idx))
        return total, events

    # ------------------------------------------------------------------
    # Collector observation + block selection
    # ------------------------------------------------------------------

    def _collect_block(self) -> tuple[np.ndarray, list]:
        """Observe per-env state and call each env's assigned collector.

        Returns the requested ``(n_envs, T, 3)`` tape for the upcoming block
        plus a parallel list of :class:`BlockChoice` for audit emission.
        """

        from go2nav.corpus.collectors.base import BlockChoice, EnvObservation

        robot = self.build.robot
        pos_arr = self._as_np(robot.get_pos()).astype(np.float32, copy=False)
        quat_wxyz = self._as_np(robot.get_quat()).astype(np.float32, copy=False)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr[None, :]
        if quat_wxyz.ndim == 1:
            quat_wxyz = quat_wxyz[None, :]
        yaws = np.array(
            [
                _yaw_from_quat_wxyz(
                    float(quat_wxyz[i, 0]),
                    float(quat_wxyz[i, 1]),
                    float(quat_wxyz[i, 2]),
                    float(quat_wxyz[i, 3]),
                )
                for i in range(self.n_envs)
            ],
            dtype=np.float32,
        )

        requested_block = np.zeros((self.n_envs, self._block_size, 3), dtype=np.float32)
        choices: list[BlockChoice] = []
        for env_idx in range(self.n_envs):
            base_xy = (float(pos_arr[env_idx, 0]), float(pos_arr[env_idx, 1]))
            if self._scene_graph is not None:
                hit = self._scene_graph.locate(base_xy)
                clearance = self._scene_graph.clearance_to_walls(base_xy)
                current_cell = hit.cell_id
                cell_dist = hit.distance_m
            else:
                current_cell = -1
                cell_dist = 0.0
                clearance = float("inf")

            state = self.episode_states[env_idx]
            observation = EnvObservation(
                env_idx=env_idx,
                base_xy_world=base_xy,
                base_yaw_world=float(yaws[env_idx]),
                current_cell_id=int(current_cell),
                nearest_cell_distance_m=float(cell_dist),
                clearance_to_walls_m=float(clearance),
                last_executed_cmd=(
                    float(self._last_executed[env_idx, 0]),
                    float(self._last_executed[env_idx, 1]),
                    float(self._last_executed[env_idx, 2]),
                ),
                episode_id=int(state.episode_id),
                episode_step=int(state.episode_step),
                block_idx_in_episode=int(self._blocks_in_episode[env_idx]),
            )
            policy = self._scheduler.policy_for(env_idx)
            # Route-like teachers already have privileged backout/stuck logic.
            # The rollout-level interlock can otherwise starve them in narrow
            # blocking scenes before they get a chance to replan.
            policy_is_route_like = _is_route_like_policy(policy)
            recovery_policy = self._collectors.get("recovery")
            if (
                recovery_policy is not None
                and getattr(policy, "name", "") != "recovery"
                and not policy_is_route_like
                and self._scene_graph is not None
                and float(self.config.recovery_interlock_clearance_m) > 0.0
                and (
                    observation.clearance_to_walls_m
                    <= float(self.config.recovery_interlock_clearance_m)
                )
                and self._recovery_interlock_blocks_remaining[env_idx] <= 0
            ):
                recovery_policy.on_episode_reset(env_idx)
                self._recovery_interlock_blocks_remaining[env_idx] = max(
                    1, int(self.config.recovery_interlock_blocks)
                )
                self._per_env_recovery_handoffs[env_idx] += 1

            if (
                recovery_policy is not None
                and getattr(policy, "name", "") != "recovery"
                and not policy_is_route_like
                and self._recovery_interlock_blocks_remaining[env_idx] > 0
            ):
                choice = recovery_policy.on_block(
                    observation=observation,
                    scene=self._scene_graph,
                    rng=self._rng,
                )
                self._recovery_interlock_blocks_remaining[env_idx] -= 1
            else:
                choice = policy.on_block(
                    observation=observation,
                    scene=self._scene_graph,
                    rng=self._rng,
                )
            requested_block[env_idx] = choice.requested_block
            choices.append(choice)

            # Metrics: accumulate displacement, visited cells, primitive +
            # source counts, and re-target events for the profiler.
            prev_xy = self._per_env_last_xy[env_idx]
            if prev_xy is not None:
                self._per_env_path_length_m[env_idx] += math.hypot(
                    base_xy[0] - prev_xy[0], base_xy[1] - prev_xy[1]
                )
            self._per_env_last_xy[env_idx] = base_xy
            if current_cell >= 0:
                self._per_env_cells_visited[env_idx].add(int(current_cell))

            # Beacon achievement tracking.
            if self._scene_graph is not None and current_cell >= 0:
                for cell in _iter_route_like_landmark_cells(self._collectors, env_idx):
                    landmark_id = self._landmark_cell_to_id.get(int(cell))
                    if landmark_id is not None:
                        self._per_env_achieved_landmarks[env_idx].add(landmark_id)

                # We reuse the logic from RouteTeacher._is_arrived to detect achievement.
                # Since the collector already ran, we can check if it considered itself arrived.
                if choice.command_source == "route_teacher" and choice.primitive_name == "hold":
                    # route_teacher emits 'hold' exactly once upon arrival.
                    # Verify it was a landmark goal.
                    goal_id = choice.route_target_id
                    for lm_name, lm_cell in self._scene_graph.landmark_cells:
                        if lm_cell == goal_id:
                            self._per_env_achieved_landmarks[env_idx].add(lm_name)
                            break

            counts = self._per_env_primitive_counts[env_idx]
            counts[choice.primitive_name] = counts.get(choice.primitive_name, 0) + 1
            source_counts = self._per_env_source_counts[env_idx]
            source_counts[choice.command_source] = (
                source_counts.get(choice.command_source, 0) + 1
            )

            # Per-collector marginal-isotropy counters. We bin yaw at the
            # block-request instant — the same cadence the encoder
            # ultimately consumes — and bucket primitives by name.
            source = choice.command_source
            self._isotropy_block_counts[source] = (
                self._isotropy_block_counts.get(source, 0) + 1
            )
            prim_counter = self._isotropy_primitive_counts.setdefault(
                source, Counter()
            )
            prim_counter[choice.primitive_name] += 1
            if current_cell >= 0:
                cell_map = self._isotropy_cell_yaw_counts.setdefault(source, {})
                yaw_norm = float(yaws[env_idx]) + math.pi
                bin_idx = int(
                    yaw_norm / (2.0 * math.pi) * self._isotropy_yaw_bins
                ) % self._isotropy_yaw_bins
                hist = cell_map.get(int(current_cell))
                if hist is None:
                    hist = np.zeros(self._isotropy_yaw_bins, dtype=np.int64)
                    cell_map[int(current_cell)] = hist
                hist[bin_idx] += 1
            target = int(choice.route_target_id)
            if target >= 0:
                self._per_env_goals_targeted[env_idx].add(target)
                if target != self._per_env_prev_target[env_idx]:
                    self._per_env_target_changes[env_idx] += 1
                    self._per_env_prev_target[env_idx] = target
        return requested_block, choices

    # ------------------------------------------------------------------
    # Inner loop steps
    # ------------------------------------------------------------------

    def _clip_block(self, requested_block: np.ndarray) -> _BlockTrajectory:
        executed, clipped = apply_safety_limits_batch(
            requested_block, self._last_executed, self.safety
        )
        return _BlockTrajectory(requested=requested_block, executed=executed, clipped=clipped)

    def _step_command_tick(self, target_cmd: np.ndarray) -> None:
        """Run ``policy_steps_per_command_tick`` policy steps for one 100 ms tick."""

        for _ in range(self._policy_steps_per_command_tick):
            self._step_policy_step(target_cmd)

    def _step_policy_step(self, target_cmd: np.ndarray) -> None:
        """Execute one policy step and its configured physics decimation."""

        obs = self._build_observation(target_cmd)
        joint_targets = self.policy.act(obs)
        self._apply_joint_targets(joint_targets)
        for _step in range(self._physics_steps_per_policy):
            self.build.scene.step()
        self._sim_time_ns += self._policy_dt_ns

    def _build_observation(self, target_cmd: np.ndarray) -> dict[str, np.ndarray]:
        robot = self.build.robot
        pos = self._as_np(robot.get_pos())
        quat_wxyz = self._as_np(robot.get_quat())
        lin_world = self._as_np(robot.get_vel())
        ang_world = self._as_np(robot.get_ang())
        joint_pos = self._as_np(robot.get_dofs_position(self._leg_dof_idx.tolist()))
        joint_vel = self._as_np(robot.get_dofs_velocity(self._leg_dof_idx.tolist()))
        quat_xyzw = np.stack(
            [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
            axis=-1,
        )
        return {
            "base_pos_world": pos.astype(np.float32),
            "base_quat_xyzw": quat_xyzw.astype(np.float32),
            "base_lin_vel_world": lin_world.astype(np.float32),
            "base_ang_vel_world": ang_world.astype(np.float32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "command": np.asarray(target_cmd, dtype=np.float32),
        }

    def _apply_joint_targets(self, joint_targets: np.ndarray) -> None:
        robot = self.build.robot
        targets = np.asarray(joint_targets, dtype=np.float32)
        if targets.shape[-1] != len(self._leg_dof_idx):
            raise ValueError(
                f"policy returned {targets.shape[-1]} joint targets; "
                f"expected {len(self._leg_dof_idx)}"
            )
        robot.control_dofs_position(targets, self._leg_dof_idx.tolist())

    def _emit_per_tick_records(self, writer: Any) -> None:
        """Write episode info, joint state, base state, odometry, IMU, mode and foot contacts per env at this
        command tick."""

        robot = self.build.robot
        pos_arr = self._as_np(robot.get_pos())
        quat_wxyz = self._as_np(robot.get_quat())
        lin_world = self._as_np(robot.get_vel())
        ang_world = self._as_np(robot.get_ang())
        joint_pos = self._as_np(robot.get_dofs_position(self._leg_dof_idx.tolist()))
        joint_vel = self._as_np(robot.get_dofs_velocity(self._leg_dof_idx.tolist()))
        quat_xyzw = np.stack(
            [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
            axis=-1,
        )
        contacts = self._extract_foot_contacts()

        for env_idx in range(self.n_envs):
            self.episode_states[env_idx].step()
            info_rec = self.episode_states[env_idx].episode_info(stamp_ns=self._sim_time_ns)
            writer.write_env(
                env_idx, "episode_info", messages.episode_info(info_rec), self._sim_time_ns
            )
            writer.write_env(
                env_idx,
                "joint_states",
                messages.joint_state(
                    joint_names=DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER,
                    positions=joint_pos[env_idx],
                    velocities=joint_vel[env_idx],
                    stamp_ns=self._sim_time_ns,
                ),
                self._sim_time_ns,
            )
            base = base_state_from_genesis(
                pos_world=pos_arr[env_idx],
                quat_xyzw=quat_xyzw[env_idx],
                linear_vel_body=self._world_to_body(lin_world[env_idx], quat_xyzw[env_idx]),
                angular_vel_body=self._world_to_body(ang_world[env_idx], quat_xyzw[env_idx]),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(env_idx, "base_state", messages.base_state(base), self._sim_time_ns)
            writer.write_env(
                env_idx,
                "odom",
                messages.odometry(base),
                self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "imu",
                messages.imu(
                    quat_xyzw=tuple(float(q) for q in quat_xyzw[env_idx]),
                    angular_vel_body=base.twist_body_angular,
                    linear_accel_body=(0.0, 0.0, 0.0),
                    stamp_ns=self._sim_time_ns,
                    frame_id="imu_link",
                ),
                self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "mode",
                messages.mode_state(
                    stamp_ns=self._sim_time_ns,
                    backend_id=str(self.config.backend_id),
                    controller_mode="cmd_vel",
                    gait_name="genesis_contract_ppo",
                    moving=True,
                    status_text="genesis bulk PPO rollout",
                ),
                self._sim_time_ns,
            )

            foot_rec = foot_contacts_record(
                tuple(contacts[env_idx].tolist()),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "foot_contacts", messages.foot_contacts(foot_rec), self._sim_time_ns
            )


    def _emit_command_block_request(
        self,
        requested_block: np.ndarray,
        choices: list,
        writer: Any,
    ) -> None:
        for env_idx in range(self.n_envs):
            sequence_id = self._next_sequence_id()
            req = requested_block[env_idx]  # (T, 3)
            choice = choices[env_idx]
            record = CommandBlockRecord(
                sequence_id=sequence_id,
                block_size=int(req.shape[0]),
                command_dt_s=float(self.timing.command_dt_s),
                primitive_name=str(choice.primitive_name),
                vx_body_mps=[float(v) for v in req[:, 0]],
                vy_body_mps=[float(v) for v in req[:, 1]],
                yaw_rate_radps=[float(v) for v in req[:, 2]],
                event_name="",
                event_allowed_in_training=False,
                command_source=str(choice.command_source),
                route_target_id=int(choice.route_target_id),
                next_waypoint_id=int(choice.next_waypoint_id),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "command_block", messages.command_block(record), self._sim_time_ns
            )

    def _emit_executed_command_block(
        self,
        block: _BlockTrajectory,
        choices: list,
        writer: Any,
    ) -> None:
        for env_idx in range(self.n_envs):
            requested = [tuple(map(float, block.requested[env_idx, t])) for t in range(self._block_size)]
            executed = [tuple(map(float, block.executed[env_idx, t])) for t in range(self._block_size)]
            sequence_id = self._sequence_id_counter - self.n_envs + env_idx
            record = reconstruct_executed_block(
                requested,
                executed,
                sequence_id=sequence_id,
                primitive_name=str(choices[env_idx].primitive_name),
                command_dt_s=float(self.timing.command_dt_s),
                clipped=bool(block.clipped[env_idx]),
                safety_overridden=False,
                controller_mode="cmd_vel",
                backend_id=str(self.config.backend_id),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx,
                "executed_command_block",
                messages.executed_command_block(record),
                self._sim_time_ns,
            )

    # ------------------------------------------------------------------
    # Reset handling
    # ------------------------------------------------------------------

    def _emit_initial_reset_events(self, writer: Any) -> None:
        for env_idx in range(self.n_envs):
            spawn_xyz = tuple(float(v) for v in self._spawn_xyz_per_env[env_idx])
            spawn_wxyz = tuple(float(v) for v in self._spawn_quat_wxyz_per_env[env_idx])
            event = self.episode_states[env_idx].reset(
                reason="initial_spawn",
                success=True,
                spawn_pose_xyz=spawn_xyz,
                spawn_pose_quat_xyzw=self._wxyz_to_xyzw(spawn_wxyz),
                stamp_ns=self._sim_time_ns,
            )
            writer.write_env(
                env_idx, "reset_event", messages.reset_event(event), self._sim_time_ns
            )

    def _check_and_reset_fallen_envs(self, writer: Any) -> None:
        pos = self._as_np(self.build.robot.get_pos())
        quat_wxyz = self._as_np(self.build.robot.get_quat())
        if pos.ndim == 1:
            pos = pos[None, :]
        if quat_wxyz.ndim == 1:
            quat_wxyz = quat_wxyz[None, :]
        fell = pos[:, 2] < float(self.config.fall_z_threshold_m)
        (xmin, ymin), (xmax, ymax) = self.pack.world_bounds_xy_m
        pad = float(self.config.out_of_bounds_pad_m)
        oob = (
            (pos[:, 0] < xmin - pad)
            | (pos[:, 0] > xmax + pad)
            | (pos[:, 1] < ymin - pad)
            | (pos[:, 1] > ymax + pad)
        )
        # Tipped-body trigger: robot is on its side (|roll| or |pitch| > limit)
        # for ``tip_recovery_blocks`` consecutive blocks. Reset criterion is
        # body-frame, so it fires whether or not z stayed above the fall floor.
        tip = np.array(
            [
                max(
                    abs(_roll_from_quat_wxyz(float(quat_wxyz[i, 0]), float(quat_wxyz[i, 1]), float(quat_wxyz[i, 2]), float(quat_wxyz[i, 3]))),
                    abs(_pitch_from_quat_wxyz(float(quat_wxyz[i, 0]), float(quat_wxyz[i, 1]), float(quat_wxyz[i, 2]), float(quat_wxyz[i, 3]))),
                )
                for i in range(quat_wxyz.shape[0])
            ],
            dtype=np.float32,
        )
        tip_limit = float(self.config.tip_threshold_rad)
        tipped_now = tip > tip_limit
        self._consecutive_tipped_blocks = np.where(
            tipped_now,
            self._consecutive_tipped_blocks + 1,
            np.zeros_like(self._consecutive_tipped_blocks),
        )
        stuck_tipped = self._consecutive_tipped_blocks >= int(self.config.tip_recovery_blocks)
        to_reset = np.where(fell | oob | stuck_tipped)[0]
        if to_reset.size == 0:
            return
        reasons = []
        for i in to_reset:
            if fell[i]:
                reasons.append("fall")
            elif oob[i]:
                reasons.append("out_of_bounds")
            else:
                reasons.append("stuck_tipped")
        self._reset_robot_to_spawn(envs_idx=to_reset.tolist())
        for env_idx, reason in zip(to_reset.tolist(), reasons):
            spawn_xyz = tuple(float(v) for v in self._spawn_xyz_per_env[env_idx])
            spawn_wxyz = tuple(float(v) for v in self._spawn_quat_wxyz_per_env[env_idx])
            event = self.episode_states[env_idx].reset(
                reason=reason,
                success=True,
                spawn_pose_xyz=spawn_xyz,
                spawn_pose_quat_xyzw=self._wxyz_to_xyzw(spawn_wxyz),
                stamp_ns=self._sim_time_ns,
            )
            self._last_executed[env_idx] = 0.0
            self._blocks_in_episode[env_idx] = 0
            self._consecutive_tipped_blocks[env_idx] = 0
            self._recovery_interlock_blocks_remaining[env_idx] = 0
            # New episode for this env: redraw the collector so the §13 mix
            # is preserved across resets, and let it clear its per-env state.
            self._scheduler.on_episode_reset(int(env_idx))
            writer.write_env(
                env_idx, "reset_event", messages.reset_event(event), self._sim_time_ns
            )

    def _check_and_reset_completed_envs(self, writer: Any) -> None:
        """Respawn (same collector) envs whose route teacher claimed every beacon.

        Without this an env that finishes its route early spends the rest of
        the block budget emitting ``hold`` (~half the budget on a generous
        tier) — a stationary robot that wastes compute and floods the dataset
        with near-identical idle frames. We respawn from a fresh corridor
        spawn so the env keeps producing data.

        Crucially the env is **kept on its current collector** rather than
        redrawn from the scheduler. Redrawing handed completed route episodes
        to other collectors, so route_teacher accrued far fewer blocks than
        its §13 assignment weight (pilot: 18% of blocks vs 30% target).
        Keeping the collector means a route env runs back-to-back fresh route
        episodes, so its transition share matches its assignment weight while
        still getting fresh-spawn diversity and zero idle.

        Revisit collectors (loop_revisit) are skipped (re-targeting cleared
        beacons is their signal); non-route collectors never report
        all-claimed, so they run their full budget as intended.
        """

        if self._scene_graph is None or not self._landmark_cell_to_id:
            return
        all_cells = {int(c) for c in self._landmark_cell_to_id}
        to_reset: list[int] = []
        for env_idx in range(self.n_envs):
            if int(self._blocks_in_episode[env_idx]) < _MIN_BLOCKS_BEFORE_COMPLETE_RESET:
                continue
            policy = self._scheduler.policy_for(env_idx)
            if not _is_route_like_policy(policy):
                continue
            if bool(getattr(policy, "revisit_after_arrival", False)):
                continue
            visited = getattr(policy, "visited_landmark_cells", None)
            if not callable(visited):
                continue
            claimed = {int(c) for c in visited(env_idx)}
            if all_cells.issubset(claimed):
                to_reset.append(env_idx)
        if not to_reset:
            return
        self._reset_robot_to_spawn(envs_idx=to_reset)
        for env_idx in to_reset:
            self._per_env_route_completions[env_idx] += 1
            spawn_xyz = tuple(float(v) for v in self._spawn_xyz_per_env[env_idx])
            spawn_wxyz = tuple(float(v) for v in self._spawn_quat_wxyz_per_env[env_idx])
            event = self.episode_states[env_idx].reset(
                reason="route_complete",
                success=True,
                spawn_pose_xyz=spawn_xyz,
                spawn_pose_quat_xyzw=self._wxyz_to_xyzw(spawn_wxyz),
                stamp_ns=self._sim_time_ns,
            )
            self._last_executed[env_idx] = 0.0
            self._blocks_in_episode[env_idx] = 0
            self._consecutive_tipped_blocks[env_idx] = 0
            self._recovery_interlock_blocks_remaining[env_idx] = 0
            # Keep the same collector (don't redraw) and just clear its per-env
            # state for the fresh episode — see the docstring.
            self._scheduler.policy_for(int(env_idx)).on_episode_reset(int(env_idx))
            writer.write_env(
                env_idx, "reset_event", messages.reset_event(event), self._sim_time_ns
            )

    def _spawn_yaw_planning_grid(self) -> Any | None:
        """Return a cached planning grid for spawn-yaw alignment.

        Built once per scene from whichever route-aware collector already
        has one, or from the scene manifest if no collector publishes one.
        Returning ``None`` is safe — the spawn helper simply skips yaw
        alignment.
        """

        if self._spawn_planning_grid is not None:
            return self._spawn_planning_grid
        for collector in self._collectors.values():
            grid_for = getattr(collector, "spawn_planning_grid", None)
            if callable(grid_for):
                grid = grid_for(self.pack)
                if grid is not None:
                    self._spawn_planning_grid = grid
                    return grid
        manifest = getattr(self._scene_graph, "manifest", None)
        if manifest is None:
            return None
        from go2nav.corpus.grid import InflatedOccupancyGrid

        self._spawn_planning_grid = InflatedOccupancyGrid(
            manifest, cell_size_m=0.05, inflation_m=0.20
        )
        return self._spawn_planning_grid

    def _reset_robot_to_spawn(self, envs_idx: list[int] | None) -> None:
        robot = self.build.robot
        if envs_idx is None:
            envs = list(range(self.n_envs))
        else:
            envs = list(envs_idx)
        if not envs:
            return

        # Resolve a safe per-env spawn pose. The helper enforces the
        # ``spawn_clearance_floor_m`` gate for both randomized sampling and
        # the manifest (fixed-spawn) path: a manifest spawn that fails the
        # floor falls back to a sampled pose so the robot never starts
        # wedged between walls. Updating the per-env cache here means the
        # ResetEvent log, the camera-pose helper, and the next reset all
        # see the same spawn.
        restrict_to_cells = None
        for collector in self._collectors.values():
            if hasattr(collector, "spawn_restriction_cells"):
                restrict_to_cells = collector.spawn_restriction_cells(self.pack)
                if restrict_to_cells:
                    break

        planning_grid = self._spawn_yaw_planning_grid()

        for env_idx in envs:
            xyz, wxyz, _cell_id, source = _select_spawn_pose(
                scene_graph=self._scene_graph,
                manifest_xyz=tuple(self.pack.robot.spawn_xyz_m),
                manifest_quat_wxyz=tuple(self.pack.robot.spawn_quat_wxyz),
                spawn_z_m=float(self.pack.robot.spawn_xyz_m[2]),
                randomize=bool(self.config.randomize_spawn_pose),
                clearance_floor_m=float(self.config.spawn_clearance_floor_m),
                rng=self._spawn_rng,
                restrict_to_cells=restrict_to_cells,
                planning_grid=planning_grid,
            )
            if source == "safety_fallback":
                print(
                    f"[rollout] spawn safety gate: scene={self.pack.scene_id} env={env_idx} "
                    f"manifest spawn clearance below "
                    f"{self.config.spawn_clearance_floor_m:.2f} m; "
                    f"falling back to sampled cell xy=({xyz[0]:.2f}, {xyz[1]:.2f})",
                    file=sys.stderr,
                )
            self._spawn_xyz_per_env[env_idx] = xyz
            self._spawn_quat_wxyz_per_env[env_idx] = wxyz

        pos_batch = self._spawn_xyz_per_env[envs].astype(np.float32, copy=False)
        quat_batch = self._spawn_quat_wxyz_per_env[envs].astype(np.float32, copy=False)
        reset_stance = getattr(self.policy, "reset_stance_rad", self._stance)
        stance_batch = np.tile(np.asarray(reset_stance, dtype=np.float32), (len(envs), 1))
        robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
        robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)
        robot.set_dofs_position(stance_batch, self._leg_dof_idx.tolist(), envs_idx=envs)
        robot.set_dofs_velocity(
            np.zeros_like(stance_batch), self._leg_dof_idx.tolist(), envs_idx=envs
        )
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(envs_idx)

    # ------------------------------------------------------------------
    # Camera + contacts
    # ------------------------------------------------------------------



    def _extract_foot_contacts(self) -> np.ndarray:
        """Return per-env ``(n_envs, 4)`` bool array in LeWM fl/fr/rl/rr order.

        Uses ``get_links_net_contact_force`` against the calf links as a
        contact proxy. The Genesis-shipped Go2 URDF has no ``_foot`` links;
        the foot is the terminal feature of the calf. If a future URDF
        exposes explicit foot links, switch the link-name lookup.
        """

        if self.config.foot_contact_source == "zero":
            return np.zeros((self.n_envs, 4), dtype=bool)

        robot = self.build.robot
        target_names = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
        link_indices = []
        for name in target_names:
            for link in robot.links:
                if link.name == name:
                    link_indices.append(link.idx)
                    break
        if len(link_indices) != 4:
            return np.zeros((self.n_envs, 4), dtype=bool)

        try:
            forces = self._as_np(robot.get_links_net_contact_force(link_indices))
        except Exception:
            return np.zeros((self.n_envs, 4), dtype=bool)
        # forces shape varies across Genesis backends: (n_envs, 4, 3),
        # (4, n_envs, 3), or (4, 3) for a single env.
        if forces.ndim == 2:
            forces = forces[None, ...]
        elif forces.ndim == 3 and forces.shape[0] == 4 and forces.shape[1] == self.n_envs:
            forces = np.transpose(forces, (1, 0, 2))
        if forces.ndim != 3 or forces.shape[0] != self.n_envs or forces.shape[1] != 4:
            return np.zeros((self.n_envs, 4), dtype=bool)
        magnitudes = np.linalg.norm(forces, axis=-1)  # (n_envs, 4)
        return magnitudes > 1e-3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_np(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return np.asarray(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: Iterable[float]) -> tuple[float, float, float, float]:
        w, x, y, z = (float(c) for c in q_wxyz)
        return (x, y, z, w)

    @staticmethod
    def _world_to_body(v_world: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        # Inverse of rotate_body_to_world: use conjugate quaternion.
        return _rotate_world_to_body(v_world[None, :], q_xyzw[None, :])[0]

    def _next_sequence_id(self) -> int:
        sid = self._sequence_id_counter
        self._sequence_id_counter += 1
        return sid


def _yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    """Return body yaw (rad) from a wxyz quaternion (ZYX convention)."""

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _roll_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _pitch_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        return float(math.copysign(math.pi / 2.0, sinp))
    return float(math.asin(sinp))


def _rotate_world_to_body(v_world: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the body frame using xyzw quaternions."""

    q_inv = np.asarray(q_xyzw, dtype=np.float64).copy()
    q_inv[..., :3] *= -1.0
    return rotate_body_to_world(np.asarray(v_world, dtype=np.float64), q_inv).astype(np.float32)


def _resolve_manifest_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_rollout_leg_dof_indices(robot: Any, configured_indices: Iterable[int]) -> np.ndarray:
    configured = tuple(int(idx) for idx in configured_indices)
    if configured != DEFAULT_GO2_LEG_DOF_INDICES:
        return np.array(configured, dtype=np.int64)
    joints = getattr(robot, "joints", None)
    if joints is None:
        return np.array(configured, dtype=np.int64)
    joint_by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
    if not all(name in joint_by_name for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER):
        return np.array(configured, dtype=np.int64)
    return np.array(
        [
            _single_dof_index(joint_by_name[name])
            for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER
        ],
        dtype=np.int64,
    )


def _entity_local_dof_index(robot: Any, joint: Any) -> int:
    return _single_dof_index(joint) - int(getattr(robot, "dof_start", 0))


def _single_dof_index(joint: Any) -> int:
    dofs_idx = getattr(joint, "dofs_idx", None)
    if dofs_idx is None:
        dofs_idx = getattr(joint, "dof_idx", None)
    if isinstance(dofs_idx, (list, tuple)):
        if len(dofs_idx) != 1:
            raise ValueError(
                f"joint {getattr(joint, 'name', '<unnamed>')} has {len(dofs_idx)} DOFs; expected 1"
            )
        return int(dofs_idx[0])
    if dofs_idx is None:
        raise ValueError(f"joint {getattr(joint, 'name', '<unnamed>')} has no DOF index")
    return int(dofs_idx)




