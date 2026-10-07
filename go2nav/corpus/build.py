"""Build a scene corpus on disk from a plan (the May corpus: data/corpus/may_2026_plan.json).

    python -m go2nav.corpus.build OUT_DIR [--plan data/corpus/may_2026_plan.json]
    python -m go2nav.corpus.build OUT_DIR --standard --plan-seed SEED --train N --val N --test-id N --test-hard N

Layout: OUT_DIR/corpus.json (the plan and one summary row per scene) and, per scene,
OUT_DIR/<split>/<family>/<scene_id>/ with
- manifest.json: the scene manifest (`go2nav.corpus.manifest`) and its SHA-256;
- genesis_scene.json: the simulator's construction spec (objects, graph, spawn, randomization);
- topology.json: graph statistics (`go2nav.corpus.topology`).

Every scene is checked for reachability before it is written (`go2nav.corpus.validation`); the planner already
rerolls seeds that fail, so a failure here means the plan was not built with validation. The May plan also carries
two seeds rerolled by hand (salt 1) after their scenes made the rigid solver produce NaNs at rollout
(rough_local_dynamics train slots 6 and 38).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from go2nav.corpus.manifest import SceneManifest, build_scene_manifest, manifest_sha256
from go2nav.corpus.plan import CorpusPlan, SceneAssignment, plan_sha256, standard_corpus_plan
from go2nav.corpus.topology import topology_summary
from go2nav.corpus.validation import audit_scene_reachability

REPO = Path(__file__).resolve().parents[2]
MAY_PLAN = REPO/'data/corpus/may_2026_plan.json'


@dataclass(frozen=True)
class CorpusSceneResult:
    split: str
    family: str
    scene_id: str
    scene_seed: int
    relative_dir: str
    manifest_sha256: str
    node_count: int
    edge_count: int
    cycle_count: int
    obstacle_count: int
    wall_count: int
    landmark_count: int


def build_corpus(plan: CorpusPlan, out_dir: Path) -> list[CorpusSceneResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [build_scene(assignment, out_dir) for assignment in plan.assignments]
    corpus = dict(plan=plan.to_dict(), plan_sha256=plan_sha256(plan), scenes=[asdict(r) for r in results])
    write_json(out_dir/'corpus.json', corpus)
    return results


def build_scene(assignment: SceneAssignment, out_dir: Path) -> CorpusSceneResult:
    manifest = build_scene_manifest(scene_seed=assignment.scene_seed, family=assignment.family, split=assignment.split)
    if manifest.scene_id != assignment.scene_id:
        raise RuntimeError(f'scene_id drift between plan and manifest: {assignment.scene_id} != {manifest.scene_id}')
    report = audit_scene_reachability(manifest, cell_size_m=0.05, inflation_m=0.20, standoff_m=0.85,
                                      spawn_clearance_floor_m=0.20, min_navigable_standoffs_per_beacon=1)
    if not report.is_valid:
        raise RuntimeError(f'scene {assignment.scene_id} failed the reachability audit: {report.failure_reason}')
    scene_dir = out_dir/assignment.split/assignment.family/assignment.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_json(scene_dir/'manifest.json', manifest.to_dict() | dict(manifest_sha256=manifest_sha256(manifest)))
    summary = topology_summary(manifest)
    write_json(scene_dir/'topology.json', summary)
    write_json(scene_dir/'genesis_scene.json', genesis_scene(manifest))
    return CorpusSceneResult(split=assignment.split, family=assignment.family, scene_id=assignment.scene_id,
                             scene_seed=assignment.scene_seed, relative_dir=str(scene_dir.relative_to(out_dir)),
                             manifest_sha256=manifest_sha256(manifest), node_count=summary['node_count'],
                             edge_count=summary['edge_count'], cycle_count=summary['cycle_count'],
                             obstacle_count=len(manifest.obstacles), wall_count=len(manifest.walls),
                             landmark_count=len(manifest.landmarks))


def genesis_scene(manifest: SceneManifest) -> dict[str, Any]:
    """The simulator's construction spec: static objects, graph, spawn and randomization."""
    def optional(value):
        return asdict(value) if value is not None else None
    return {
        'scene_id': manifest.scene_id,
        'family': manifest.family,
        'manifest_sha256': manifest_sha256(manifest),
        'physics_seed': manifest.physics_seed,
        'visual_seed': manifest.visual_seed,
        'world_bounds_xy_m': manifest.world_bounds_xy_m,
        'spawn': {'xyz_m': manifest.spawn.xyz_m, 'quat_wxyz': manifest.spawn.quat_wxyz},
        'objects': [{'object_id': o.object_id, 'kind': o.kind, 'center_xyz_m': o.center_xyz_m,
                     'size_xyz_m': o.size_xyz_m, 'yaw_rad': o.yaw_rad, 'roll_rad': o.roll_rad,
                     'pitch_rad': o.pitch_rad, 'material_id': o.material_id} for o in manifest.static_objects],
        'graph': {'nodes': [n.__dict__ for n in manifest.graph_nodes], 'edges': [e.__dict__ for e in manifest.graph_edges]},
        'camera_constraints': manifest.camera_constraints.__dict__,
        'visual_randomization': optional(manifest.visual_randomization),
        'physics_randomization': optional(manifest.physics_randomization),
        'camera_extrinsic_jitter': optional(manifest.camera_extrinsic_jitter),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--plan', type=Path, default=MAY_PLAN)
    p.add_argument('--standard', action='store_true', help='plan a new corpus (spec family shares) instead of --plan')
    p.add_argument('--plan-seed', type=int, default=1)
    for split, default in (('train', 200), ('val', 50), ('test-id', 50), ('test-hard', 50)):
        p.add_argument(f'--{split}', type=int, default=default)
    a = p.parse_args()
    if a.standard:
        plan = standard_corpus_plan(plan_seed=a.plan_seed, train_scenes=a.train, val_scenes=a.val,
                                    test_id_scenes=a.test_id, test_hard_scenes=a.test_hard)
    else:
        plan = CorpusPlan.from_dict(json.loads(a.plan.read_text()))
    results = build_corpus(plan, a.out)
    print(json.dumps(dict(out=str(a.out), scenes=len(results), plan_sha256=plan_sha256(plan))))


if __name__ == '__main__':
    main()
