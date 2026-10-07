"""Bulk rollouts of a scene corpus: the Go2 driven by the data collectors in parallel environments, recorded as
messages.

    python -m go2nav.corpus.bulk CORPUS OUT --split SPLIT [--family F] [--scene-offset N] [--scene-limit M]
                                 [--seed 7] [--n-envs 48] [--n-blocks 200]

Scenes are taken in name order within (split, family) and the scene at position k of the selected range (after
--scene-offset) is rolled out with seed `--seed + k`; the May corpus ran each (split, family) in chunks of 40 scenes
with the default seed, so its scene at index i of a chunk had seed 7 + i. Physics runs on the CPU backend with the
shipped locomotion policy (the May settings: no RGB, recovery interlock disabled, randomised spawns with a 0.45-m
wall-clearance floor, 0.15-m fall height, 0.5-m out-of-bounds pad).

Output per scene: OUT/<scene_id>/messages.jsonl (`go2nav.corpus.messages`) and summary.json (scene identity and the
rollout statistics). A scene whose physics fails (the rigid solver can produce NaNs on rough terrain) is skipped and
listed in OUT/run_summary.json; its partial folder is removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from go2nav import artifacts
from go2nav.corpus.contract import PrimitiveRegistry, SafetyLimits
from go2nav.corpus.messages import MessageWriter
from go2nav.corpus.packs import find_scene_dirs, load_platform_manifest, load_scene_pack
from go2nav.corpus.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner
from go2nav.corpus.world import build_scene_from_pack

REPO = Path(__file__).resolve().parents[2]
PLATFORM = REPO/'configs/corpus/go2_platform_manifest.yaml'
PRIMITIVES = REPO/'configs/corpus/go2_primitive_registry.yaml'


def run(corpus, out, *, split, family=None, scene_offset=0, scene_limit=None, seed=7, n_envs=48, n_blocks=200):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    platform = load_platform_manifest(PLATFORM)
    registry = PrimitiveRegistry.from_yaml(PRIMITIVES)
    safety = SafetyLimits.from_manifest(platform)
    scenes = sorted(find_scene_dirs(corpus, split=split, family=family), key=lambda p: p.name)[scene_offset:]
    if scene_limit is not None:
        scenes = scenes[:scene_limit]
    if not scenes:
        raise SystemExit(f'no scenes under {corpus} for split={split!r} family={family!r}')
    policy = GenesisGo2PPOPolicy(artifacts.path('go2_locomotion_policy'),
                                 cfg_path=artifacts.path('go2_locomotion_policy_cfgs'), device='cpu')
    summary = dict(corpus=str(corpus), split=split, family=family, n_envs=n_envs, n_blocks=n_blocks, scenes=[],
                   skipped=[])
    for index, scene_dir in enumerate(scenes):
        pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO)
        print(f'[{index+1}/{len(scenes)}] scene={pack.scene_id} family={pack.family} split={pack.split}', flush=True)
        build = build_scene_from_pack(pack, n_envs=n_envs, backend='cpu', show_viewer=False, render_robot=False)
        config = RolloutConfig(n_blocks=n_blocks, fall_z_threshold_m=.15, out_of_bounds_pad_m=.5,
                               rgb_capture_per_block=False, seed=seed+index, log_progress_every_blocks=0,
                               foot_contact_source='genesis', collector_mix=None, randomize_spawn_pose=True,
                               recovery_interlock_clearance_m=0., recovery_interlock_blocks=16,
                               spawn_clearance_floor_m=.45)
        scene_out = out/pack.scene_id
        scene_out.mkdir()
        writer = MessageWriter(scene_out/'messages.jsonl')
        started = time.time()
        try:
            stats = RolloutRunner(build, policy, registry, safety, config=config).run(writer)
        except Exception as error:  # a scene the physics cannot run (e.g. rigid-solver NaN) is skipped
            writer.close()
            shutil.rmtree(scene_out, ignore_errors=True)
            print(f'[SKIP] scene={pack.scene_id} {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            summary['skipped'].append(dict(scene=pack.scene_id, error=f'{type(error).__name__}: {error}'))
            continue
        writer.close()
        record = dict(scene_id=pack.scene_id, family=pack.family, split=pack.split,
                      difficulty_tier=pack.difficulty_tier, manifest_sha256=pack.manifest_sha256, seed=seed+index,
                      n_envs=n_envs, messages=sum(writer.counts.values()), per_topic_counts=writer.counts,
                      first_stamp_ns=writer.first_ns, last_stamp_ns=writer.last_ns, wall_s=time.time()-started,
                      rollout_stats=stats)
        (scene_out/'summary.json').write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
        summary['scenes'].append(pack.scene_id)
    (out/'run_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('corpus', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--split', required=True)
    p.add_argument('--family')
    p.add_argument('--scene-offset', type=int, default=0)
    p.add_argument('--scene-limit', type=int)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--n-envs', type=int, default=48)
    p.add_argument('--n-blocks', type=int, default=200)
    a = p.parse_args()
    run(a.corpus, a.out, split=a.split, family=a.family, scene_offset=a.scene_offset, scene_limit=a.scene_limit,
        seed=a.seed, n_envs=a.n_envs, n_blocks=a.n_blocks)


if __name__ == '__main__':
    main()
