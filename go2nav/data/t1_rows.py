"""T1 (temporal predictor) training rows from a bulk-rollout dataset: the shipped row selection, with every row's
actions and command history recomputed from the rollouts.

    python -m go2nav.data.t1_rows DATAGEN_ROOT --out ROWS.jsonl [--selection data/training/t1_temporal/rows.jsonl]
                                  [--render-root RENDER_ROOT] [--workers 12]

DATAGEN_ROOT is a `go2nav.corpus.datagen` output (or the May one, whose plan folders carry an index prefix):
rollout/<split>/<family>/chunk_*/raw/<scene>/messages.jsonl and plan/<scene>/frames.jsonl.

**Selection.** Which rows exist (scene, environment and start time, split, family, episode cluster and the five
frames) is fixed data, like the corpus plan: the original programme chose them through its earlier paired-navigation
and supervision datasets, whose scene roles are not reproduced here. Rows are written in selection order, with the
selection's ids, splits and frames.

**Per row**, with s the 100-ms step of the ctx2 frame (frame index // 48 + 1; a render step holds the 48
environments' frames):
- a1, a2: the applied command trajectories (5 ticks x (forward, yaw)) of the command blocks starting at s and s + 5.
  They are reconstructed from the requested commands by the rate limiter (0.25 m/s and 0.35 rad/s per 100-ms tick,
  restarting from stand at a respawn) and must match the logged executed blocks tick by tick; a2 repeats a1 when the
  second block leaves the episode.
- control: the applied command (forward, yaw) at the step before each of steps s-14 .. s (stand before the first).
With --render-root, rows whose frames are not rendered there are reported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
SELECTION = REPO/'data/training/t1_temporal/rows.jsonl'
ENVS_PER_STEP = 48
TICKS = 5
HISTORY = 15
RATES = (0.25, 0.25, 0.35)  # per tick: forward, lateral, yaw
STAND = (0., 0., 0.)
FORWARD_YAW = (0, 2)
FRAME = re.compile(r'(?P<scene>[^/]+)/rgb/frame_(?P<frame>\d+)_env_(?P<env>\d+)\.png$')


def slew(requested, previous):
    """The applied trajectory of one block: each tick moves the applied command toward the request by at most
    RATES; returns the trajectory and the final applied command."""
    applied, trajectory = list(previous), []
    for tick in requested:
        for channel, rate in enumerate(RATES):
            applied[channel] += max(-rate, min(rate, tick[channel]-applied[channel]))
        trajectory.append(list(applied))
    return trajectory, tuple(applied)


def scene_files(root: Path, scene: str) -> tuple[Path, Path]:
    for plan in root.glob('rollout/*/*/chunk_*/plan/*'):
        if plan.name == scene or plan.name.endswith('_'+scene):
            messages = plan.parent.parent/'raw'/scene/'messages.jsonl'
            if messages.is_file() and (plan/'frames.jsonl').is_file():
                return plan/'frames.jsonl', messages
    raise FileNotFoundError(f'no frames.jsonl and messages.jsonl for {scene} under {root}')


def _logged_blocks(messages: Path) -> dict:
    """(env, block) -> the logged executed trajectory, blocks numbered in time order per environment."""
    per_env = defaultdict(list)
    with messages.open() as stream:
        for line in stream:
            if '"/lewm/go2/executed_command_block"' not in line:
                continue
            record = json.loads(line)
            if record['canonical_topic'] == '/lewm/go2/executed_command_block':
                per_env[record['env_index']].append((record['timestamp_ns'], record['payload']))
    logged = {}
    for env, entries in per_env.items():
        for block, (_, p) in enumerate(sorted(entries, key=lambda entry: entry[0])):
            logged[env, block] = [[p['executed_vx_body_mps'][t], p['executed_vy_body_mps'][t],
                                   p['executed_yaw_rate_radps'][t]] for t in range(TICKS)]
    return logged


def _steps(frames: Path) -> dict:
    """(env, step) -> (episode identity, requested command at that tick)."""
    table = {}
    with frames.open() as stream:
        for line in stream:
            record = json.loads(line)
            env, step = record['env_index'], record['frame_index']//ENVS_PER_STEP+1
            command, tick = record['command_context'], (step-1) % TICKS
            table[env, step] = ((env, record['episode']['episode_id'], record['episode']['reset_count']),
                                [command['vx_body_mps'][tick], command['vy_body_mps'][tick],
                                 command['yaw_rate_radps'][tick]])
    return table


def _applied(table: dict) -> tuple[dict, dict]:
    """Run the limiter through each environment's blocks in time order; returns the applied command per (env, step)
    and the trajectory per (env, block)."""
    applied, blocks = {}, {}
    for env in sorted({env for env, _ in table}):
        previous, episode, block = STAND, None, 0
        while all((env, block*TICKS+1+t) in table for t in range(TICKS)):
            steps = [block*TICKS+1+t for t in range(TICKS)]
            here = table[env, steps[0]][0]
            if episode is not None and here != episode:
                previous = STAND  # a respawn: the controller restarts from stand
            episode = here
            trajectory, previous = slew([table[env, s][1] for s in steps], previous)
            blocks[env, block] = trajectory
            applied.update({(env, s): value for s, value in zip(steps, trajectory)})
            block += 1
    return applied, blocks


def scene_rows(task) -> list[dict]:
    root, scene, selected = task
    frames_path, messages_path = scene_files(Path(root), scene)
    table = _steps(frames_path)
    applied, blocks = _applied(table)
    logged = _logged_blocks(messages_path)
    for key, trajectory in blocks.items():
        if key in logged and any(abs(a-b) >= 1e-6 for tick, ref in zip(trajectory, logged[key])
                                 for a, b in zip(tick, ref)):
            raise ValueError(f'{scene}: env {key[0]} block {key[1]} differs from the logged executed block')
    out = []
    for row in selected:
        match = FRAME.search(row['frames']['ctx2'])
        env, step = int(match['env']), int(match['frame'])//ENVS_PER_STEP+1
        episode = table[env, step][0]
        first = (step-1)//TICKS
        actions = []
        for block in (first, first+1):
            span = [block*TICKS+1+t for t in range(TICKS)]
            if (env, block) not in blocks or any(table[env, s][0] != episode for s in span if (env, s) in table):
                break
            if (env, block) not in logged:
                raise ValueError(f'{scene}: env {env} block {block} was not logged')
            actions.append([blocks[env, block][t][c] for t in range(TICKS) for c in FORWARD_YAW])
        if not actions:
            raise ValueError(f'{row["id"]}: no command block starts at its ctx2 frame')
        history = range(step-HISTORY+1, step+1)
        control = [[applied.get((env, s-1), STAND)[c] for c in FORWARD_YAW] for s in history]
        out.append(dict(row, a1=actions[0], a2=actions[-1], control=control))
    return out


def derive(root, *, selection=SELECTION, render_root=None, workers=12) -> tuple[list[dict], list[str]]:
    selection = [json.loads(line) for line in Path(selection).read_text().splitlines() if line.strip()]
    by_scene = defaultdict(list)
    for row in selection:
        by_scene[FRAME.search(row['frames']['ctx2'])['scene']].append(row)
    with Pool(workers) as pool:
        derived = {row['id']: row for rows in pool.imap_unordered(
            scene_rows, [(str(root), scene, rows) for scene, rows in sorted(by_scene.items())]) for row in rows}
    rows = [{key: derived[row['id']][key] for key in ('id', 'split', 'family', 'episode_cluster', 'frames', 'a1',
                                                       'a2', 'control')} for row in selection]
    missing = []
    if render_root is not None:
        missing = sorted({path for row in rows for path in row['frames'].values()
                          if not (Path(render_root)/path).is_file()})
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selection', type=Path, default=SELECTION)
    p.add_argument('--render-root', type=Path)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    rows, missing = derive(a.root, selection=a.selection, render_root=a.render_root, workers=a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print(json.dumps(dict(rows=len(rows), train=sum(row['split'] == 'train' for row in rows),
                          frames_not_rendered=len(missing), first_missing=missing[:3])))


if __name__ == '__main__':
    main()
