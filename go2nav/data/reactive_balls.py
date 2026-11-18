"""P27's data (docs/dynamic/PLAN.md P27 item 1): 2,000 training and 200 validation episodes of 100 frames of
`sim.reactive_balls` per policy, with exact branches (3 moments a training episode, 6 a validation one).

    python -m go2nav.data.reactive_balls build --out OUT [--policy interactive confounded] [--workers N]

OUT/<policy>/<split>/ holds frames_t2.npy and frames_t1.npy (E, 100, 64, 64, 3) uint8 (kinds hidden and shown, the same
episodes); labels.npz (the actions, kinds and per-frame states); branches.npz (t0, target, actions and states of the 4
tapes); for the interactive training split, branch_frames_t2.npy and branch_frames_t1.npy (E, M, 4, 16, 64, 64, 3);
and meta.json. Each chunk of episodes draws from its own seed (SEED, policy, split, chunk).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from go2nav.sim import reactive_balls as world

SEED, FRAMES, CHUNK = 2026111601, 100, 50
SPLITS = dict(train=2000, val=200)
MOMENTS = dict(train=3, val=6)
INDEX = dict(train=0, val=1, interactive=0, confounded=1)
FRAME_KEYS = ('frames', 'frames_t1')


def with_branch_frames(policy, split):
    return policy == 'interactive' and split == 'train'


def chunk(args):
    policy, split, index, count, *hard = args
    hard = bool(hard and hard[0])
    key = (SEED, INDEX[policy], INDEX[split], index)+((33,) if hard else ())          # P33: the hard variant's own seeds
    rng = np.random.default_rng(key)
    return index, world.episodes(rng, count, FRAMES, policy, MOMENTS[split],
                                 with_branch_frames(policy, split) and not hard, hard=hard)


def build(out, policies=('interactive', 'confounded'), workers=16, hard=False):
    out = Path(out)
    for policy in policies:
        for split, episodes in SPLITS.items():
            folder = out/policy/split
            folder.mkdir(parents=True, exist_ok=True)
            names = dict(frames='frames_t2', frames_t1='frames_t1')
            keys = ('frames',) if hard else FRAME_KEYS
            maps = {k: np.lib.format.open_memmap(folder/f'{names[k]}.npy.partial', mode='w+', dtype=np.uint8,
                                                 shape=(episodes, FRAMES, world.SIZE, world.SIZE, 3)) for k in keys}
            if with_branch_frames(policy, split) and not hard:
                shape = (episodes, MOMENTS[split], len(world.TAPES), world.BRANCH_STEPS, world.SIZE, world.SIZE, 3)
                branch_maps = {k: np.lib.format.open_memmap(folder/f'branch_{names[k]}.npy.partial', mode='w+',
                                                            dtype=np.uint8, shape=shape) for k in FRAME_KEYS}
            else:
                branch_maps = {}
            labels, branches = {}, {}
            jobs = [(policy, split, i, min(CHUNK, episodes-i*CHUNK), hard) for i in range((episodes+CHUNK-1)//CHUNK)]
            with ProcessPoolExecutor(workers) as pool:
                for index, (episode, branch) in pool.map(chunk, jobs):
                    rows = slice(index*CHUNK, index*CHUNK+len(episode['frames']))
                    for k in keys:
                        maps[k][rows] = episode.pop(k)
                        if branch_maps:
                            branch_maps[k][rows] = branch.pop(k)
                    for store, part in ((labels, episode), (branches, branch)):
                        for k, v in part.items():
                            store.setdefault(k, [None]*len(jobs))[index] = v
            for m in (*maps.values(), *branch_maps.values()):
                m.flush()
            del maps, branch_maps
            for partial in folder.glob('*.npy.partial'):
                partial.rename(partial.with_suffix(''))
            np.savez(folder/'labels.npz', **{k: np.concatenate(v) for k, v in labels.items()})
            np.savez(folder/'branches.npz', **{k: np.concatenate(v) for k, v in branches.items()})
            meta = dict(schema='go2nav_reactive_balls.v1', hard=hard, policy=policy, split=split, episodes=episodes, frames=FRAMES,
                        moments=MOMENTS[split], seed=SEED, react=world.REACT, flee=world.FLEE, speed=list(world.SPEED),
                        delay=list(world.DELAY), tapes=list(world.TAPES), kinds=list(world.KINDS))
            (folder/'meta.json').write_text(json.dumps(meta, indent=1))
            print(f'{policy} {split}: {episodes} episodes', flush=True)


def load(root, policy, split, variant, mmap=True, branch_frames=False):
    """(frames (E, T, 64, 64, 3) uint8 of `variant` ('t1' or 't2'), labels dict, branches dict). With `branch_frames`,
    branches['frames'] holds the tapes' frames (the interactive training split only)."""
    folder = Path(root)/policy/split
    mode = 'r' if mmap else None
    frames = np.load(folder/f'frames_{variant}.npy', mmap_mode=mode)
    branches = dict(np.load(folder/'branches.npz'))
    if branch_frames:
        branches['frames'] = np.load(folder/f'branch_frames_{variant}.npy', mmap_mode=mode)
    return frames, dict(np.load(folder/'labels.npz')), branches


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--policy', nargs='+', default=['interactive', 'confounded'])
    b.add_argument('--workers', type=int, default=16)
    b.add_argument('--hard', action='store_true', help="P33's hard variant: one colour, 1-3 obstacles, bars")
    args = parser.parse_args()
    build(args.out, tuple(args.policy), args.workers, args.hard)


if __name__ == '__main__':
    main()
