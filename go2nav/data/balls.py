"""P25's data (docs/dynamic/PLAN.md P25 item 1): per n (1, 3 and 6 white balls), 1,000 training and 100 validation
episodes of 100 frames of `sim.balls` under its random policy.

    python -m go2nav.data.balls build --out OUT [--n 1 3 6] [--workers N]

OUT/n<n>/<split>/ holds frames.npy (E, 100, 64, 64, 3) uint8, memory-mappable; labels.npz (the actions (E, 99, 2) as
executed, red (E, 100, 2), white (E, 100, n, 2), velocity (E, 100, n, 2)); and meta.json. Each chunk of episodes draws
from its own seed (SEED, n, split, chunk), so a build is a pure function of them. The labels are for scoring and for
the supervised reference (L1) only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from go2nav.sim import balls

SEED, FRAMES, CHUNK = 2026111501, 100, 50
SPLITS = dict(train=1000, val=100)
SPLIT_INDEX = dict(train=0, val=1)


def chunk(args):
    n, split, index, count = args
    rng = np.random.default_rng((SEED, n, SPLIT_INDEX[split], index))
    return index, balls.episodes(rng, count, n, FRAMES)


def build(out, ns=(1, 3, 6), workers=16):
    out = Path(out)
    for n in ns:
        for split, episodes in SPLITS.items():
            folder = out/f'n{n}'/split
            folder.mkdir(parents=True, exist_ok=True)
            frames = np.lib.format.open_memmap(folder/'frames.npy.partial', mode='w+', dtype=np.uint8,
                                               shape=(episodes, FRAMES, balls.SIZE, balls.SIZE, 3))
            labels = dict(actions=np.zeros((episodes, FRAMES-1, 2), np.float32),
                          red=np.zeros((episodes, FRAMES, 2), np.float32),
                          white=np.zeros((episodes, FRAMES, n, 2), np.float32),
                          velocity=np.zeros((episodes, FRAMES, n, 2), np.float32))
            jobs = [(n, split, i, min(CHUNK, episodes-i*CHUNK)) for i in range((episodes+CHUNK-1)//CHUNK)]
            with ProcessPoolExecutor(workers) as pool:
                for index, part in pool.map(chunk, jobs):
                    rows = slice(index*CHUNK, index*CHUNK+len(part['frames']))
                    frames[rows] = part['frames']
                    for key in labels:
                        labels[key][rows] = part[key]
            frames.flush()
            del frames
            (folder/'frames.npy.partial').rename(folder/'frames.npy')
            np.savez(folder/'labels.npz', **labels)
            meta = dict(schema='go2nav_balls.v1', n=n, split=split, episodes=episodes, frames=FRAMES, size=balls.SIZE,
                        radius=balls.RADIUS, max_action=balls.MAX_ACTION, speed=list(balls.SPEED),
                        hold=list(balls.HOLD), seed=SEED)
            (folder/'meta.json').write_text(json.dumps(meta, indent=1))
            print(f'n{n} {split}: {episodes} episodes', flush=True)


def load(root, n, split, mmap=True):
    """(frames (E, T, 64, 64, 3) uint8, labels dict) of a built split."""
    folder = Path(root)/f'n{n}'/split
    frames = np.load(folder/'frames.npy', mmap_mode='r' if mmap else None)
    return frames, dict(np.load(folder/'labels.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--n', type=int, nargs='+', default=[1, 3, 6])
    b.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    build(args.out, tuple(args.n), args.workers)


if __name__ == '__main__':
    main()
