"""P33 (docs/dynamic/PLAN.md P33): where the hard toy's objects are, by an engineered tracker or learned keypoints, and
their chains through a window.

    python -m go2nav.data.hard_tracks blobs --data ROOT [--workers N]
    python -m go2nav.data.hard_tracks keypoints --data ROOT --locator LOCATOR.pt

**The agent** is found by its colour (`models.object_tokens.detect`): the robot knows itself, as the ring world's robot
token is given.

**Obstacles** (one colour, 1-3 of them, bars over them), up to `SLOTS` a frame, NaN past the last:
- `blobs`, engineered (the toy's ring tracker): pixels near the obstacles' colour, grouped into 8-connected regions; a
  region of at least `MIN_MASS` is one detection at its weighted centre. Discs that touch merge; a bar can split one.
- `keypoints`, learned: P30's locator (`models.keypoints`, K = `SLOTS`) trained on the hard frames' pairs, every
  keypoint a detection (the agent's included; the measures ignore whatever sits on it).

**Joins** (`joins`): each detection to the nearest detection of the next (or previous) frame within `GATE_PX`, as the
ring tracker joins. **Chains** (`chain`): from a window's last history frame t0, each detection followed back and on;
where a chain breaks it is not seen (in the history it stays where it was next seen), and a slot empty at t0 is not
present. Files: ROOT/<split>/tracks_<estimator>.npz with `xy` (E, T, SLOTS, 2) and `agent` (E, T, 2).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from go2nav.sim import reactive_balls as world

SLOTS, GATE_PX, MIN_MASS, REACH = 6, 5., 3., .25


def blob_frames(frames):
    """uint8 frames (N, 64, 64, 3) to detections (N, SLOTS, 2) px, NaN past each frame's count."""
    from scipy import ndimage
    rgb = frames.astype(np.float32)/255.
    weight = np.clip(1-np.linalg.norm(rgb-world.SAME/255., axis=-1)/REACH, 0, 1)            # (N, 64, 64)
    centres = np.arange(world.SIZE)+.5
    out = np.full((len(frames), SLOTS, 2), np.nan, np.float32)
    for n in range(len(frames)):
        labels, count = ndimage.label(weight[n] > .05, structure=np.ones((3, 3)))
        if not count:
            continue
        index = np.arange(1, count+1)
        mass = ndimage.sum(weight[n], labels, index)
        ys = ndimage.sum(weight[n]*centres[:, None], labels, index)/np.maximum(mass, 1e-9)
        xs = ndimage.sum(weight[n]*centres[None, :], labels, index)/np.maximum(mass, 1e-9)
        keep = np.argsort(-mass)[:SLOTS]
        keep = keep[mass[keep] >= MIN_MASS]
        out[n, :len(keep)] = np.stack((xs[keep], ys[keep]), -1)
    return out


def blob_job(args):
    path, start, stop = args
    frames = np.load(path, mmap_mode='r')
    return start, blob_frames(np.asarray(frames[start:stop]).reshape(-1, world.SIZE, world.SIZE, 3)).reshape(stop-start, -1, SLOTS, 2)


@torch.no_grad()
def agents(frames, device, batch=4096):
    """The agent's position by its colour (E, T, 2)."""
    from go2nav.models.object_tokens import detect
    from go2nav.training.balls import to_pixels
    flat = frames.reshape(-1, world.SIZE, world.SIZE, 3)
    out = [detect(to_pixels(torch.from_numpy(np.asarray(flat[i:i+batch])).to(device)))[0][:, 0].cpu().numpy()
           for i in range(0, len(flat), batch)]
    return np.concatenate(out).reshape(*frames.shape[:2], 2)


def joins(xy, step, gate=GATE_PX):
    """Each detection of frame t to the nearest of frame t+step within `gate`: its slot there, or -1 (E, T, S)."""
    e, t, s, _ = xy.shape
    out = np.full((e, t, s), -1, np.int64)
    lo, hi = (0, t-step) if step > 0 else (-step, t)
    a, b = xy[:, lo:hi], xy[:, lo+step:hi+step]
    d = np.linalg.norm(a[:, :, :, None]-b[:, :, None], axis=-1)                              # (E, T', S, S)
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.argmin(-1)
    out[:, lo:hi] = np.where(np.take_along_axis(d, best[..., None], -1)[..., 0] < gate, best, -1)
    return out


def chain(xy, ahead, back, e, t0, before=2, after=8):
    """Windows ending their history at frame t0 of episode e (B,): positions (B, before+1+after, S, 2), seen (same
    without the 2) and present (B, S); see the module's description."""
    b, s = len(e), xy.shape[2]
    length, h = before+1+after, before
    index = np.full((b, length, s), -1, np.int64)
    present = np.isfinite(xy[e, t0][..., 0])
    index[:, h] = np.where(present, np.arange(s), -1)
    rows = np.arange(b)[:, None]
    for k in range(h+1, length):
        prev = index[:, k-1]
        index[:, k] = np.where(prev >= 0, ahead[e[:, None], t0[:, None]+k-1-h, np.maximum(prev, 0)], -1)
    for k in range(h-1, -1, -1):
        nxt = index[:, k+1]
        index[:, k] = np.where(nxt >= 0, back[e[:, None], t0[:, None]+k+1-h, np.maximum(nxt, 0)], -1)
    frames = t0[:, None]+np.arange(-h, after+1)
    pos = np.take_along_axis(xy[e[:, None], frames], np.maximum(index, 0)[..., None], 2)
    seen = index >= 0
    pos = np.where(seen[..., None], pos, np.nan)
    for k in range(h-1, -1, -1):
        pos[:, k] = np.where(seen[:, k, :, None], pos[:, k], pos[:, k+1])
    return pos, seen, present


def save_blobs(root, workers=16):
    for split in ('train', 'val'):
        folder = Path(root)/'interactive'/split
        path = folder/'frames_t2.npy'
        frames = np.load(path, mmap_mode='r')
        e = len(frames)
        xy = np.full((e, frames.shape[1], SLOTS, 2), np.nan, np.float32)
        jobs = [(path, i, min(i+50, e)) for i in range(0, e, 50)]
        with ProcessPoolExecutor(workers) as pool:
            for start, part in pool.map(blob_job, jobs):
                xy[start:start+len(part)] = part
        agent = agents(frames, torch.device('cuda'))
        np.savez(folder/'tracks_blobs.npz', xy=xy, agent=agent)
        print(split, 'blobs', flush=True)


@torch.no_grad()
def save_keypoints(root, locator, batch=1024, name='keypoints'):
    from go2nav.models.keypoints import KeypointConfig, KeypointNet
    from go2nav.training.balls import to_pixels
    device = torch.device('cuda')
    state = torch.load(locator, map_location='cpu', weights_only=False)
    net = KeypointNet(KeypointConfig(**state['config'])).to(device)
    net.load_state_dict(state['keypoints'])
    net.eval()
    for split in ('train', 'val'):
        folder = Path(root)/'interactive'/split
        frames = np.load(folder/'frames_t2.npy', mmap_mode='r')
        flat = frames.reshape(-1, world.SIZE, world.SIZE, 3)
        xy = np.concatenate([net(to_pixels(torch.from_numpy(np.asarray(flat[i:i+batch])).to(device))).cpu().numpy()
                             for i in range(0, len(flat), batch)]).reshape(*frames.shape[:2], -1, 2)
        np.savez(folder/f'tracks_{name}.npz', xy=xy.astype(np.float32), agent=agents(frames, device))
        print(split, name, flush=True)


def load(root, split, estimator):
    with np.load(Path(root)/'interactive'/split/f'tracks_{estimator}.npz') as z:
        xy, agent = z['xy'], z['agent']
    return dict(xy=xy, agent=agent, ahead=joins(xy, 1), back=joins(xy, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('blobs')
    b.add_argument('--data', required=True)
    b.add_argument('--workers', type=int, default=16)
    k = sub.add_parser('keypoints')
    k.add_argument('--data', required=True)
    k.add_argument('--locator', required=True)
    k.add_argument('--name', default='keypoints', help='the tracks file: tracks_<name>.npz (P35: joint)')
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.2)
    if args.command == 'blobs':
        save_blobs(args.data, args.workers)
    else:
        save_keypoints(args.data, args.locator, name=args.name)


if __name__ == '__main__':
    main()
