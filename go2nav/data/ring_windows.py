"""P31 stage 2 (docs/dynamic/PLAN.md P31 items 3 and 4): windows of the ring corpus as tokens, for two models that
differ only in where the objects come from.

    python -m go2nav.data.ring_windows locate --corpus CORPUS --locator LOCATOR.pt

**Objects:**
- `tracker`: the ring tracker's slots (`data.object_tracks`): a slot is present at a step if its track is the one it
  holds at the window's last history step t0, seen if detected then; its velocity is the tracker's.
- `learned`: P31's locator on every frame's scan drawn in the robot's own pose (`locate`, stored once a locator as
  world positions). Each keypoint is joined to the nearest keypoint 0.4 s earlier and 0.4 s later within the tracker's
  0.4-m gate (PLAN P31 item 3, as corrected after round 1), and its velocity is its joined displacement over 0.3 s
  (none if unjoined). The chain of joins gives each keypoint at t0 its keypoint at every other step; where the chain
  breaks, the keypoint is not seen in the history (it stays where it was next seen, as a remembered track) and has no
  target in the future. Every keypoint at t0 is present.

**Tokens** (`Windows.batch`), in the robot's pose at t0 as `models.object_jepa`'s features: an object's walls are the
distance to the nearest scan return of its step on M5's 5 x 5 grid (`data.ring_rasters.nearest_return`), not the
layout's; the robot's on the 7 x 7 grid, from its own step's scan in the history and from t0's scan in the future
(what the robot has seen; the future is not observed). Steps are 0.4 s: 3 history steps, then `future` steps.

**Labels** (measures and the read-out head only): each token's true walker, the nearest walker in sight within
0.3 m of its position at that step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data import ring_rasters as rr
from go2nav.models.object_jepa import (CLIP_M, OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING, STEP_FRAMES,
                                       STEP_S, actions_input, object_features, offsets, robot_features)

SLOTS, HISTORY, VELOCITY_FRAMES, MATCH_M, GATE_M = 6, 3, 3, .3, .4


def nearest_join(a, b, gate=GATE_M, chunk=100000):
    """Keypoints a, b (N, K, 2): each keypoint of a to the nearest keypoint of b within `gate` (N, K), -1 if none."""
    out = []
    for i in range(0, len(a), chunk):
        d = torch.cdist(a[i:i+chunk].float(), b[i:i+chunk].float())
        best, which = d.min(-1)
        out.append(torch.where(best < gate, which, torch.full_like(which, -1)))
    return torch.cat(out)


@torch.no_grad()
def locate(corpus_root, locator_path, device, batch=1024):
    """Every frame's keypoints in the world (F, K, 2), each scan drawn in the robot's own pose, for both splits;
    saved beside the locator as keypoints_<split>.npy."""
    from go2nav.models.keypoints import KeypointConfig, KeypointNet
    state = torch.load(locator_path, map_location='cpu', weights_only=False)
    net = KeypointNet(KeypointConfig(**state['config'])).to(device)
    net.load_state_dict(state['keypoints'])
    net.eval()
    for split in ('train', 'val'):
        corpus = rr.Corpus(corpus_root, split, device)
        out = []
        for i in range(0, len(corpus.pose), batch):
            pose = corpus.pose[i:i+batch]
            px = net(rr.rasterise(corpus.ranges[i:i+batch], pose, pose)*2-1)
            out.append(rr.to_world(rr.to_local(px), pose).cpu().numpy())
        np.save(Path(locator_path).parent/f'keypoints_{split}.npy', np.concatenate(out))
        print(split, len(corpus.pose), 'frames located', flush=True)
        del corpus
        torch.cuda.empty_cache()


class Frames:
    """A split's per-frame arrays on the device, for both kinds of objects."""

    def __init__(self, corpus_root, split, device, keypoints=None, labels=True):
        self.corpus = rr.Corpus(corpus_root, split, device, labels=labels)
        starts = self.corpus.starts
        self.episode = np.repeat(np.arange(len(starts)-1), np.diff(starts))
        xy, seen, ids, velocity = [], [], [], []
        for k in range(len(starts)-1):
            with np.load(Path(corpus_root)/'objects'/split/f'{k:05d}.npz') as z:
                xy.append(z['xy']), seen.append(z['seen']), ids.append(z['id']), velocity.append(z['velocity'])
        t = lambda parts: torch.as_tensor(np.concatenate(parts), device=device)
        self.tracker = dict(xy=t(xy), seen=t(seen), id=t(ids), velocity=t(velocity))
        if labels:
            self.walkers = torch.as_tensor(self.corpus.walkers_xy, device=device)
            self.sight = torch.as_tensor(self.corpus.in_sight, device=device)
        self.device = device
        if keypoints is not None:
            kp = torch.as_tensor(np.load(keypoints), device=device)
            n, self.keypoints = len(kp), kp.shape[1]
            same = lambda shift: torch.as_tensor(np.r_[self.episode[shift:] == self.episode[:n-shift],
                                                       np.zeros(shift, bool)], device=device)
            frame = torch.arange(n, device=device)

            def joins(shift):                       # frame f's keypoints -> frame f+shift's (shift may be negative)
                out = torch.full((n, self.keypoints), -1, dtype=torch.long, device=device)
                ok = same(abs(shift))
                src = frame[ok] if shift > 0 else frame[ok]+abs(shift)
                out[src] = nearest_join(kp[src], kp[src+shift])
                return out
            self.ahead4, self.back4, back3 = joins(STEP_FRAMES), joins(-STEP_FRAMES), joins(-VELOCITY_FRAMES)
            prior = torch.gather(kp[(frame-VELOCITY_FRAMES).clamp(min=0)], 1, back3.clamp(min=0)[..., None].expand(-1, -1, 2))
            velocity = torch.where((back3 >= 0)[..., None], (kp-prior)/(VELOCITY_FRAMES*.1), torch.zeros_like(kp))
            self.learned = dict(xy=kp, velocity=velocity)

    def windows(self, future):
        """Every frame that can be a window's t0: 3 history steps (and 0.3 s before the first) and `future` steps
        inside its episode."""
        s = self.corpus.starts
        return np.concatenate([np.arange(a+STEP_FRAMES*(HISTORY-1)+VELOCITY_FRAMES, b-STEP_FRAMES*future)
                               for a, b in zip(s[:-1], s[1:]) if b-a > STEP_FRAMES*(HISTORY-1+future)+VELOCITY_FRAMES])

    def scan(self, frames):
        return rr.returns(self.corpus.ranges[frames], self.corpus.pose[frames])

    def objects(self, kind, t0, future):
        """Each object's world position, velocity, seen and present at every step (B, T, S, ·)."""
        steps = t0[:, None]+STEP_FRAMES*torch.arange(-(HISTORY-1), future+1, device=self.device)   # (B, T)
        if kind == 'tracker':
            o = self.tracker
            same = (o['id'][steps] == o['id'][t0][:, None]) & (o['id'][t0][:, None] >= 0)
            return dict(xy=o['xy'][steps], velocity=o['velocity'][steps].float(), seen=o['seen'][steps] & same,
                        present=same, steps=steps)
        o, h, k_all = self.learned, HISTORY-1, self.keypoints
        index = torch.full((*steps.shape, k_all), -1, dtype=torch.long, device=self.device)
        index[:, h] = torch.arange(k_all, device=self.device)
        follow = lambda table, step, prev: torch.where(prev >= 0, torch.gather(table[step], 1, prev.clamp(min=0)), prev)
        for k in range(h+1, steps.shape[1]):                      # forward: t0's keypoint -> its keypoint at step k
            index[:, k] = follow(self.ahead4, steps[:, k-1], index[:, k-1])
        for k in range(h-1, -1, -1):                              # back
            index[:, k] = follow(self.back4, steps[:, k+1], index[:, k+1])
        alive = index >= 0
        take = lambda a: torch.gather(a[steps], 2, index.clamp(min=0)[..., None].expand(-1, -1, -1, 2))
        xy, velocity = take(o['xy']), take(o['velocity'])*alive[..., None]
        for k in range(h-1, -1, -1):                              # unseen in the history: where it was next seen
            xy[:, k] = torch.where(alive[:, k, :, None], xy[:, k], xy[:, k+1])
        xy = torch.where(alive[..., None] | (torch.arange(steps.shape[1], device=self.device) <= h)[None, :, None, None],
                         xy, torch.full_like(xy, float('nan')))
        present = torch.ones_like(alive)
        return dict(xy=xy, velocity=velocity, seen=alive, present=present, steps=steps)

    def batch(self, kind, t0, future, future_pose=None, future_requested=None):
        """`models.ring_tokens.token_losses`' batch for windows ending their history at frames t0 (B,). For F004's
        scenes, `future_pose` (B, future, 3) and `future_requested` (B, 4 future, 3) replace the recorded ones."""
        o = self.objects(kind, t0, 0 if future_pose is not None else future)
        steps = o['steps']
        frame = self.corpus.pose[t0]
        heading = frame[:, 2]
        local = rr.to_frame(torch.nan_to_num(o['xy']), frame)
        velocity = rr.rotate(o['velocity'], -heading[:, None, None])*o['seen'][..., None]
        b, t, s = local.shape[:3]
        scan_points, scan_valid = self.scan(steps.flatten())
        grid = rr.rotate(offsets(OBJECT_GRID, OBJECT_SPACING, self.device)[None], heading[:, None])      # (B, G, 2)
        object_points = (torch.nan_to_num(o['xy'])[..., None, :]+grid[:, None, None]).reshape(b*t, -1, 2)
        walls = rr.nearest_return(object_points, scan_points, scan_valid, CLIP_M).reshape(b, t, s, -1)
        tokens = object_features(local, o['seen'], o['present'], walls, velocity)
        pose = self.corpus.pose[steps] if future_pose is None else torch.cat((self.corpus.pose[steps], future_pose), 1)
        robot_grid = rr.rotate(offsets(ROBOT_GRID, ROBOT_SPACING, self.device)[None, None], pose[..., 2, None])
        scans = torch.cat((steps[:, :HISTORY], t0[:, None].expand(-1, pose.shape[1]-HISTORY)), 1)        # future: t0's
        sp, sv = self.scan(scans.flatten())
        robot_walls = rr.nearest_return((pose[..., None, :2]+robot_grid).reshape(-1, ROBOT_GRID**2, 2), sp, sv,
                                        CLIP_M).reshape(*pose.shape[:2], -1)
        robot = robot_features(rr.to_frame(pose[..., :2], frame), pose[..., 2]-heading[:, None], robot_walls)
        commands = self.corpus.requested[steps[:, :-1, None]+torch.arange(STEP_FRAMES, device=self.device)]
        if future_requested is not None:
            past = self.corpus.requested[t0[:, None]+torch.arange(-STEP_FRAMES*(HISTORY-1), 0, device=self.device)]
            commands = torch.cat((past, future_requested), 1).unflatten(1, (-1, STEP_FRAMES))
        out = dict(objects=tokens[:, :HISTORY], robot=robot[:, :HISTORY], actions=actions_input(commands),
                   future_robot=robot[:, HISTORY:], seen=o['seen'][:, :HISTORY], present=o['present'][:, :HISTORY],
                   frame=frame, xy=o['xy'], steps=steps)
        if future_pose is None:
            out.update(future_objects=tokens[:, HISTORY:], valid=o['seen'][:, HISTORY:])
        return out

    def truth(self, xy, steps):
        """Each object's true walker at its steps: the nearest walker in sight within MATCH_M of its position (B, T,
        S, 2; NaN where none)."""
        w, sight = self.walkers[steps], self.sight[steps]                                    # (B, T, 4, 2), (B, T, 4)
        d = torch.cdist(torch.nan_to_num(xy, nan=1e3).flatten(0, 1).float(),
                        torch.nan_to_num(w, nan=1e3).flatten(0, 1).float()).unflatten(0, xy.shape[:2])
        d = torch.where(sight[:, :, None], d, torch.full_like(d, 1e3))
        best, which = d.min(-1)
        near = torch.gather(w, 2, which[..., None].expand(-1, -1, -1, 2))
        return torch.where((best < MATCH_M)[..., None], near, torch.full_like(near, float('nan')))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    l = sub.add_parser('locate')
    l.add_argument('--corpus', required=True)
    l.add_argument('--locator', required=True)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.45)
    locate(args.corpus, args.locator, torch.device('cuda'))


if __name__ == '__main__':
    main()
