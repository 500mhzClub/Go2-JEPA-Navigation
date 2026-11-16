"""P28's data (docs/dynamic/PLAN.md P28 item 1): the F004 export's scenes on the GPU, as M5's inputs.

A scene is a branch point of P20's corpus (`data.branches`) with a walker in sight within 1.5 m: the ring tracker's 6
slots at 3 history steps of 0.4 s, and 7 futures of 16 steps (6 exact branches under random tapes, then the recorded
one), each with the robot's path, the requests and the matched walkers' true positions. ROOT/<split>/ holds one .npy
per field (world coordinates; written and checked on 16 November, `outputs/gated/F004/data`).

`Scenes.batch` gives `models.object_jepa`'s features in the robot's frame at the branch point (x forward, y left):
- `objects` (B, 3, 6, 31): each slot's position, seen, present, the walls' distance on a 5 x 5 grid round it, and the
  tracker's velocity (`object_features`); a slot is present where its track is the branch point's;
- `robot` (B, 3, 53) (`robot_features`) and `actions` (B, 18, 12): the 2 steps of requests before the point, then the
  chosen future's 16;
- for the chosen future (index 0-5 a branch, 6 the recorded one): `future_robot` (B, 16, 53); `future_objects`
  (B, 16, 6, 31), the walkers' true positions as tokens (seen and present where `valid`, the velocity the change over
  the step); `truth` (B, 16, 6, 2) and `valid` (B, 16, 6).
`world` turns positions in that frame back into world coordinates; `walls` looks up the grid round such positions.
"""
from pathlib import Path

import numpy as np
import torch

from go2nav.models.object_jepa import (OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING, STEP_FRAMES, STEP_S,
                                       actions_input, object_features, offsets, robot_features)
from go2nav.training.object_jepa import Rasters, rotate

FIELDS = ('history_xy', 'history_seen', 'history_id', 'history_velocity', 'history_pose', 'past_commands',
          'future_commands', 'future_pose', 'target', 'valid', 'maze')
LABELS = ('walker', 'waiting', 'contact', 'kind', 'episode', 't0', 'visible_count', 'seen_count')
FUTURES, STEPS, FACTUAL = 7, 16, 6


def load(root, split, limit=None):
    """The split's arrays as numpy (memory-mapped), every field of the export."""
    folder = Path(root)/split
    return {k: np.load(folder/f'{k}.npy', mmap_mode='r')[:limit] for k in FIELDS+LABELS}


def rasters_for(root, corpus, device):
    """The wall rasters of every layout the export's two splits use."""
    mazes = sorted({int(m) for split in ('train', 'val') for m in np.unique(np.load(Path(root)/split/'maze.npy'))})
    return Rasters(Path(corpus)/'objects'/'rasters', mazes, device)


class Scenes:
    def __init__(self, root, split, rasters, device, limit=None):
        self.numpy = load(root, split, limit)
        for k in FIELDS:
            if k != 'maze':
                setattr(self, k, torch.as_tensor(np.ascontiguousarray(self.numpy[k]), device=device))
        self.layout = torch.as_tensor([rasters.index[int(m)] for m in self.numpy['maze']], device=device)
        self.rasters, self.device, self.count = rasters, device, len(self.layout)

    def frame(self, index):
        pose = self.history_pose[index, -1]
        return dict(origin=pose[:, :2], heading=pose[:, 2], layout=self.layout[index])

    def to_body(self, f, world):
        """World positions (B, ..., 2) in the scenes' frames."""
        lead = (len(world),)+(1,)*(world.dim()-2)
        return rotate(world-f['origin'].reshape(*lead, 2), -f['heading'].reshape(lead))

    def world(self, f, body):
        lead = (len(body),)+(1,)*(body.dim()-2)
        return rotate(body, f['heading'].reshape(lead))+f['origin'].reshape(*lead, 2)

    def walls(self, f, body):
        """The walls' distance on the object grid round positions (B, ..., 2) in the scenes' frames: (B, ..., 25)."""
        world = self.world(f, body)
        grid = rotate(offsets(OBJECT_GRID, OBJECT_SPACING, self.device), f['heading'][:, None])       # (B, G, 2)
        points = world.reshape(len(world), -1, 1, 2)+grid[:, None]
        return self.rasters.lookup(f['layout'], points).reshape(*world.shape[:-1], -1)

    def robot_tokens(self, f, pose):
        """Poses (B, T, 3) in the world to `robot_features` (B, T, 53)."""
        grid = rotate(offsets(ROBOT_GRID, ROBOT_SPACING, self.device)[None, None], pose[..., 2, None])
        walls = self.rasters.lookup(f['layout'], pose[:, :, None, :2]+grid)
        return robot_features(self.to_body(f, pose[..., :2]), pose[..., 2]-f['heading'][:, None], walls)

    def batch(self, index, future):
        """Scenes `index` (B,) under futures `future` (B,); see the module's description."""
        f = self.frame(index)
        ids = self.history_id[index]
        present = (ids == ids[:, -1:]) & (ids[:, -1:] >= 0)
        seen = self.history_seen[index] & present
        position = torch.nan_to_num(self.to_body(f, torch.nan_to_num(self.history_xy[index])))
        velocity = rotate(self.history_velocity[index].float(), -f['heading'][:, None, None])*seen[..., None]
        objects = object_features(position, seen, present, self.walls(f, position), velocity)
        steps = lambda commands: actions_input(commands.unflatten(1, (-1, STEP_FRAMES)))
        actions = torch.cat((steps(self.past_commands[index]), steps(self.future_commands[index, future])), 1)
        valid = self.valid[index, future]
        truth = self.to_body(f, self.target[index, future])*valid[..., None]
        before = torch.cat((position[:, -1:], truth[:, :-1]), 1)
        moving = (truth-before)/STEP_S*valid[..., None]
        future_objects = object_features(truth, valid, valid, self.walls(f, truth), moving)
        return dict(frame=f, objects=objects, robot=self.robot_tokens(f, self.history_pose[index]), actions=actions,
                    future_robot=self.robot_tokens(f, self.future_pose[index, future]), future_objects=future_objects,
                    truth=truth, valid=valid, seen=seen, present=present)
