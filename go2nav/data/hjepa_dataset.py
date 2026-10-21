"""H-JEPA training batches from the LeWM corpus: level 1's windows and the upper levels' clips, from the same mazes.

- **Level 1** draws exactly what span8w's LeWM drew: `windows_per_maze` windows of `history + rollout_steps` frames
  from each of `mazes_per_batch` mazes (`data.lewm_dataset.Corpus.sample`, the same keyed stream). With the same seed,
  level 1 sees span8w's batches step for step, so the only difference from the flat model is the upper levels'
  gradients.
- **Upper levels** draw `clips_per_maze` clips from the same mazes' `upper_kinds` units (mission and explore in the Go2
  configuration; the scripted free units are noise at these timescales, docs/hjepa/TIMESCALES.md). A clip is
  `HJEPAConfig.clip_length` consecutive frames of one unit (257 frames, 25.7 s), from the window index of that length
  (`lewm_corpus index --length 257`), so it never crosses a unit. Per clip, each upper level l (period P_l level-1
  steps, crop c_l states) takes a random crop of c_l consecutive level-l states, frames P_l apart, with the level-1
  action blocks spanning its transitions, as in the paper (§C). Only those frames are read.

The batch sizes are fixed (`torch.compile`). Batches are a pure function of (seed, step), so a resumed run sees exactly
the batches it would have seen.
"""
import numpy as np
import torch

from go2nav.data.lewm_dataset import keyed_rng


def upper_layout(config):
    """Per upper level, (period in level-1 steps, states in the crop)."""
    return [(p, u.crop) for p, u in zip(config.periods[1:], config.upper)]


def draw_crops(rng, clip_length, layout):
    """Per level, the crop's first state index in that level's own stream."""
    return [int(rng.integers(0, (clip_length-1)//period+1-count+1)) for period, count in layout]


def clip_crops(corpus, indices, rng, layout):
    """Per upper level, the crops' frames (B, c_l, H, W, 3) uint8 and level-1 action blocks (B, (c_l - 1) P_l, A)."""
    frames = [[] for _ in layout]
    actions = [[] for _ in layout]
    for i in indices:
        unit, start = int(corpus.unit[i]), int(corpus.start[i])
        blocks = corpus.actions(unit)[start:start+corpus.length-1]
        rows = []
        for level, ((period, count), first) in enumerate(zip(layout, draw_crops(rng, corpus.length, layout))):
            rows.append([start+(first+k)*period for k in range(count)])
            actions[level].append(blocks[first*period:(first+count-1)*period])
        pixels = np.asarray(corpus.frames(unit)[np.concatenate(rows)])
        at = 0
        for level, row in enumerate(rows):
            frames[level].append(pixels[at:at+len(row)])
            at += len(row)
    return [torch.from_numpy(np.stack(f)) for f in frames], [torch.from_numpy(np.stack(a)) for a in actions]


def upper_draw(corpus, mazes, rng, clips_per_maze):
    """`clips_per_maze` clip indices from each maze (with replacement)."""
    return np.concatenate([rng.choice(corpus.by_maze[int(m)], size=clips_per_maze) for m in mazes])


class HJEPABatches(torch.utils.data.Dataset):
    """Batches for steps [first, last): item k is batch `first + k` (see the module docstring). `windows` is the
    level-1 Corpus (12-frame windows, training units of every kind); `clips` the upper levels' (clip-length windows of
    the `upper_kinds` units)."""

    def __init__(self, windows, clips, config, *, seed, first, last, mazes_per_batch=16, windows_per_maze=8,
                 clips_per_maze=8):
        if clips.length != config.clip_length:
            raise ValueError(f'the clip index has {clips.length}-frame windows; clips need {config.clip_length}')
        missing = set(windows.mazes.tolist())-set(clips.mazes.tolist())
        if missing:
            raise ValueError(f'{len(missing)} training mazes have no clip of the upper kinds')
        self.windows, self.clips, self.seed, self.first, self.last = windows, clips, seed, first, last
        self.layout = upper_layout(config)
        self.mazes_per_batch, self.windows_per_maze, self.clips_per_maze = mazes_per_batch, windows_per_maze, clips_per_maze

    def __len__(self):
        return self.last-self.first

    def __getitem__(self, k):
        step = self.first+k
        indices = self.windows.sample(self.seed, step, mazes_per_batch=self.mazes_per_batch,
                                      windows_per_maze=self.windows_per_maze)
        frames, actions, keys = self.windows.batch(indices)
        mazes = np.unique(np.array([u['maze'] for u in self.windows.units])[self.windows.unit[indices]])
        rng = keyed_rng(self.seed, 'hjepa_upper', step)
        upper_frames, upper_actions = clip_crops(self.clips, upper_draw(self.clips, mazes, rng, self.clips_per_maze), rng,
                                                 self.layout)
        return dict(frames=[frames, *upper_frames], actions=[actions, *upper_actions], keys=keys)
