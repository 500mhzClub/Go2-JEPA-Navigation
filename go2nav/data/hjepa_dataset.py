"""H-JEPA training batches from the LeWM corpus: clips long enough for the top level, each level's crop within them.

A clip is `HJEPAConfig.clip_length` consecutive frames of one unit (257 frames, 25.7 s, in the Go2 configuration), from
the corpus's window index of that length (`lewm_corpus index --length 257`), so clips never cross a unit. Per clip, as
in the paper (§C):
- level 1: a random crop of `history + rollout_steps` consecutive frames and their action blocks;
- each upper level l (period P_l level-1 steps, crop c_l states): a random crop of c_l consecutive level-l states,
  frames P_l apart, and the level-1 action blocks spanning its c_l - 1 transitions.
Only those frames are read (about 19 per clip, not 257). `upper_mask` marks the clips the upper levels train on (units
of the `upper_kinds`, mission and explore in the Go2 configuration).

Batches are a pure function of (seed, step), as `data.lewm_dataset`'s: the same maze-grouped clip draw
(`Corpus.sample`), then the crops from a keyed stream, so a resumed run sees exactly the batches it would have seen.
"""
import numpy as np
import torch

from go2nav.data.lewm_dataset import keyed_rng


def crop_layout(config, level1_length):
    """Per level, (period in level-1 steps, states in the crop); level 1 first."""
    return [(1, level1_length)]+[(p, u.crop) for p, u in zip(config.periods[1:], config.upper)]


def draw_crops(rng, clip_length, layout):
    """Per level, the crop's first state index in that level's own stream."""
    out = []
    for period, count in layout:
        states = (clip_length-1)//period+1
        out.append(int(rng.integers(0, states-count+1)))
    return out


def clip_batch(corpus, indices, rng, layout, clip_length, upper_kinds):
    """The batch dict of `models.hjepa.losses` for clips `indices` of `corpus`, crops drawn from `rng`."""
    frames = [[] for _ in layout]
    actions = [[] for _ in layout]
    mask = []
    for i in indices:
        unit, start = int(corpus.unit[i]), int(corpus.start[i])
        blocks = corpus.actions(unit)[start:start+clip_length-1]
        rows, spans = [], []
        for (period, count), first in zip(layout, draw_crops(rng, clip_length, layout)):
            rows.append([start+(first+k)*period for k in range(count)])
            spans.append(blocks[first*period:(first+count-1)*period])
        pixels = np.asarray(corpus.frames(unit)[np.concatenate(rows)])
        at = 0
        for level, row in enumerate(rows):
            frames[level].append(pixels[at:at+len(row)])
            actions[level].append(spans[level])
            at += len(row)
        mask.append(corpus.units[unit]['kind'] in upper_kinds)
    keys = np.stack([corpus.unit[indices], corpus.start[indices]], axis=1)
    return dict(frames=[torch.from_numpy(np.stack(f)) for f in frames],
                actions=[torch.from_numpy(np.stack(a)) for a in actions],
                upper_mask=torch.tensor(mask), keys=torch.from_numpy(keys))


class ClipBatches(torch.utils.data.Dataset):
    """Batches for steps [first, last): item k is batch `first + k` (see the module docstring)."""

    def __init__(self, corpus, config, *, level1_length, seed, first, last, mazes_per_batch=16, windows_per_maze=8,
                 batch_size=128, upper_kinds=('mission', 'explore')):
        if corpus.length != config.clip_length:
            raise ValueError(f'the corpus index has {corpus.length}-frame windows; clips need {config.clip_length}')
        self.corpus, self.seed, self.first, self.last = corpus, seed, first, last
        self.layout, self.clip_length = crop_layout(config, level1_length), config.clip_length
        self.mazes_per_batch, self.windows_per_maze, self.batch_size = mazes_per_batch, windows_per_maze, batch_size
        self.upper_kinds = tuple(upper_kinds)

    def __len__(self):
        return self.last-self.first

    def __getitem__(self, k):
        step = self.first+k
        indices = self.corpus.sample(self.seed, step, mazes_per_batch=self.mazes_per_batch,
                                     windows_per_maze=self.windows_per_maze, batch_size=self.batch_size)
        return clip_batch(self.corpus, indices, keyed_rng(self.seed, 'hjepa_crop', step), self.layout, self.clip_length,
                          self.upper_kinds)
