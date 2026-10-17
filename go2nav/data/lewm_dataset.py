"""Training windows from the LeWM corpus (`go2nav.data.lewm_corpus`): frames and action blocks of `length`
consecutive 100-ms steps, batched by maze.

A batch is a pure function of (seed, step): `mazes_per_batch` mazes drawn without replacement, then
`windows_per_maze` windows of each (with replacement), all from keyed random streams. A run resumed at step k
therefore sees exactly the batches it would have seen. With `mazes_per_batch=None` the windows are drawn uniformly
from every window instead (the paper's i.i.d. batches). Few mazes per batch keeps SIGReg from being met by maze
identity alone: within a batch, the Gaussian spread has to come from views and poses.

`WindowBatches` is a map-style dataset over steps for a `torch.utils.data.DataLoader` with `batch_size=None` and
several workers (frames are memory-mapped .npy, so workers share the page cache). Each item: frames uint8
(B, length, 168, 224, 3), actions float32 (B, length-1, 10) (normalised blocks of five (forward, yaw) commands), and
the window keys (unit, start).
"""
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import torch

from go2nav.models.lewm import control_stats


def keyed_rng(*keys):
    return np.random.default_rng(int.from_bytes(hashlib.sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'big'))


def raise_open_file_limit():
    """Each unit's memory-mapped frames hold a file descriptor, so a full corpus needs thousands. systemd services
    start with a soft limit of 1,024; past it, every open fails, Triton's kernel loads included (driver error 301)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class Corpus:
    """The window index of a corpus root (built by `lewm_corpus index`), filtered by role and unit kind."""

    def __init__(self, root, *, length=12, roles=('lewm_train',), kinds=None):
        self.root, self.length = Path(root), length
        meta = json.loads((self.root/f'index_{length}.json').read_text())
        index = np.load(self.root/f'index_{length}.npz')
        self.units = meta['units']
        keep = np.array([u['role'] in roles and (kinds is None or u['kind'] in kinds) for u in self.units], bool)
        chosen = keep[index['unit']] if len(index['unit']) else np.zeros(0, bool)
        self.unit, self.start = index['unit'][chosen], index['start'][chosen]
        mazes = np.array([u['maze'] for u in self.units])[self.unit] if len(self.unit) else np.zeros(0, int)
        self.mazes = np.unique(mazes)
        self.by_maze = {int(m): np.flatnonzero(mazes == m) for m in self.mazes}
        if not len(self.unit):
            raise ValueError(f'no windows of length {length} for roles {roles} and kinds {kinds}')
        self._frames, self._actions = {}, {}
        raise_open_file_limit()
        self.mean, self.std = control_stats()

    def __len__(self):
        return len(self.unit)

    def frames(self, unit):
        if unit not in self._frames:
            self._frames[unit] = np.load(self.root/self.units[unit]['path']/'frames.npy', mmap_mode='r')
        return self._frames[unit]

    def actions(self, unit):
        """Normalised action blocks (n, 10) of a unit."""
        if unit not in self._actions:
            with np.load(self.root/self.units[unit]['path']/'steps.npz') as steps:
                executed = steps['executed'][:, :, [0, 2]].astype(np.float32)
            self._actions[unit] = ((executed-self.mean)/self.std).reshape(len(executed), -1)
        return self._actions[unit]

    def window(self, i):
        u, s = int(self.unit[i]), int(self.start[i])
        return np.asarray(self.frames(u)[s:s+self.length]), self.actions(u)[s:s+self.length-1]

    def batch(self, indices):
        frames, actions = zip(*(self.window(i) for i in indices))
        keys = np.stack([self.unit[indices], self.start[indices]], axis=1)
        return torch.from_numpy(np.stack(frames)), torch.from_numpy(np.stack(actions)), torch.from_numpy(keys)

    def sample(self, seed, step, *, mazes_per_batch, windows_per_maze, batch_size=None):
        """The window indices of batch `step`."""
        rng = keyed_rng(seed, 'lewm_batch', step)
        if mazes_per_batch is None:
            return rng.integers(len(self.unit), size=batch_size)
        mazes = rng.choice(self.mazes, size=min(mazes_per_batch, len(self.mazes)), replace=False)
        return np.concatenate([rng.choice(self.by_maze[int(m)], size=windows_per_maze) for m in mazes])

    def fixed(self, count, seed):
        """A fixed, seeded selection of windows (validation)."""
        rng = keyed_rng(seed, 'lewm_fixed', count)
        return np.sort(rng.choice(len(self.unit), size=min(count, len(self.unit)), replace=False))


class WindowBatches(torch.utils.data.Dataset):
    """Batches for steps [first, last): item k is batch `first + k` (see the module docstring)."""

    def __init__(self, corpus, *, seed, first, last, mazes_per_batch=16, windows_per_maze=8, batch_size=128):
        self.corpus, self.seed, self.first, self.last = corpus, seed, first, last
        self.mazes_per_batch, self.windows_per_maze, self.batch_size = mazes_per_batch, windows_per_maze, batch_size

    def __len__(self):
        return self.last-self.first

    def __getitem__(self, k):
        indices = self.corpus.sample(self.seed, self.first+k, mazes_per_batch=self.mazes_per_batch,
                                     windows_per_maze=self.windows_per_maze, batch_size=self.batch_size)
        return self.corpus.batch(indices)
