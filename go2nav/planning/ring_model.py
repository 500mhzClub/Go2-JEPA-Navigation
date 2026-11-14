"""The learned ring model as the walkers' part of the sampling planner's model (docs/dynamic/PLAN.md P21 item 2).

`LearnedRingModel` keeps the last `history` scans the robot saw and the commands it requested after them (`observe`,
`record`). At a plan (`walkers`), for every tape it rolls the world model out over the tape's frames from that history
(`models.ring_jepa.RingJEPA.rollout`: one batch, every tape at once), decodes each predicted latent to a scan, and
reads the walkers out of each predicted scan at the pose the tape reaches there with the hand tracker's own detection
(`perception.ring_walkers.detect_batch`, against the known walls). No simulator state is read: the scans, the
commands and the layout.

Walkers are read from every predicted frame of the first `DENSE_FRAMES` (1.6 s) and from every `SPARSE_EVERY`-th
after it, each read standing for the frames up to the next (the far frames matter less to the clearance, and reading
every frame cost 1.3 s a plan). The planner's cost wants the walkers as (tapes, frames, n, 2). A predicted scan may show any number; each frame's
detections fill the first slots and the rest are placed `FAR` away, where they cost nothing.
"""
import numpy as np
import torch

from go2nav.models.ring_jepa import command_input, ranges_of, scan_input
from go2nav.perception import ring_walkers

FAR = 1e3
MAX_WALKERS = 8
DENSE_FRAMES, SPARSE_EVERY = 16, 4


class LearnedRingModel:
    takes_tapes = True

    def __init__(self, path, device=None):
        from go2nav.training.ring_jepa import load_model
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = load_model(path, self.device)
        self.history = self.model.config.history
        self.scans, self.commands, self.fields = [], [], None
        self.cache = None

    def bind(self, fields):
        self.fields = fields

    def observe(self, t, scan, xy, yaw, geometry):
        self.scans = (self.scans+[np.where(scan[1], scan[0], 0.).astype(np.float32)])[-self.history:]
        self.cache = None

    def record(self, requested):
        """The command requested at the frame just observed."""
        self.commands = (self.commands+[np.asarray(requested, np.float32)])[-(self.history-1):]

    @torch.no_grad()
    def predicted_scans(self, requested):
        """Predicted ranges (K, T, 720) for tapes of requests (K, T, 3), from the current history."""
        scans = list(self.scans)
        scans = [scans[0]]*(self.history-len(scans))+scans                        # the first frames of a mission
        past = list(self.commands)
        past = [np.zeros(3, np.float32)]*(self.history-1-len(past))+past
        count = len(requested)
        if self.cache is None:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                self.cache = self.model.encode(scan_input(torch.as_tensor(np.stack(scans), device=self.device)))
        z = self.cache[None].expand(count, -1, -1, -1)
        actions = np.concatenate((np.broadcast_to(np.stack(past), (count, self.history-1, 3)), requested), 1)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            rolled = self.model.rollout(z, command_input(torch.as_tensor(actions, device=self.device)))
            decoded = self.model.decode(rolled).float()
        return ranges_of(decoded).cpu().numpy()

    def walkers(self, snapshot, xy, yaw, frames, geometry=None, requested=None):
        count, horizon = xy.shape[:2]
        read = np.unique(np.concatenate((np.arange(min(DENSE_FRAMES, horizon)),
                                         np.arange(DENSE_FRAMES+SPARSE_EVERY-1, horizon, SPARSE_EVERY), [horizon-1])))
        ranges = self.predicted_scans(np.asarray(requested, np.float32))[:, read].reshape(count*len(read), -1)
        poses = np.concatenate((xy[:, read].reshape(-1, 2), yaw[:, read].reshape(-1, 1)), 1)
        centres, found = ring_walkers.detect_batch(ranges, ranges > 0, poses, self.fields.wall_clearance)
        n = int(min(max(found.max(initial=0), 1), MAX_WALKERS))
        out = np.full((count*len(read), n, 2), FAR)
        width = min(centres.shape[1], n)
        if width:
            take = centres[:, :width]
            out[:, :width] = np.where(np.isnan(take), FAR, take)
        out = out.reshape(count, len(read), n, 2)
        stand = np.searchsorted(read, np.arange(horizon))                   # each frame takes the next read at or after it
        return out[:, stand]
