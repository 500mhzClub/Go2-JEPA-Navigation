"""The world model's walkers (docs/dynamic/PLAN.md, P6): finding them from the model's state instead of the depth map.

**Finding** (`WalkerFinder`). The four latest frames are encoded as C5's are, and the model's mover head reads each image
column of the state: a presence probability and the walker's body-frame position. A column above `PRESENCE` is a detection.

**Tracking** (`ModelTracker`). The hand tracker's association and velocity fit stay (`perception.tracks`), in the map frame
through the estimated pose: a detection joins the nearest track within `GATE_M` plus `MAX_SPEED_M_S` times the time since,
else starts one, and the velocity is the least-squares slope over `WINDOW_S`. What changes is the evidence. The head names
walkers, so a track needs `CONFIRM_DETECTIONS` detections, not motion in the map, and a standing walker is a track too.

A track is the head's centre, so its `cells` are a walker's disk, `stored_cells` the disk grown by `STORED_MARGIN_M` (what
the map perceived of it, give or take pose error) and `trail` the 5-cm cells it has crossed: the planner's track interface.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from go2nav.models.lewm import MOVER_SCALE_M, normalise_frames, resize_frame
from go2nav.perception.mapping import FINE_CELL_M
from go2nav.perception.tracks import GATE_M, KEEP_UNSEEN_S, MAX_SPEED_M_S, MIN_SPAN_S, RATIO, WINDOW_S, Track
from go2nav.sim.obstacles import RADIUS_M

PRESENCE = .5
CONFIRM_DETECTIONS = 3
STORED_MARGIN_M = .08


def disk(centre, radius_m):
    """The 1-cm map cells of a disk."""
    reach = int(np.ceil(radius_m/FINE_CELL_M))
    g = np.arange(-reach, reach+1)
    i, j = np.meshgrid(g, g, indexing='ij')
    keep = (i*FINE_CELL_M)**2+(j*FINE_CELL_M)**2 <= radius_m**2
    c = np.floor(np.asarray(centre, float)/FINE_CELL_M).astype(int)
    return frozenset(zip((i[keep]+c[0]).tolist(), (j[keep]+c[1]).tolist()))


@dataclass
class ModelTrack(Track):
    """A track from the head's detections: see the module docstring."""
    stored_cells: frozenset = frozenset()
    detections: int = 0

    @property
    def confirmed(self):
        return self.detections >= CONFIRM_DETECTIONS


class WalkerFinder:
    """See the module docstring. `detect` returns (frames, columns, 3): presence probability, x and y (m)."""

    def __init__(self, world_model, *, device='cuda:0'):
        if not world_model.config.mover_head:
            raise ValueError('the walker finder needs a world model with the mover head')
        self.device = torch.device(device)
        self.model = world_model.to(self.device).eval()

    @torch.inference_mode()
    def detect(self, packets):
        frames = torch.from_numpy(np.stack([resize_frame(p['image']['rgb']) for p in packets])).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            out = self.model.movers(self.model.encode(normalise_frames(frames)[None])[0]).float()
        return torch.cat((out[..., :1].sigmoid(), out[..., 1:]*MOVER_SCALE_M), -1).cpu().numpy()


class ModelTracker:
    """See the module docstring."""

    def __init__(self):
        self.tracks, self.next_id = [], 0

    def update(self, detected, times_ns, poses, map_from_initial):
        """Advance the tracks with `detect`'s frames. `times_ns` and `poses` are per frame, a pose (position, rotation) in
        the initial-body frame or None where the frame has none, and `map_from_initial` the map's orientation."""
        B = np.asarray(map_from_initial, float)
        for rows, now, pose in zip(detected, times_ns, poses):
            if pose is None:
                continue
            position, rotation = pose
            points = [(B@(position+rotation@np.array([x, y, 0.])))[:2] for p, x, y in rows if p > PRESENCE]
            self._step(points, int(now))
        return list(self.tracks)

    def _step(self, points, now):
        self.tracks = [t for t in self.tracks if now-t.seen_ns <= KEEP_UNSEEN_S*1e9]
        pairs = [(float(np.linalg.norm(c-t.history[-1][1])), k, t.id) for k, c in enumerate(points) for t in self.tracks
                 if np.linalg.norm(c-t.history[-1][1]) <= GATE_M+MAX_SPEED_M_S*(now-t.seen_ns)/1e9]
        used, taken, by_id = set(), set(), {t.id: t for t in self.tracks}
        for _, k, tid in sorted(pairs):
            if k not in used and tid not in taken:
                used.add(k)
                taken.add(tid)
                self._observe(by_id[tid], points[k], now)
        for k, c in enumerate(points):
            if k not in used:
                track = ModelTrack(self.next_id, frozenset(), now)
                self.next_id += 1
                self._observe(track, c, now)
                self.tracks.append(track)

    @staticmethod
    def _observe(track, centre, now):
        track.cells, track.stored_cells = disk(centre, RADIUS_M), disk(centre, RADIUS_M+STORED_MARGIN_M)
        track.footprint = frozenset((i//RATIO, j//RATIO) for i, j in track.cells)
        track.trail = track.trail|track.footprint
        track.seen_ns = now
        track.detections += 1
        track.motion_updates = track.detections
        track.history = [h for h in track.history if now-h[0] <= WINDOW_S*1e9]+[(now, np.asarray(centre, float))]
        times = np.array([(h[0]-now)/1e9 for h in track.history])
        if len(times) < 3 or times[-1]-times[0] < MIN_SPAN_S:
            track.velocity = np.zeros(2)
            return
        xy = np.array([h[1] for h in track.history])
        dt = times-times.mean()
        velocity = (dt[:, None]*(xy-xy.mean(0))).sum(0)/float((dt*dt).sum())
        speed = float(np.linalg.norm(velocity))
        track.velocity = velocity*min(1., MAX_SPEED_M_S/speed) if speed > 0 else velocity
