"""The world model's walkers (docs/dynamic/PLAN.md, P6): finding them from the model's state instead of the depth map.

**Finding** (`WalkerFinder`). The four latest frames are encoded as C5's are (with one model, the same pass as the forecast:
`encoder`), and the model's mover head reads each image column of the state: a presence probability and the walker's
body-frame position. A column above `PRESENCE` is a detection.

**Tracking** (`ModelTracker`). The hand tracker's association and velocity fit stay (`perception.tracks`), in the map frame
through the estimated pose: detections within `MERGE_M` of each other in one frame are one walker seen by two columns, a
detection joins the nearest track within `GATE_M` plus `MAX_SPEED_M_S` times the time since, else starts one, and the velocity
is the least-squares slope over `WINDOW_S`. What changes is the evidence. The head names walkers, so a track needs
`CONFIRM_DETECTIONS` detections, not motion in the map, and a standing walker is a track too. A head's detections jump more
than the depth map's centroid and a new track has no velocity for `MIN_SPAN_S`, so the gate is wider and the span and window
differ from the hand tracker's (tuned on the head's 10-Hz detections of the validation mazes, PLAN P6).

**Held tracks** (`held`). While the robot turns, a standing walker's fitted speed is often false (above 0.15 m/s on 36% of
readings, 4% otherwise), and the planner backs away from what it predicts. With `held`, `velocity` stays zero and the fit is
kept as `fitted` for the logs: a track then says where a walker is, and the map's depth cells keep the clearance checks.

A track is the head's centre, so its `cells` are a walker's disk, `stored_cells` the disk grown by `STORED_MARGIN_M` (what
the map perceived of it, give or take pose error) and `trail` the 5-cm cells it has crossed: the planner's track interface.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from go2nav.models.lewm import MOVER_SCALE_M, PacketEncoder
from go2nav.perception.mapping import FINE_CELL_M
from go2nav.perception.tracks import KEEP_UNSEEN_S, MAX_SPEED_M_S, RATIO, Track
from go2nav.sim.obstacles import RADIUS_M

PRESENCE = .5
CONFIRM_DETECTIONS = 3
STORED_MARGIN_M = .08
GATE_M, MERGE_M = .25, .3
WINDOW_S, MIN_SPAN_S = 1.6, .5


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
    fitted: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def confirmed(self):
        return self.detections >= CONFIRM_DETECTIONS


def merged(points):
    """One point (the mean) for each group of a frame's detections within `MERGE_M` of the group's first."""
    out, used = [], set()
    for i, point in enumerate(points):
        if i not in used:
            group = [j for j in range(i, len(points)) if j not in used and np.linalg.norm(points[j]-point) <= MERGE_M]
            used.update(group)
            out.append(np.mean([points[j] for j in group], axis=0))
    return out


class WalkerFinder:
    """See the module docstring. `detect` returns (frames, columns, 3): presence probability, x and y (m)."""

    def __init__(self, world_model, *, device='cuda:0', encoder=None):
        if not world_model.config.mover_head:
            raise ValueError('the walker finder needs a world model with the mover head')
        self.device = torch.device(device)
        self.model = world_model.to(self.device).eval()
        self.encoder = encoder or PacketEncoder(self.model, self.device)

    @torch.inference_mode()
    def detect(self, packets):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            out = self.model.movers(self.encoder(packets)[0]).float()
        return torch.cat((out[..., :1].sigmoid(), out[..., 1:]*MOVER_SCALE_M), -1).cpu().numpy()


class ModelTracker:
    """See the module docstring."""

    def __init__(self, held=False):
        self.tracks, self.next_id, self.held = [], 0, held

    def update(self, detected, times_ns, poses, map_from_initial):
        """Advance the tracks with `detect`'s frames. `times_ns` and `poses` are per frame, a pose (position, rotation) in
        the initial-body frame or None where the frame has none, and `map_from_initial` the map's orientation."""
        B = np.asarray(map_from_initial, float)
        for rows, now, pose in zip(detected, times_ns, poses):
            if pose is None:
                continue
            position, rotation = pose
            points = [(B@(position+rotation@np.array([x, y, 0.])))[:2] for p, x, y in rows if p > PRESENCE]
            self._step(merged(points), int(now))
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

    def _observe(self, track, centre, now):
        track.cells, track.stored_cells = disk(centre, RADIUS_M), disk(centre, RADIUS_M+STORED_MARGIN_M)
        track.footprint = frozenset((i//RATIO, j//RATIO) for i, j in track.cells)
        track.trail = track.trail|track.footprint
        track.seen_ns = now
        track.detections += 1
        track.motion_updates = track.detections
        track.history = [h for h in track.history if now-h[0] <= WINDOW_S*1e9]+[(now, np.asarray(centre, float))]
        times = np.array([(h[0]-now)/1e9 for h in track.history])
        if len(times) < 3 or times[-1]-times[0] < MIN_SPAN_S:
            track.fitted = track.velocity = np.zeros(2)
            return
        xy = np.array([h[1] for h in track.history])
        dt = times-times.mean()
        velocity = (dt[:, None]*(xy-xy.mean(0))).sum(0)/float((dt*dt).sum())
        speed = float(np.linalg.norm(velocity))
        track.fitted = velocity*min(1., MAX_SPEED_M_S/speed) if speed > 0 else velocity
        track.velocity = np.zeros(2) if self.held else track.fitted
