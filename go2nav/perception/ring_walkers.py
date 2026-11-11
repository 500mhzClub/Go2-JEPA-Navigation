"""Walkers from the range ring where the layout is known (docs/dynamic/PLAN.md P20): engineered estimation, nothing
privileged. The inputs are a frame's scan (`perception.range`), the robot's pose, and the range each beam would read
with no walker in the maze (`sim.kinematic.Geometry.wall_ranges`).

- **Detection** (`detect`): a valid return shorter than the walls' by `MARGIN_M` is a walker's. Such beams are grouped
  while they are neighbours (the ring closes on itself) and their ranges differ by less than `STEP_M`; a group of at
  least `MIN_BEAMS` gives one walker: a circle of the walkers' radius fitted to its points (Gauss-Newton from the
  nearest point pushed one radius along its beam).
- **Tracking** (`Tracker`): each frame's detections are joined to the nearest track within `GATE_M`. A track's velocity
  is the least-squares slope of its positions over the last `WINDOW_S` (zero until it spans `MIN_SPAN_S`). A track
  with no detection stays where it was last seen, at rest, until that place has been in the sensor's line of sight and
  empty for `EMPTY_FRAMES` frames, as P17's `frozen_seen` model forgets.
"""
from collections import deque

import numpy as np

from go2nav.perception.range import AZIMUTHS, BEAMS, MOUNT_BODY_M
from go2nav.sim.obstacles import RADIUS_M

MARGIN_M, STEP_M, MIN_BEAMS = .06, .25, 3
GATE_M, WINDOW_S, MIN_SPAN_S, EMPTY_FRAMES = .4, .5, .2, 3


def mount_of(xy, yaw):
    """The sensor's position (2,) for a robot pose."""
    return np.asarray(xy, float)+MOUNT_BODY_M[0]*np.array([np.cos(yaw), np.sin(yaw)])


def fit_circle(points, origin, radius=RADIUS_M, steps=4):
    """The centre (2,) of the circle of `radius` through `points` (n, 2), seen from `origin` (so on their far side)."""
    near = points[np.linalg.norm(points-origin, axis=1).argmin()]
    centre = near+radius*(near-origin)/max(float(np.linalg.norm(near-origin)), 1e-9)
    for _ in range(steps):
        offset = points-centre
        distance = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
        jacobian = -offset/distance[:, None]
        centre = centre+np.linalg.lstsq(jacobian, radius-distance, rcond=None)[0]
    return centre


def detect(ranges, valid, walls, xy, yaw):
    """The walkers a scan shows: centres (n, 2) in the world frame."""
    ranges = np.asarray(ranges, float)
    hit = np.asarray(valid, bool) & (ranges < np.asarray(walls, float)-MARGIN_M)
    if not hit.any():
        return np.zeros((0, 2))
    origin = mount_of(xy, yaw)
    start = int(np.flatnonzero(~hit)[0]) if not hit.all() else 0          # begin at a gap, so no group straddles the seam
    order = (start+np.arange(BEAMS)) % BEAMS
    groups, group = [], []
    for k in order:
        if hit[k] and (not group or abs(ranges[k]-ranges[group[-1]]) < STEP_M):
            group.append(k)
            continue
        if len(group) >= MIN_BEAMS:
            groups.append(group)
        group = [k] if hit[k] else []
    if len(group) >= MIN_BEAMS:
        groups.append(group)
    centres = []
    for group in groups:
        angle = yaw+AZIMUTHS[group]
        points = origin+ranges[group, None]*np.stack((np.cos(angle), np.sin(angle)), 1)
        centres.append(fit_circle(points, origin))
    return np.array(centres).reshape(-1, 2)


class Tracker:
    """See the module docstring. `update` takes a frame; `state` gives the tracks' positions and velocities."""

    def __init__(self):
        self.tracks, self.next_id = [], 0

    def update(self, t, detections, origin, geometry):
        """`t`: the frame's time (s); `detections`: (n, 2); `origin`: the sensor's position; `geometry`: for the line
        of sight (`sim.kinematic.Geometry.visible`)."""
        free = list(range(len(detections)))
        pairs = sorted((float(np.linalg.norm(track['xy']-detections[j])), i, j)
                       for i, track in enumerate(self.tracks) for j in free)
        matched = set()
        for distance, i, j in pairs:
            if distance > GATE_M or i in matched or j not in free:
                continue
            matched.add(i)
            free.remove(j)
            track = self.tracks[i]
            track['history'].append((t, detections[j]))
            while t-track['history'][0][0] > WINDOW_S:
                track['history'].popleft()
            times = np.array([h[0] for h in track['history']])
            track.update(xy=np.array(detections[j], float), empty=0, velocity=np.zeros(2))
            if times[-1]-times[0] >= MIN_SPAN_S:
                track['velocity'] = np.polyfit(times-times[-1], np.array([h[1] for h in track['history']]), 1)[0]
        kept = []
        for i, track in enumerate(self.tracks):
            if i not in matched:
                track.update(velocity=np.zeros(2), history=deque())
                track['empty'] = track['empty']+1 if geometry.visible(origin, track['xy']) else 0
            if track['empty'] < EMPTY_FRAMES:
                kept.append(track)
        self.tracks = kept
        for j in free:
            self.tracks.append(dict(id=self.next_id, xy=np.array(detections[j], float), velocity=np.zeros(2), empty=0,
                                    history=deque([(t, detections[j])])))
            self.next_id += 1

    def state(self):
        """(positions (n, 2), velocities (n, 2)) of the tracks."""
        xy = np.array([t['xy'] for t in self.tracks], float).reshape(-1, 2)
        return xy, np.array([t['velocity'] for t in self.tracks], float).reshape(-1, 2)
