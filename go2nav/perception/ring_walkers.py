"""Walkers from the range ring where the layout is known (docs/dynamic/PLAN.md P20): engineered estimation, nothing
privileged. The inputs are a frame's scan (`perception.range`), the robot's pose, and the walls' distance at any point
(`planning.fields.Fields.wall_clearance`, from the known layout).

- **Detection** (`detect`, `detect_batch`): a valid return whose point lies more than `MARGIN_M` from every wall is a
  walker's. Such beams are grouped while they are neighbours (the ring closes on itself) and their ranges differ by
  less than `STEP_M`; a group of at least `MIN_BEAMS` gives one walker: a circle of the walkers' radius fitted to its
  points (Gauss-Newton from the nearest point pushed one radius along its beam; at most `FIT_POINTS` of them, evenly
  spaced along the group, so a long group costs no more than a short one). The rule was first a range shorter
  than the walls' cast range by the margin; casting the walls costs 1.5 ms a scan, and P21 reads walkers from
  thousands of predicted scans a plan, so both the tracker and P21 use the wall-distance lookup.
- **Tracking** (`Tracker`): each frame's detections are joined to the nearest track within `GATE_M`. A track's velocity
  is the least-squares slope of its positions over the last `WINDOW_S` (zero until it spans `MIN_SPAN_S`). The window
  was 0.5 s at first: a walker that stopped beside the robot was still read as walking on for half a second, and the
  planner drove into it (all 14 walker contacts of the first gate suite). At 0.3 s, contacts with walkers that do not
  yield fall from 15 to 10 in 80 missions; at 0.2 s the noise costs round trips (STATUS P20). A track
  with no detection stays where it was last seen, at rest, until that place has been in the sensor's line of sight and
  empty for `EMPTY_FRAMES` frames, as P17's `frozen_seen` model forgets.
"""
from collections import deque

import numpy as np

from go2nav.perception.range import AZIMUTHS, BEAMS, MOUNT_BODY_M
from go2nav.sim.obstacles import RADIUS_M

MARGIN_M, STEP_M, MIN_BEAMS = .06, .25, 3
GATE_M, WINDOW_S, MIN_SPAN_S, EMPTY_FRAMES = .4, .3, .15, 3
FIT_STEPS, FIT_POINTS = 4, 32


def mount_of(xy, yaw):
    """The sensor's position (2,) for a robot pose."""
    return np.asarray(xy, float)+MOUNT_BODY_M[0]*np.array([np.cos(yaw), np.sin(yaw)])


def detect_batch(ranges, valid, poses, wall_clearance):
    """The walkers each of S scans shows: centres (S, M, 2) in the world frame, NaN past each scan's count, and the
    counts (S,). `ranges`, `valid` (S, BEAMS); `poses` (S, 3): x, y, yaw of the robot; `wall_clearance`: points
    (..., 2) to the walls' distance."""
    ranges, valid, poses = np.asarray(ranges, float), np.asarray(valid, bool), np.asarray(poses, float)
    count = len(ranges)
    yaw = poses[:, 2]
    origin = poses[:, :2]+MOUNT_BODY_M[0]*np.stack((np.cos(yaw), np.sin(yaw)), 1)                    # (S, 2)
    angle = yaw[:, None]+AZIMUTHS[None]
    points = origin[:, None]+ranges[..., None]*np.stack((np.cos(angle), np.sin(angle)), -1)          # (S, BEAMS, 2)
    hit = valid & (wall_clearance(points) > MARGIN_M)
    first = np.where(hit.all(1), 0, (~hit).argmax(1))                     # begin at a gap: no group straddles the seam
    order = (first[:, None]+np.arange(BEAMS)[None]) % BEAMS
    hit, r = np.take_along_axis(hit, order, 1), np.take_along_axis(ranges, order, 1)
    points = np.take_along_axis(points, order[..., None], 1)
    breaks = np.ones_like(hit)
    breaks[:, 1:] = ~hit[:, :-1] | (np.abs(np.diff(r, axis=1)) >= STEP_M)
    run = np.cumsum(hit & breaks, axis=1)
    rows, beams = np.nonzero(hit)
    out, found = np.full((count, 0, 2), np.nan), np.zeros(count, int)
    if not len(rows):
        return out, found
    key = rows*(BEAMS+1)+run[rows, beams]
    groups, inverse, sizes = np.unique(key, return_inverse=True, return_counts=True)
    kept = sizes >= MIN_BEAMS
    if not kept.any():
        return out, found
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    place = np.arange(len(key))-starts[inverse]                             # each beam's index in its group
    stride = -(-sizes//FIT_POINTS)                                          # ceil: at most FIT_POINTS a group
    take = kept[inverse] & (place % stride[inverse] == 0)
    slot = place//stride[inverse]
    width = int(min(sizes[kept].max(), FIT_POINTS))
    pts = np.full((len(groups), width, 2), np.nan)
    pts[inverse[take], slot[take]] = points[rows[take], beams[take]]
    pts, owner = pts[kept], groups[kept]//(BEAMS+1)
    mask = ~np.isnan(pts[..., 0])
    seen_from = origin[owner][:, None]
    reach = np.where(mask, np.linalg.norm(pts-seen_from, axis=-1), np.inf)
    near = pts[np.arange(len(pts)), reach.argmin(1)]
    direction = near-origin[owner]
    centre = near+RADIUS_M*direction/np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    filled = np.where(mask[..., None], pts, 0.)
    for _ in range(FIT_STEPS):
        offset = np.where(mask[..., None], filled-centre[:, None], 0.)
        distance = np.maximum(np.linalg.norm(offset, axis=-1), 1e-9)
        jacobian = np.where(mask[..., None], -offset/distance[..., None], 0.)
        residual = np.where(mask, RADIUS_M-distance, 0.)
        jx, jy = jacobian[..., 0], jacobian[..., 1]
        a, b, d = (jx*jx).sum(1)+1e-9, (jx*jy).sum(1), (jy*jy).sum(1)+1e-9
        rx, ry = (jx*residual).sum(1), (jy*residual).sum(1)
        det = a*d-b*b
        centre = centre+np.stack(((d*rx-b*ry)/det, (a*ry-b*rx)/det), 1)
    found = np.bincount(owner, minlength=count)
    out = np.full((count, int(found.max()), 2), np.nan)
    slot = np.arange(len(owner))-np.searchsorted(owner, owner)              # owner is sorted (keys are)
    out[owner, slot] = centre
    return out, found


def detect(ranges, valid, xy, yaw, wall_clearance):
    """The walkers one scan shows: centres (n, 2) in the world frame."""
    centres, found = detect_batch(np.asarray(ranges)[None], np.asarray(valid)[None], np.array([[*xy, yaw]], float),
                                  wall_clearance)
    return centres[0, :found[0]]


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


class Slots:
    """The tracker's tracks in a fixed number of slots, for the object-level predictor (docs/dynamic/PLAN.md P22): a
    track keeps its slot for its life; a new one takes a free slot, else the slot of the remembered track seen longest
    ago, else it is left out (every slot holds a track seen this frame)."""

    def __init__(self, count):
        self.ids, self.last_seen = [None]*count, [-1]*count

    def update(self, tracker, frame):
        """Returns xy (count, 2) with NaN in empty slots, seen (count,) bool, id (count,) int, how many tracks were
        left out, and the tracks' velocities (count, 2) (zero for a remembered track and an empty slot)."""
        alive = {track['id']: track for track in tracker.tracks}
        self.ids = [i if i in alive else None for i in self.ids]
        left_out = 0
        for track_id in sorted(alive):
            if track_id in self.ids:
                continue
            if None in self.ids:
                k = self.ids.index(None)
            else:
                stale = [k for k, i in enumerate(self.ids) if not alive[i]['history']]
                if not stale:
                    left_out += 1
                    continue
                k = min(stale, key=lambda j: self.last_seen[j])
            self.ids[k], self.last_seen[k] = track_id, frame
        count = len(self.ids)
        xy, seen, ids = np.full((count, 2), np.nan), np.zeros(count, bool), np.full(count, -1)
        velocity = np.zeros((count, 2))
        for k, track_id in enumerate(self.ids):
            if track_id is None:
                continue
            track = alive[track_id]
            xy[k], seen[k], ids[k], velocity[k] = track['xy'], bool(track['history']), track_id, track['velocity']
            if seen[k]:
                self.last_seen[k] = frame
        return xy, seen, ids, left_out, velocity

