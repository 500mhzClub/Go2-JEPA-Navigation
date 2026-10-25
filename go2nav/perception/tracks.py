"""Constant-velocity tracks of moving obstacles, from the routing map's current 1-cm cells (docs/dynamic/PLAN.md, P2b).

The non-learned motion-prediction baseline (constant velocity; Schöller et al., 2020). Nothing here knows the shape
or number of the obstacles.

**Detection** (each map update, `ObstacleTracker.update`). This observation's 1-cm occupied cells are grouped into
clusters (cells within `JOIN_M` of each other). A cluster is a candidate when it is compact (bounding-box diagonal at
most `MAX_EXTENT_M`, at least `MIN_CELLS` cells) and at least `MIN_FLOOR_SHARE` of its 5-cm cells were once seen as
floor: something now stands where the floor was seen free. A wall's cell is never seen as floor, so walls, and a wall
end seen through a gap, never become candidates.

**Evidence of motion.** A static fragment's centroid also drifts as its visible part changes with the view, so a
track is predicted only once it has shown motion in `CONFIRM_UPDATES` updates, each with at least `MOTION_CELLS` 5-cm
cells that it either *entered* (now occupied, seen as free floor in an earlier update within `RECENT_FREE_S`) or
*vacated* (occupied by the track before, seen as free floor now). A wall neither enters floor seen free nor leaves its
cells free behind it.

**Tracking.** Each candidate joins the nearest track whose last centroid lies within `GATE_M` plus `MAX_SPEED_M_S`
times the time since, else starts a track. The velocity is the least-squares slope of the centroid over the last
`WINDOW_S`, from at least three observations spanning `MIN_SPAN_S`, capped at `MAX_SPEED_M_S`. Only the visible face
is measured, so the centroid sits nearer the robot than the true centre; the offset changes slowly with the view, so
the slope is still the obstacle's motion. A track not seen for `KEEP_UNSEEN_S` is dropped (its cells stay in the map
as stored cells).

**Trails** (`trail`): every 5-cm cell a track has occupied; with mover trails on, the routing map keeps a confirmed
track's trail only while it is seen occupied (`perception.mapping`).

**Moving tracks** (`moving`): confirmed, with speed at least `MIN_SPEED_M_S`. The planner takes their cells out of the stored cells
and checks each forecast segment against them shifted by the velocity times the segment's time
(`planning.prediction`).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from go2nav.perception.mapping import CELL_M, FINE_CELL_M

JOIN_M, MIN_CELLS, MAX_EXTENT_M, MIN_FLOOR_SHARE = .03, 20, .60, .5
GATE_M, MAX_SPEED_M_S, MIN_SPEED_M_S = .15, .60, .05
WINDOW_S, MIN_SPAN_S, KEEP_UNSEEN_S = 1.2, .7, 1.2
RECENT_FREE_S, MOTION_CELLS, CONFIRM_UPDATES = 3., 2, 2
RATIO = int(round(CELL_M/FINE_CELL_M))


@dataclass
class Track:
    """One tracked obstacle: its latest cells (1-cm map indices), centroid history and velocity (map frame, m/s)."""
    id: int
    cells: frozenset
    seen_ns: int
    history: list = field(default_factory=list)   # [(t_ns, centroid xy)]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    footprint: frozenset = frozenset()             # its 5-cm cells at the previous observation
    trail: frozenset = frozenset()                 # every 5-cm cell it has occupied
    motion_updates: int = 0                        # updates that showed motion (see the module docstring)

    @property
    def confirmed(self):
        return self.motion_updates >= CONFIRM_UPDATES

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    def summary(self, now_ns):
        return dict(id=self.id, centroid_m=[round(float(v), 3) for v in self.history[-1][1]],
                    velocity_m_s=[round(float(v), 3) for v in self.velocity], cells=len(self.cells),
                    observations=len(self.history), age_s=round((now_ns-self.seen_ns)/1e9, 2),
                    motion_updates=self.motion_updates)


def clusters(cells):
    """Groups of 1-cm cells within `JOIN_M` of each other, as int arrays (n, 2)."""
    if not cells:
        return []
    points = np.asarray(sorted(cells), dtype=int)
    reach = int(round(JOIN_M/FINE_CELL_M))
    low = points.min(0)-reach
    shape = tuple(points.max(0)-low+reach+1)
    mask = np.zeros(shape, bool)
    local = points-low
    mask[local[:, 0], local[:, 1]] = True
    joined = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=reach)
    labels, _ = ndimage.label(joined, structure=np.ones((3, 3), bool))
    label = labels[local[:, 0], local[:, 1]]
    order = np.argsort(label, kind='stable')
    split = np.flatnonzero(np.diff(label[order]))+1
    return [points[g] for g in np.split(order, split)]


def candidate(points, floor):
    """Whether a cluster is a compact obstacle standing on once-seen floor (see the module docstring)."""
    if len(points) < MIN_CELLS:
        return False
    extent = (points.max(0)-points.min(0)+1)*FINE_CELL_M
    if float(np.linalg.norm(extent)) > MAX_EXTENT_M:
        return False
    coarse = {(int(i)//RATIO, int(j)//RATIO) for i, j in points}
    return sum(c in floor for c in coarse) >= MIN_FLOOR_SHARE*len(coarse)


def centroid(points):
    return (points.mean(0)+.5)*FINE_CELL_M


class ObstacleTracker:
    """See the module docstring."""

    def __init__(self):
        self.tracks, self.next_id = [], 0
        self.free_ns = {}   # 5-cm cell -> when last seen as free floor (floor, nothing above it)

    def update(self, snapshot):
        """Advance the tracks with one map snapshot; returns the live tracks."""
        now = int(snapshot.measured_ns)
        found = [p for p in clusters(snapshot.current_fine_occupied) if candidate(p, snapshot.floor)]
        free = snapshot.current_floor-snapshot.current_occupied
        self.tracks = [t for t in self.tracks if now-t.seen_ns <= KEEP_UNSEEN_S*1e9]
        pairs = []
        for k, points in enumerate(found):
            c = centroid(points)
            for t in self.tracks:
                gap = float(np.linalg.norm(c-t.history[-1][1]))
                if gap <= GATE_M+MAX_SPEED_M_S*(now-t.seen_ns)/1e9:
                    pairs.append((gap, k, t.id))
        used, taken = set(), set()
        by_id = {t.id: t for t in self.tracks}
        for _, k, tid in sorted(pairs):
            if k in used or tid in taken:
                continue
            used.add(k)
            taken.add(tid)
            self._observe(by_id[tid], found[k], now, free)
        for k, points in enumerate(found):
            if k not in used:
                track = Track(self.next_id, frozenset(), now)
                self.next_id += 1
                self._observe(track, points, now, free)
                self.tracks.append(track)
        self.free_ns.update(dict.fromkeys(free, now))
        return list(self.tracks)

    def _observe(self, track, points, now, free):
        coarse = frozenset((int(i)//RATIO, int(j)//RATIO) for i, j in points)
        entered = sum(0 < now-self.free_ns.get(c, -10**19) <= RECENT_FREE_S*1e9 for c in coarse)
        vacated = len((track.footprint-coarse)&free)
        track.motion_updates += entered+vacated >= MOTION_CELLS
        track.footprint = coarse
        track.trail = track.trail|coarse
        track.cells = frozenset(map(tuple, points.tolist()))
        track.seen_ns = now
        track.history = [h for h in track.history if now-h[0] <= WINDOW_S*1e9]+[(now, centroid(points))]
        times = np.array([(h[0]-now)/1e9 for h in track.history])
        if len(times) < 3 or times[-1]-times[0] < MIN_SPAN_S:
            track.velocity = np.zeros(2)
            return
        xy = np.array([h[1] for h in track.history])
        dt = times-times.mean()
        velocity = (dt[:, None]*(xy-xy.mean(0))).sum(0)/float((dt*dt).sum())
        speed = float(np.linalg.norm(velocity))
        track.velocity = velocity*min(1., MAX_SPEED_M_S/speed) if speed > 0 else velocity


def moving(tracks):
    """The confirmed tracks fast enough to predict."""
    return [t for t in tracks if t.confirmed and t.speed >= MIN_SPEED_M_S]
