"""Constant-velocity obstacle prediction in the forecast clearance check (docs/dynamic/PLAN.md, P2b).

A moving track's cells (`perception.tracks`) leave the stored cells. Each forecast segment, from `t0` to `t1` seconds
after the plan, is checked against them where the track will be: its cells shifted by the velocity times the time since
they were seen. Equivalently, the segment moves the other way in the track's frame, from `a - v(L+t0)` to
`b - v(L+t1)` (L: the cells' age at the plan), which is exact for constant velocity.

Only `selection.clearance_check` (and the turn re-checks that reuse its segment clearances) sees the prediction. The
stopping projection, the dispatch depth stop, the route and the coverage rule keep the stored cells, moving or not.
"""
from go2nav.planning import grid


class PredictedClearance:
    """A clearance (`inner`: stored cells without the moving tracks) plus the moving tracks, predicted."""

    def __init__(self, inner, tracks, now_ns):
        self.inner = inner
        self.tracks = [(grid.FineClearance(t.cells), t.velocity, (now_ns-t.seen_ns)/1e9) for t in tracks]

    def minimum(self, start, end):
        return self.minimum_during(start, end, 0., 0.)

    def minimum_during(self, start, end, t0, t1):
        """Clearance of the segment from `start` (at `t0` s after the plan) to `end` (at `t1`)."""
        value = self.inner.minimum(start, end)
        for clearance, velocity, age in self.tracks:
            moved = clearance.minimum(start-velocity*(age+t0), end-velocity*(age+t1))
            if moved is not None:
                value = moved if value is None else min(value, moved)
        return value

    def __getattr__(self, name):
        return getattr(self.inner, name)


def without(cells, tracks):
    """The stored 1-cm cells without the tracks' cells."""
    removed = frozenset().union(*(t.cells for t in tracks)) if tracks else frozenset()
    return cells-removed if removed else cells
