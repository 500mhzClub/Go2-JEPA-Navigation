"""Cautious clearance for the calibrated-margin experiment: a margin on remembered walls, and never-observed cells
counted as obstacles ("pessimistic unknown").

Both act only where a forecast meets the map: the forecast-path clearance check (`selection.clearance_check`, and C2's
dispatched-path check) and the route-target look-ahead (`routing.clear_route_target`). The stopping projection, the
dispatch depth stop, the coverage rule and the routing graph keep the true distances.

- **Margin**: every remembered-wall distance is reduced by the controller's calibrated forecast-error bound (p95 or
  p99; configs/calibrated_margins_2026-10-02.json), so every clearance requirement grows by that bound.
- **Pessimistic unknown**: never-observed 5-cm cells within 0.8 m block a move when the forecast centre path comes
  within body reach (0.425 m) plus the calibrated bound of them: they enter as a second clearance, shifted so that they
  block exactly at that radius under the 0.48-m requirement. Known free without observation: cells within 0.20 m of a
  past planning position (the traversed track) and the start area. Variants:
  - `revised`: the start area is the 0.425-m reach disc (coarse cells whose centre lies within it);
  - `seeded`: the start area is the 0.5-m cleared disc of the operating precondition, at 1-cm resolution;
  - `lookaround`: `seeded`, and never-observed cells do not block during the scripted initial look-around while the
    body centre stays within 7 cm of the start.
"""
import json
import math
from pathlib import Path

import numpy as np

from go2nav.planning import grid

MARGINS_FILE = Path(__file__).resolve().parents[2]/'configs/calibrated_margins_2026-10-02.json'
BODY_REACH_M, START_REACH_M, TRACK_RADIUS_M, SCOPE_M, CHECK_REQUIRED_M = .425, .425, .20, .80, .48
PRECONDITION_CLEAR_M, LOOKAROUND_DRIFT_M = .50, .07
LOOKAROUND_STATUS = 'INITIAL_PANORAMA_REQUIRES_VIEW'
VARIANTS = ('revised', 'seeded', 'lookaround')
_OFFSETS = np.stack(np.meshgrid(np.arange(5), np.arange(5), indexing='ij'), axis=-1).reshape(-1, 2)


def calibrated_bound(arm, level):
    """A controller's calibrated forecast-error bound (m) at p95 or p99."""
    return json.loads(MARGINS_FILE.read_text())['controllers'][arm][level]['margin_m']


class CautiousClearance:
    """A clearance with remembered-wall distances reduced by `margin` and, optionally, never-observed cells
    (`unknown`: a clearance and the offset added to its distances) as a second obstacle set."""

    def __init__(self, inner, margin=0., unknown=None):
        self.inner, self.margin = inner, margin
        self.hidden, self.offset = unknown if unknown is not None else (None, 0.)

    def minimum(self, start, end):
        value = self.inner.minimum(start, end)
        value = None if value is None else value-self.margin
        if self.hidden is not None:
            hidden = self.hidden.minimum(start, end)
            if hidden is not None:
                hidden += self.offset
                value = hidden if value is None else min(value, hidden)
        return value

    def __getattr__(self, name):
        return getattr(self.inner, name)


class UnknownCells:
    """The pessimistic-unknown rule's state (start, traversed track, look-around exemption) for one mission."""

    def __init__(self, bound_m, variant):
        if variant not in VARIANTS:
            raise ValueError(f'pessimistic-unknown variant must be one of {VARIANTS}')
        self.bound_m, self.variant = bound_m, variant
        self.radius = BODY_REACH_M+bound_m
        self.start, self.track = None, []
        self.lookaround = dict(ended=None, max_drift_m=0., seen=False)

    def context(self, snapshot, position, route_status):
        """(hidden clearance, offset) for a plan at `position` (map xy), or None (nothing unknown, or exempt)."""
        p = tuple(float(v) for v in np.asarray(position, float)[:2])
        if self.start is None:
            self.start = p
        if not self.track or math.dist(p, self.track[-1]) > .005:
            self.track.append(p)
        if self.variant == 'revised':
            fine = revised_unknown_cells(snapshot, p, self.start, self.track)
        else:
            fine = seeded_unknown_cells(snapshot, p, self.start, self.track)
        context = (grid.FineClearance(fine), CHECK_REQUIRED_M-self.radius) if fine else None
        if self.variant == 'lookaround' and self._exempt(p, route_status):
            return None
        return context

    def _exempt(self, p, route_status):
        state = self.lookaround
        drift = math.dist(p, self.start)
        state['max_drift_m'] = max(state['max_drift_m'], drift)
        inside = route_status == LOOKAROUND_STATUS
        if state['ended'] is None:
            if inside:
                state['seen'] = True
                if drift > LOOKAROUND_DRIFT_M:
                    state['ended'] = 'centre drift beyond limit'
            elif state['seen']:
                state['ended'] = 'look-around complete'
        return inside and state['ended'] is None


def _scope(snapshot, p, track):
    """Coarse keys within scope of p that are not near the track, and their centres."""
    lo, hi = np.floor((np.asarray(p)-SCOPE_M)/grid.CELL_M).astype(int), np.floor((np.asarray(p)+SCOPE_M)/grid.CELL_M).astype(int)
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1), indexing='ij')
    keys = np.stack((ii.ravel(), jj.ravel()), axis=1)
    keys = keys[np.all((keys >= -grid.HALF_CELLS)&(keys < grid.HALF_CELLS), axis=1)]
    centres = (keys+.5)*grid.CELL_M
    candidate = np.linalg.norm(centres-np.asarray(p), axis=1) <= SCOPE_M
    return keys, centres, candidate


def revised_unknown_cells(snapshot, p, start, track):
    keys, centres, candidate = _scope(snapshot, p, track)
    candidate &= np.linalg.norm(centres-np.asarray(start), axis=1) > START_REACH_M
    track = np.asarray(track, float).reshape(-1, 2)
    if len(track):
        candidate &= np.min(np.linalg.norm(centres[:, None]-track[None], axis=2), axis=1) > TRACK_RADIUS_M
    observed = snapshot.floor | snapshot.occupied
    unknown = [tuple(map(int, k)) for k in keys[candidate] if tuple(map(int, k)) not in observed]
    return frozenset((5*a+u, 5*b+v) for a, b in unknown for u in range(5) for v in range(5))


def seeded_unknown_cells(snapshot, p, start, track):
    keys, centres, candidate = _scope(snapshot, p, track)
    track = np.asarray(track, float).reshape(-1, 2)
    if len(track):
        candidate &= np.min(np.linalg.norm(centres[:, None]-track[None], axis=2), axis=1) > TRACK_RADIUS_M
    observed = snapshot.floor | snapshot.occupied
    unknown = np.array([k for k in keys[candidate] if (int(k[0]), int(k[1])) not in observed], int).reshape(-1, 2)
    if not len(unknown):
        return frozenset()
    fine = (unknown[:, None, :]*5+_OFFSETS[None]).reshape(-1, 2)
    low = fine*grid.FINE_CELL_M-np.asarray(start, float)[:2]
    far = np.hypot(np.maximum(np.abs(low[:, 0]), np.abs(low[:, 0]+grid.FINE_CELL_M)),
                   np.maximum(np.abs(low[:, 1]), np.abs(low[:, 1]+grid.FINE_CELL_M)))
    return frozenset(map(tuple, fine[far > PRECONDITION_CLEAR_M].tolist()))
