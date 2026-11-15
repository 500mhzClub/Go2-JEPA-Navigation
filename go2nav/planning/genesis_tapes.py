"""P24 stage 2b (docs/dynamic/PLAN.md): P17's planner under the frozen rule in a Genesis mission, with M5 (or constant
velocity) for the walkers, in place of the forecaster's lattice (`controllers.controller`, `decision='tapes'`).

- **At each map update** (`bind`): the routing map becomes a layout (`perception.map_geometry.MapGeometry`) with its
  distance fields. The walker model is bound to it, for the tracker's detection and for M5's wall inputs, and a
  planner is made on it.
- **Every frame** (`observe`): the model's ring tracker reads the frame's range scan at the registered pose, in the
  map frame, and M5 records the frame's request.
- **A plan** (`plan`):
  - The robot's state when the plan's commands start: the pose carried through the committed prefix by the kinematic
    model (`planning.sampling`). Its path is within 2.4 cm of the Go2's at 3.2 s (STATUS P24 2a).
  - A look around (the route's `scan_error`) is a turn on the spot towards the heading, at most the lattice's
    0.45 rad/s.
  - Otherwise P17's tapes run to the route's waypoint, costed under the frozen rule (no maze cells, no sideways
    commands, the mission's arrival: `Planner(cells=False, lateral=False, arrive_m=...)`). The model's walkers are
    predicted from now, through the prefix and along each tape (`Prefixed`).
  - The hand harness's rule: a tape that would sweep floor not yet seen within the first 0.8 s is discarded (`uncovered`,
    the controller's `Planner.uncovered`, the coverage rule P18's cost kept). When the cheapest tape is lost to it,
    the route is asked for a view of that floor (`coverage`, returned to the controller). The hold never sweeps new
    floor, so a tape is always left.
  - The cheapest tape's first `COMMIT_TICKS` commands go to `planning.dispatch.TapeCommand`.
"""
import math
from pathlib import Path

import numpy as np

from go2nav.perception.map_geometry import MapGeometry
from go2nav.planning.actions import COMMIT_TICKS
from go2nav.planning.fields import Fields
from go2nav.planning.sampling import ARRIVE_M, FRAME_S, Planner, advance, integrate, limit, make_model, wrap

HORIZON_S, COVERAGE_TICKS, LOOK_RATE = 6.4, 8, .45
REPO = Path(__file__).resolve().parents[2]
ARRIVAL_SHARE = ARRIVE_M/.04          # the ring world's 0.03 m of its missions' 0.04-m arrival radius


class Prefixed:
    """A walker model seen from a plan whose tapes start after the committed prefix: each tape's path and requests are
    prefixed with the prefix's, the model predicts from its last observation, and the prefix's frames are dropped."""
    takes_tapes = True

    def __init__(self, model, xy, yaw, requests):
        self.model, self.xy, self.yaw, self.requests = model, xy, yaw, requests
        self.weights = self.groups = self.ends = None

    def walkers(self, snapshot, xy, yaw, frames, geometry=None, requested=None):
        count, n = xy.shape[0], len(self.xy)
        if not n:
            out = self._call(snapshot, xy, yaw, frames, geometry, requested)
            return out
        full_xy = np.concatenate((np.broadcast_to(self.xy, (count, n, 2)), xy), 1)
        full_yaw = np.concatenate((np.broadcast_to(self.yaw, (count, n)), yaw), 1)
        full_requested = None
        if requested is not None:
            full_requested = np.concatenate((np.broadcast_to(self.requests, (count, n, 3)), requested), 1)
        return self._call(snapshot, full_xy, full_yaw, np.asarray(frames)+n, geometry, full_requested)[:, n:]

    def _call(self, snapshot, xy, yaw, frames, geometry, requested):
        extra = dict(requested=requested) if getattr(self.model, 'takes_tapes', False) and requested is not None else {}
        out = self.model.walkers(snapshot, xy, yaw, frames, geometry, **extra)
        self.weights, self.groups = getattr(self.model, 'weights', None), getattr(self.model, 'groups', None)
        return out


class TapesDecision:
    def __init__(self, model, *, arrival_radius_m, seed=0):
        if model.startswith('objects:') and not Path(model[8:]).is_absolute():   # a path in the repository
            model = 'objects:'+str(REPO/model[8:])
        self.name, self.seed = model, seed
        self.model = make_model(model, seed)
        self.arrive_m = ARRIVAL_SHARE*arrival_radius_m
        self.geometry = self.fields = self.planner = None
        self.previous, self.target, self.plans, self.binds = None, None, 0, 0

    def bind(self, snapshot, xy):
        """A map update: the map as a layout, its fields, the model bound to them, and a planner on them."""
        self.geometry = MapGeometry(snapshot, xy)
        self.fields = Fields(self.geometry)
        self.model.bind(self.fields)
        self.planner = Planner(self.geometry, horizon_s=HORIZON_S, terminal='walkers', seed=self.seed*100_003+self.binds,
                               model=self.model, cells=False, lateral=False, arrive_m=self.arrive_m)
        if not self.planner.exact or self.planner.walker_price is None:
            raise RuntimeError('the tapes decision plans under the frozen rule (GO2NAV_EXACT, GO2NAV_RECOVER, '
                               'GO2NAV_WALKER_PRICE, GO2NAV_W_STILL)')
        self.binds += 1

    def observe(self, t, scan, xy, yaw, request):
        """A frame: the tracker on its range scan at the registered pose (map frame); the frame's request recorded."""
        if self.fields is None or scan is None:
            return
        self.model.observe(t, scan, np.asarray(xy, float), float(yaw), self.geometry)
        if hasattr(self.model, 'record'):
            self.model.record(np.asarray(request, np.float32))

    def plan(self, *, xy, yaw, executed, prefix, target, look=None, uncovered=None):
        """The commands (COMMIT_TICKS, 3) to dispatch after the prefix, and a record (see the module docstring).
        xy, yaw: the registered pose (map frame); executed: the last request sent; prefix: the committed commands of
        the next ticks; target: the route's waypoint (map frame); look: the heading to look to (or None); uncovered:
        `f(body_paths (K, COVERAGE_TICKS, 2), names)` -> per tape the unobserved cells it would sweep."""
        prefix = np.asarray(prefix, float).reshape(-1, 3)
        p_xy, p_yaw, ex = np.asarray(xy, float)[None], np.array([float(yaw)]), np.asarray(executed, float)[None]
        path_xy, path_yaw = [], []
        for c in prefix:
            ex = limit(c[None], ex)
            p_xy, p_yaw = advance(p_xy, p_yaw, ex)
            path_xy.append(p_xy[0].copy())
            path_yaw.append(float(p_yaw[0]))
        start = dict(xy=p_xy[0].copy(), yaw=float(p_yaw[0]), executed=ex[0].copy())
        if look is not None:
            error = wrap(float(look)-start['yaw'])
            rate = float(np.clip(error/(COMMIT_TICKS*FRAME_S), -LOOK_RATE, LOOK_RATE))
            return np.tile([0., 0., rate], (COMMIT_TICKS, 1)), dict(kind='look', heading_error=round(error, 3))
        planner, target = self.planner, np.asarray(target, float)
        planner.model = Prefixed(self.model, np.array(path_xy).reshape(-1, 2), np.array(path_yaw), prefix)
        planner.previous, planner.target = self.previous, self.target
        if planner.target is None or not np.array_equal(target, planner.target):
            planner.previous, planner.target = None, target.copy()
        requested, names = planner.proposals(start, target)
        total, terms = planner.costs(start, target, requested)
        cheapest, coverage, blocked = int(total.argmin()), None, np.zeros(len(names), bool)
        if uncovered is not None:
            ticks = COVERAGE_TICKS-len(prefix)
            tape_xy, _ = integrate(start, requested[:, :ticks])
            full = np.concatenate((np.broadcast_to(np.array(path_xy).reshape(-1, 2)[None], (len(names), len(prefix), 2)),
                                   tape_xy), 1)
            c, s, d = math.cos(yaw), math.sin(yaw), full-np.asarray(xy, float)
            body = np.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]), -1)
            swept = uncovered(body, names)
            blocked = np.array([bool(cells) for cells in swept])
            if blocked[cheapest]:
                coverage = swept[cheapest]
        k = int(np.where(blocked, np.inf, total).argmin())
        self.previous, self.target, self.plans = requested[k].copy(), planner.target, self.plans+1
        tracked = int(len(planner.model.model.tracker.state()[0])) if hasattr(self.model, 'tracker') else None
        record = dict(kind='tape', chosen=names[k], cost=round(float(total[k]), 3), cheapest=names[cheapest],
                      blocked=int(blocked.sum()), breached=bool(terms['breached'][k]), tracked=tracked,
                      **{key: round(float(terms[key][k]), 3) for key in ('time', 'clearance', 'progress')})
        return requested[k, :COMMIT_TICKS], dict(record, coverage=coverage)
