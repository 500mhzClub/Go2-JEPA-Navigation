"""The decision by an intrinsic cost (docs/dynamic/PLAN.md P18): in a mission, the candidate commands' forecasts are
scored by a cost in seconds, in place of `Planner.select`'s stages.

`select` takes what `Planner.select` takes from the controller, less the map: the forecaster's prediction for each
candidate (N, 8, 5: x, y, sin yaw, cos yaw, contact logit, in the body frame at the frame; `planning.actions`: three
prefix ticks, the candidate's commit, then zero), the waypoint in the body frame (`Planner.objective`: the route's
next point, or zero with a `scan_error`, the heading to turn to), and the range sensor's returns at the frame as
body-frame points. The cost of a candidate, over its ticks after the prefix:
- **to go:** from the pose it ends at, the distance to the waypoint over `V_REF` plus the turn towards it over
  `W_REF` (the turn counted in full beyond `NEAR_M` from the waypoint and in proportion nearer, so that a robot beside
  its target turns to it and one on it stands still); nothing within the arrival radius. With a `scan_error`: the turn
  left to that heading over `W_REF`;
- **clearance:** `CLEARANCE_RATE` seconds a second as the body outline's clearance to the returns falls from `SOFT_M`
  to `FLOOR_M`; under the floor the candidate is discarded. A robot already under the floor may still move wherever
  its clearance does not fall further;
- **the reflex:** a candidate the dispatch check would stop is discarded too (`planning.dispatch`: the 0.45-m disk
  swept by the command over its commit, the returns' age and the stopping allowance must be clear of the latest
  obstacle cells). The reflex is kept by decision; the cost only looks ahead to it, so that it does not choose what
  will be vetoed.
The cheapest candidate is chosen; of equal ones, the one that moves the base least, then turns least (so a scan turns
on the spot and a tie with hold holds). If every candidate is discarded, hold.

Six constants: `V_REF`, `W_REF`, `NEAR_M`, `FLOOR_M`, `SOFT_M`, `CLEARANCE_RATE`.
"""
import numpy as np

from go2nav.planning import dispatch
from go2nav.planning.actions import COMMIT_TICKS, PREFIX_TICKS, command
from go2nav.sim.kinematic import rectangle_gap

V_REF, W_REF, NEAR_M = .3, .45, .3
FLOOR_M, SOFT_M, CLEARANCE_RATE = .10, .25, 2.
DISCARD, TICK_S, RETURNS_AGE_S = 1e4, .1, .1


def wrap(angle):
    return (np.asarray(angle)+np.pi) % (2*np.pi)-np.pi


def reflex_clear(action, cells, pulse):
    """Whether the dispatch check would let a candidate's command run for its commit (see the module docstring)."""
    vx, vy, _ = command(action)
    seconds = (1 if pulse else COMMIT_TICKS)*TICK_S+RETURNS_AGE_S
    if not dispatch._connector_clear(np.array([vx, vy])*seconds, cells):
        return False
    if vx or vy:
        return bool(dispatch._connector_clear(np.array([vx, vy])*(seconds+dispatch.STOPPING_ALLOWANCE_S), cells))
    return True


def select(prediction, candidates, goal_body, scan_error, points, *, arrival_radius_m, pulse=False, cells=None):
    """The candidate chosen by the cost: a selection record with `action`, `action_index`, `requested_command` and one
    row per candidate. `points`: the range returns (n, 2 or 3) in the body frame, or None; `cells`: the latest
    obstacle cells of the dispatch check (1-cm body cells), or None to leave the reflex out."""
    p = np.asarray(prediction, float)
    xy, yaw = p[:, PREFIX_TICKS:, :2], np.arctan2(p[:, PREFIX_TICKS:, 2], p[:, PREFIX_TICKS:, 3])
    count = len(candidates)
    clearance = np.full(xy.shape[:2], np.inf)
    now = np.inf
    if points is not None and len(points):
        returns = np.asarray(points, float)[:, :2]
        clearance = rectangle_gap(returns[None, None], xy[:, :, None], yaw[:, :, None]).min(-1)
        now = float(rectangle_gap(returns, np.zeros(2), 0.).min())
    floor = min(FLOOR_M, now-.005)
    breached = (clearance < floor).any(1)
    soft = np.clip((SOFT_M-clearance)/(SOFT_M-FLOOR_M), 0., 1.).sum(1)*TICK_S*CLEARANCE_RATE
    end, heading = xy[:, -1], yaw[:, -1]
    if scan_error is not None:
        to_go = np.abs(wrap(scan_error-heading))/W_REF
    else:
        offset = np.asarray(goal_body, float)-end
        distance = np.linalg.norm(offset, axis=1)
        turn = np.abs(wrap(np.arctan2(offset[:, 1], offset[:, 0])-heading))
        to_go = np.where(distance > arrival_radius_m, distance/V_REF+np.minimum(1., distance/NEAR_M)*turn/W_REF, 0.)
    stopped = np.array([cells is not None and not reflex_clear(a, cells, pulse) for a in candidates])
    discarded = breached | stopped
    total = to_go+soft+DISCARD*discarded
    hold = candidates.index('hold')
    moved = np.array([float(np.hypot(*command(a)[:2])) for a in candidates])
    best = min(range(count), key=lambda i: (round(float(total[i]), 6), moved[i], abs(command(candidates[i])[2])))
    if discarded.all():
        best = hold
    rows = [dict(action=a, utility_m=-float(total[i]), cost_s=float(total[i]), to_go_s=float(to_go[i]),
                 clearance_s=float(soft[i]), lowest_clearance_m=float(clearance[i].min()), breached=bool(breached[i]),
                 reflex_would_stop=bool(stopped[i]), eligible=bool(not discarded[i])) for i, a in enumerate(candidates)]
    return dict(action=candidates[best], action_index=best, requested_command=command(candidates[best]), candidates=rows,
                waypoint_body_xy_m=np.asarray(goal_body, float).tolist(), clearance_now_m=None if np.isinf(now) else now,
                all_discarded=bool(discarded.all()), selection_objective='intrinsic_cost_seconds')
