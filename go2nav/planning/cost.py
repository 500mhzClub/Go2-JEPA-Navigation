"""The decision by an intrinsic cost (docs/dynamic/PLAN.md P18): in a mission, the candidate commands' forecasts are
scored by a cost in seconds, in place of `Planner.select`'s stages.

`select` takes what `Planner.select` takes from the controller, less the map: the forecaster's prediction for each
candidate (N, 8, 5: x, y, sin yaw, cos yaw, contact logit, in the body frame at the frame; `planning.actions`: three
prefix ticks, the candidate's commit, then zero), the waypoint in the body frame (`Planner.objective`: the route's
next point, or zero with a `scan_error`, the heading to turn to), and the range sensor's returns at the frame as
body-frame points. The cost of a candidate, over its ticks after the prefix:
- **to go:** from the pose it ends at, the distance to the waypoint over `V_REF` plus the turn towards it over
  `W_REF`; nothing within the arrival radius of the mission's target (`target`: the waypoint is the beacon or home).
  At any other waypoint the distance and turn count however near it is: the first suites zeroed them there too, which
  made holding free on a frontier waypoint, where the route does not change while the robot holds (18 of 28 polite
  failures; docs/dynamic/STATUS.md P18). With a `scan_error`: the turn left to that heading over `W_REF`.
  The turn was first weighted down within 0.3 m of the waypoint; it is now counted in full, one constant fewer;
- **a reverse pulse is not a candidate.** In the terminal approach a plan runs for 100 ms (`pulse`). The lattice's
  reverse, 0.10 m/s, is then a centimetre: the forecast credits it with motion and the robot does not make it. The
  smoke's one failure (10 November) was 1,043 such plans in a row, 7 cm behind and beside the beacon. This is a fact
  about the commands the robot has, not a rule about what to do;
- **clearance:** `CLEARANCE_RATE` seconds a second as the body outline's clearance to the returns falls from `SOFT_M`
  to `FLOOR_M`; under the floor the candidate is discarded. A robot already under the floor may still move wherever
  its clearance does not fall further;
- **the reflex:** a candidate the dispatch check would stop is discarded too (`planning.dispatch`: the 0.45-m disk
  swept by the command over its commit, the returns' age and the stopping allowance must be clear of the latest
  obstacle cells). The reflex is kept by decision; the cost only looks ahead to it, so that it does not choose what
  will be vetoed.
The cheapest candidate is chosen; of equal ones, the one that moves the base least, then turns least (so a scan turns
on the spot and a tie with hold holds). If every candidate is discarded, hold.

Five constants: `V_REF`, `W_REF`, `FLOOR_M`, `SOFT_M`, `CLEARANCE_RATE`.
"""
import numpy as np

from go2nav.planning import dispatch
from go2nav.planning.actions import COMMIT_TICKS, PREFIX_TICKS, command
from go2nav.sim.kinematic import rectangle_gap

V_REF, W_REF = .3, .45
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


def select(prediction, candidates, goal_body, scan_error, points, *, arrival_radius_m, target=True, pulse=False, cells=None):
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
        to_go = np.where((distance > arrival_radius_m) | (not target), distance/V_REF+turn/W_REF, 0.)
    stopped = np.array([cells is not None and not reflex_clear(a, cells, pulse) for a in candidates])
    absent = np.array([bool(pulse and command(a)[0] < 0) for a in candidates])     # a reverse pulse is no action
    discarded = breached | stopped | absent
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
