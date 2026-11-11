"""The decision by an intrinsic cost (docs/dynamic/PLAN.md P18): in a mission, the candidate commands' forecasts are
scored by a cost in seconds, in place of `Planner.select`'s stages.

`select` takes what `Planner.select` takes from the controller, less the map: the forecaster's prediction for each
candidate (N, 8, 5: x, y, sin yaw, cos yaw, contact logit, in the body frame at the frame; `planning.actions`: three
prefix ticks, the candidate's commit, then zero), the waypoint in the body frame (`Planner.objective`: the route's
next point, or zero with a `scan_error`, the heading to turn to), and the range sensor's returns at the frame as
body-frame points. The cost of a candidate, over its ticks after the prefix:
- **to go:** from the pose it ends at, the distance to the waypoint over `V_REF` plus the turn towards it over
  `W_REF`. At the mission's target (`target`: the waypoint is the beacon or home) the turn is the one that points the
  robot at the arrival disk, not at its centre, and none inside the disk: a robot whose line of travel already
  crosses the disk has no turn left to make. The distance counts to the centre, inside the disk too; the mission
  itself stops the robot there (`planning.mission`). Earlier the cost was zero for a candidate forecast to end inside
  the disk and the turn was to the centre. Both failed within a few centimetres, where the forecast's error is as
  large as the distances: a hold forecast to drift 1 cm into the disk was free, so the robot stood 2.3 cm from a
  2-cm disk, and the bearing of a point 3 cm off swung by 15 degrees on a forecast 7 mm to the side, so every move
  read worse than holding (6 of the 35 failures of 11 November; docs/dynamic/STATUS.md P18). At any other waypoint
  the turn is to the point itself however near it is: the first suites zeroed the cost there too, which made holding
  free on a frontier waypoint, where the route does not change while the robot holds (18 of 28 polite failures).
  With a `scan_error`, a look around from where the robot stands: the turn
  left to that heading over `W_REF`, plus the distance the candidate ends from where it started over `V_REF`. The
  first suites counted the turn alone, and the robot drove about while it looked: forecasts of an arc and of a turn
  on the spot differ a little in heading, and whichever read nearer won (round trips of 60-70 m in a 15-m maze);
  The turn was first weighted down within 0.3 m of the waypoint; it is now counted in full, one constant fewer;
- **a reverse pulse is not a candidate.** In the terminal approach a plan runs for 100 ms (`pulse`). The lattice's
  reverse, 0.10 m/s, is then a centimetre: the forecast credits it with motion and the robot does not make it. The
  smoke's one failure (10 November) was 1,043 such plans in a row, 7 cm behind and beside the beacon. This is a fact
  about the commands the robot has, not a rule about what to do;
- **clearance:** `CLEARANCE_RATE` seconds a second as the body outline's clearance to the returns falls from `SOFT_M`
  to `FLOOR_M`; under the floor the candidate is discarded. A robot already under the floor may still move wherever
  its clearance does not fall further;
- **floor not yet seen:** a candidate that would sweep floor the map has not observed is discarded (`uncovered`:
  `planning.selection.footprint_extension`, the hand harness's rule that a translation must not sweep unobserved
  floor). Like the reflex it is kept, and the cost looks ahead to it. When the cheapest candidate is lost to it, the
  record names it (`coverage_wanted`), and the controller asks the route for a view of that floor, as the hand
  planner's stage did; without this the route's frontiers were never looked at on the way home. The discard waits
  for a view to be asked for, not for one that cannot be had: it is lifted while a request is with the route, and
  floor the route finds no viewpoint for is not asked for again (`Planner.uncovered`);
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


def select(prediction, candidates, goal_body, scan_error, points, *, arrival_radius_m, target=True, pulse=False, cells=None,
           uncovered=None):
    """The candidate chosen by the cost: a selection record with `action`, `action_index`, `requested_command` and one
    row per candidate. `points`: the range returns (n, 2 or 3) in the body frame, or None; `cells`: the latest
    obstacle cells of the dispatch check (1-cm body cells), or None to leave the reflex out; `uncovered`: per
    candidate, whether it would sweep floor not yet observed, or None."""
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
        to_go = np.abs(wrap(scan_error-heading))/W_REF+np.linalg.norm(end, axis=1)/V_REF
    else:
        offset = np.asarray(goal_body, float)-end
        distance = np.linalg.norm(offset, axis=1)
        turn = np.abs(wrap(np.arctan2(offset[:, 1], offset[:, 0])-heading))
        reach = arrival_radius_m if target else 0.
        aim = np.arcsin(np.clip(reach/np.maximum(distance, 1e-9), 0., 1.))          # half the angle the disk subtends
        to_go = distance/V_REF+np.where(distance <= reach, 0., np.maximum(turn-aim, 0.))/W_REF
    stopped = np.array([cells is not None and not reflex_clear(a, cells, pulse) for a in candidates])
    absent = np.array([bool(pulse and command(a)[0] < 0) for a in candidates])     # a reverse pulse is no action
    unseen = np.zeros(count, bool) if uncovered is None else np.asarray(uncovered, bool)
    hold = candidates.index('hold')
    moved = np.array([float(np.hypot(*command(a)[:2])) for a in candidates])

    def cheapest(discarded):
        total = to_go+soft+DISCARD*discarded
        best = min(range(count), key=lambda i: (round(float(total[i]), 6), moved[i], abs(command(candidates[i])[2])))
        return (hold if discarded.all() else best), total
    wanted, _ = cheapest(breached | stopped | absent)
    discarded = breached | stopped | absent | unseen
    best, total = cheapest(discarded)
    rows = [dict(action=a, utility_m=-float(total[i]), cost_s=float(total[i]), to_go_s=float(to_go[i]),
                 clearance_s=float(soft[i]), lowest_clearance_m=float(clearance[i].min()), breached=bool(breached[i]),
                 reflex_would_stop=bool(stopped[i]), sweeps_unobserved=bool(unseen[i]),
                 eligible=bool(not discarded[i])) for i, a in enumerate(candidates)]
    return dict(action=candidates[best], action_index=best, requested_command=command(candidates[best]), candidates=rows,
                waypoint_body_xy_m=np.asarray(goal_body, float).tolist(), clearance_now_m=None if np.isinf(now) else now,
                all_discarded=bool(discarded.all()),
                coverage_wanted=candidates[wanted] if unseen[wanted] else None, selection_objective='intrinsic_cost_seconds')
