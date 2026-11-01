"""The hand tracker and the world model's walkers against the true walkers, plan by plan (docs/dynamic/PLAN.md, P6).

    python -m go2nav.eval.walker_compare COHORT_DIR [COHORT_DIR ...] [--out REPORT.json] [--workers N]

Each plan of a mission's log is compared with the simulator's records, in the robot's body frame: the truth from the true
pose, a track from the estimated one, so a track's error includes the pose error the planner lives with.

- **In reach:** a walker within `REACH_M` of the robot, in the primary camera's view, its centre not hidden by the walls.
- **Found:** a track within `MATCH_M` of a walker in reach, each track used once, nearest first. Tracks are the planner's
  confirmed ones (`hand`, `model`).
- **False alarm:** a confirmed track within `REACH_M` with no walker at all within `MATCH_M`; per 100 plans.
- **Position error:** median, found walkers. **Velocity error:** median, found walkers moving faster than 0.05 m/s.

The model's tracks are in `walker_shadow.json` (shadow missions). Passes (set before scoring): `model` finds at least
`FOUND_SHARE` of what `hand` finds, with no more false alarms, and a position error at most `ERROR_RATIO` times `hand`'s.
Intervals are 1,000 bootstrap draws of missions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np

from go2nav.eval.mover_probe import CAMERA_X_M, HALF_FOV_RAD
from go2nav.sim.obstacles import RADIUS_M

REACH_M, MATCH_M = 2., .5
FOUND_SHARE, ERROR_RATIO = .9, 1.25
MOVING_M_S = .05
SYSTEMS = ('hand', 'model')


def planar_yaw(pose):
    """Yaw of (x, y, z, qx, qy, qz, qw) poses (..., 7)."""
    q = np.asarray(pose)
    return np.arctan2(2*(q[..., 6]*q[..., 5]+q[..., 3]*q[..., 4]), 1-2*(q[..., 4]**2+q[..., 5]**2))


def body(xy, position, heading):
    """Points (..., 2) in the frame of a robot at `position` with `heading`."""
    c, s = math.cos(heading), math.sin(heading)
    d = np.asarray(xy, float)-position
    return np.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]), -1)


def turn(v, heading):
    """Vectors (..., 2) into a frame turned by `heading`."""
    return body(v, np.zeros(2), heading)


def inside_walls(points, walls):
    """Whether each point (n, 2) lies in a wall box."""
    hit = np.zeros(len(points), bool)
    for w in walls:
        c, s = math.cos(w['yaw_rad']), math.sin(w['yaw_rad'])
        d = points-np.asarray(w['centre_xyz'][:2])
        hit |= (np.abs(c*d[:, 0]+s*d[:, 1]) <= w['size_xyz'][0]/2)&(np.abs(-s*d[:, 0]+c*d[:, 1]) <= w['size_xyz'][1]/2)
    return hit


def in_reach(robot, heading, walker, walls):
    """See the module docstring; `robot` and `walker` in the world."""
    x, y = body(walker, robot, heading)
    if math.hypot(x, y) > REACH_M:
        return False
    x -= CAMERA_X_M
    distance = math.hypot(x, y)
    if distance < .2:
        return False
    if abs(math.atan2(y, x)) > HALF_FOV_RAD+math.asin(min(1., RADIUS_M/distance)):
        return False
    steps = np.arange(.05, max(distance-.2, .05), .05)/distance
    return not inside_walls(robot+steps[:, None]*(walker-robot), walls).any()


def match(tracks, truth):
    """Greedy nearest pairs (track, walker) within `MATCH_M`, from (n, 2) and (m, 2)."""
    if not len(tracks) or not len(truth):
        return []
    gaps = np.linalg.norm(tracks[:, None]-truth[None], axis=2)
    pairs, used, taken = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(gaps, axis=None), gaps.shape)):
        if gaps[i, j] > MATCH_M:
            break
        if i not in used and j not in taken:
            used.add(int(i))
            taken.add(int(j))
            pairs.append((int(i), int(j)))
    return pairs


def plan_rows(run):
    """The plans of a mission: (frame, measured_ns, hand tracks), and the model's tracks by frame."""
    rows = [p for p in json.loads((run/'planning.json').read_text()) if 'selection' in p]
    shadow = {}
    path = run/'walker_shadow.json'
    if path.exists():
        shadow = {s['frame']: s['tracks'] for s in json.loads(path.read_text())}
    return rows, shadow, path.exists()


def map_turn(run, rows):
    """The map's yaw from the initial-body frame, from the outbound goal in both frames."""
    goal = json.loads((run/'episode.json').read_text())['mission']['goal_initial_body_xy_m']
    mapped = next(r['route']['goal_map_xy_m'] for r in rows if 'goal_map_xy_m' in r['route'])
    return math.atan2(mapped[1], mapped[0])-math.atan2(goal[1], goal[0])


def mission(run):
    """Per system: walkers in reach, found, false alarms, position and velocity errors, plans."""
    run = Path(run)
    rows, shadow, has_model = plan_rows(run)
    out = {s: dict(reach=0, found=0, false=0, errors=[], speed_errors=[]) for s in SYSTEMS}
    out['plans'] = len(rows)
    out['has_model'] = has_model
    if not rows:
        return out
    trace = np.load(run/'native/physics_trace.npz')
    times, pose = trace['timestamp_s'], trace['base_pose_world']
    obstacle = run/'native/obstacle_trace.npz'
    walker_t, walker_xy = (np.zeros(0), np.zeros((0, 0, 2)))
    if obstacle.exists():
        with np.load(obstacle) as o:
            walker_t, walker_xy = o['timestamp_s'], o['xy']
    walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
    registered = {p['frame']: p['registered_pose'] for p in json.loads((run/'poses.json').read_text())}
    theta = map_turn(run, rows)
    c, s = math.cos(theta), math.sin(theta)
    for row in rows:
        estimate = registered.get(row['frame'])
        if estimate is None:
            continue
        now = row['measured_ns']/1e9
        i = min(np.searchsorted(times, now), len(times)-1)
        robot, heading = pose[i, :2], float(planar_yaw(pose[i]))
        truth = np.zeros((0, 2))
        velocity = np.zeros((0, 2))
        if len(walker_t):
            k = min(np.searchsorted(walker_t, now), len(walker_t)-1)
            before = min(np.searchsorted(walker_t, now-.4), len(walker_t)-1)
            truth = walker_xy[k]
            velocity = (walker_xy[k]-walker_xy[before])/max(walker_t[k]-walker_t[before], 1e-3)
        seen = [j for j, w in enumerate(truth) if in_reach(robot, heading, w, walls)]
        truth_body = body(truth, robot, heading) if len(truth) else truth
        p, R = np.asarray(estimate['position'], float), np.asarray(estimate['rotation'], float)
        q = np.array([c*p[0]-s*p[1], s*p[0]+c*p[1]])
        map_heading = math.atan2(R[1, 0], R[0, 0])+theta
        tracks = dict(hand=[t for t in row['selection'].get('obstacle_tracks') or [] if t['motion_updates'] >= 2],
                      model=[t for t in shadow.get(row['frame']) or [] if t.get('confirmed')])
        for name in SYSTEMS:
            if name == 'model' and not has_model:
                continue
            found = np.array([t['centroid_m'] for t in tracks[name]], float).reshape(-1, 2)
            found_body = body(found, q, map_heading)
            out[name]['reach'] += len(seen)
            pairs = match(found_body, truth_body[seen]) if seen else []
            out[name]['found'] += len(pairs)
            for a, b in pairs:
                out[name]['errors'].append(float(np.linalg.norm(found_body[a]-truth_body[seen[b]])))
                v = velocity[seen[b]]
                if np.linalg.norm(v) > MOVING_M_S:
                    estimated = turn(np.array(tracks[name][a]['velocity_m_s']), map_heading)
                    out[name]['speed_errors'].append(float(np.linalg.norm(estimated-turn(v, heading))))
            near = found_body[np.linalg.norm(found_body, axis=1) <= REACH_M]
            for f in near:
                if not len(truth) or np.linalg.norm(truth_body-f, axis=1).min() > MATCH_M:
                    out[name]['false'] += 1
    return out


def summary(missions, name, rng):
    used = [m[name] for m in missions if name == 'hand' or m['has_model']]
    plans = [m['plans'] for m in missions if name == 'hand' or m['has_model']]
    if not used:
        return None
    reach, found, false = (np.array([m[k] for m in used], float) for k in ('reach', 'found', 'false'))
    draws = rng.integers(0, len(used), (1000, len(used)))
    share = found[draws].sum(1)/np.maximum(reach[draws].sum(1), 1)
    errors = [e for m in used for e in m['errors']]
    speeds = [e for m in used for e in m['speed_errors']]
    return dict(missions=len(used), plans=int(sum(plans)), reach=int(reach.sum()), found=int(found.sum()),
                found_share=float(found.sum()/max(reach.sum(), 1)),
                found_share_95=[float(v) for v in np.quantile(share, [.025, .975])],
                false_per_100_plans=float(100*false.sum()/max(sum(plans), 1)),
                position_error_m=float(np.median(errors)) if errors else None,
                velocity_error_m_s=float(np.median(speeds)) if speeds else None)


def verdict(hand, model):
    if hand is None or model is None:
        return None
    checks = dict(found=model['found_share'] >= FOUND_SHARE*hand['found_share'],
                  false_alarms=model['false_per_100_plans'] <= hand['false_per_100_plans'],
                  position=model['position_error_m'] is not None
                  and model['position_error_m'] <= ERROR_RATIO*hand['position_error_m'])
    return checks | dict(passed=all(checks.values()))


def compare(cohorts, workers=8):
    runs = sorted(r for c in cohorts for r in (Path(c)/'runs').iterdir() if (r/'planning.json').exists())
    with ProcessPoolExecutor(workers) as pool:
        missions = list(pool.map(mission, map(str, runs), chunksize=2))
    rng = np.random.default_rng(0)
    hand, model = (summary(missions, s, rng) for s in SYSTEMS)
    return dict(hand=hand, model=model, verdict=verdict(hand, model))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cohorts', nargs='+')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    report = compare(a.cohorts, a.workers)
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
