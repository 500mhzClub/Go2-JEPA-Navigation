"""Per-mission measures beyond round trips (docs/dynamic/PLAN.md P13 item 5), read from a run's native traces.

- `path_m`: the base's path over the floor, from the physics trace sampled at 10 Hz (steps under 1 mm are dropped).
- `simulated_s`: the mission's simulated length.
- `outbound_s`, `return_s`: each leg's time when it succeeded (from the evaluation), else None.
- `stand_off_s`: seconds in which a walker waits on the robot: it has stood still (moved under 1 mm per 100 ms) for at
  least `STILL_S`, within `STAND_OFF_M` of the base, with the robot ahead of the walker's last heading (the truth, as
  `outputs/vacate/standoff.py` measured for P11).
- `contact_stop`: what a disallowed contact touched, 'walker' or 'wall', or None.

Per cohort, `contacts_per_km` is the contact stops over the summed path.
"""
import json
from pathlib import Path

import numpy as np

SAMPLE_S, STILL_S, STAND_OFF_M, STILL_M = .1, 1., .70, 1e-3


def resampled(times, values, step=SAMPLE_S):
    """The rows of `values` at the trace time nearest each `step` from the first."""
    times = np.asarray(times, float)
    if not len(times):
        return np.zeros(0), np.zeros((0,)+np.asarray(values).shape[1:])
    grid = np.arange(times[0], times[-1]+1e-9, step)
    index = np.minimum(np.searchsorted(times, grid), len(times)-1)
    return grid, np.asarray(values)[index]


def stand_off_seconds(times, base_xy, walkers_xy):
    """Seconds (on the `SAMPLE_S` grid) in which some walker waits on the robot (see the module docstring)."""
    base_xy, walkers_xy = np.asarray(base_xy, float), np.asarray(walkers_xy, float).reshape(len(base_xy), -1, 2)
    n, count = walkers_xy.shape[:2]
    waiting = np.zeros(n, bool)
    need = int(round(STILL_S/SAMPLE_S))
    for j in range(count):
        heading, still = np.zeros(2), 0
        for t in range(1, n):
            d = walkers_xy[t, j]-walkers_xy[t-1, j]
            if np.linalg.norm(d) > STILL_M:
                heading, still = d/np.linalg.norm(d), 0
            else:
                still += 1
            r = base_xy[t]-walkers_xy[t, j]
            waiting[t] |= still >= need and np.linalg.norm(r) < STAND_OFF_M and heading@r > 0
    return float(waiting.sum()*SAMPLE_S)


def mission_metrics(run):
    """The measures of one run folder (see the module docstring); {} when its traces are missing."""
    run = Path(run)
    physics, walkers = run/'native/physics_trace.npz', run/'native/obstacle_trace.npz'
    if not physics.exists():
        return {}
    trace = np.load(physics)
    times, base = resampled(trace['timestamp_s'], trace['base_pose_world'][:, :2])
    steps = np.linalg.norm(np.diff(base, axis=0), axis=1)
    out = dict(path_m=round(float(steps[steps >= 1e-3].sum()), 3),
               simulated_s=round(float(times[-1]-times[0]) if len(times) else 0., 2))
    evaluation = run/'episode_evaluation.json'
    if evaluation.exists():
        ev = json.loads(evaluation.read_text())
        leg = lambda record: round(float(record['elapsed_s']), 2) if record.get('success') else None
        out.update(outbound_s=leg(ev['outbound']), return_s=leg(ev['return_leg']))
    out['stand_off_s'] = 0.
    if walkers.exists():
        w = np.load(walkers)
        if len(w['timestamp_s']) and w['xy'].shape[1]:
            index = np.minimum(np.searchsorted(w['timestamp_s'], times), len(w['timestamp_s'])-1)
            out['stand_off_s'] = round(stand_off_seconds(times, base, w['xy'][index]), 1)
    events = run/'native/contact_events.json'
    out['contact_stop'] = None
    if events.exists():
        hit = next((e for e in json.loads(events.read_text()) if e['disallowed_contacts']), None)
        if hit is not None:
            touched = hit['disallowed_contacts'][0]['environment_object_id']
            out['contact_stop'] = 'walker' if touched.startswith('obstacle') else 'wall'
    return out


def contacts_per_km(rows):
    """Contact stops over the summed path of the rows that have one (None without any path)."""
    paths = [r['path_m'] for r in rows if r.get('path_m') is not None]
    if not paths or sum(paths) <= 0:
        return None
    return round(sum(1 for r in rows if r.get('contact_stop'))/(sum(paths)/1000.), 3)
