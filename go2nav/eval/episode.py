"""Evaluate one mission run: did the robot reach the beacon and come home, how safely, and how efficiently.

    python -m go2nav.eval.episode RUN_DIR

Everything here uses the simulator's ground truth, which the controllers never see:
- **Arrival** (per leg, at the controller's own arrival event): over the 1-s dwell before it the true position stayed
  within 4 cm of the registered world target (beacon, then home), the 100-ms boundary speeds were at most 5 cm/s, and
  every request was zero. A round trip succeeds when both legs pass.
- **Safety**: disallowed contacts (`go2nav.sim.contacts`), and the clearance of the robot's collision primitives from
  the true walls at every physics step (`geometry.WallClearance`): a confirmed violation is an achieved distance
  below the threshold (hard 5 mm, operating 20 mm) or a contact; a sample is unresolved when only its lower bound is.
- **Efficiency**: per leg, the travelled path and SPL (shortest path / max(travelled, shortest) when successful).
- **Moving obstacles** (when the run has them, `go2nav.sim.obstacles`): the smallest distance from the base to an
  obstacle's surface, and each obstacle contact attributed to whichever body was closing on the other faster at the
  contact (the robot's base velocity or the obstacle's, along the line between them).
- **Holds and failures**: each plan that chose to hold is classified (an explicit rule, no eligible movement, movement
  out-scored, or not explained), and the episode's failures and stalls are tallied into a taxonomy.

The record is written to RUN_DIR/episode_evaluation.json. This is the original V4 reader with its C2 erratum (holds of
the reactive selector are classified by its own eligibility and command distances) and its failure reader (a mission
that failed before its first frame).
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

from go2nav.eval.geometry import WallClearance
from go2nav.planning.actions import ACTIONS

ARRIVAL_RADIUS_M, DWELL_FRAMES, QUIET_SPEED_M_S = .04, 10, .05
THRESHOLDS = (('hard', .005), ('operating', .02))
CHUNK = 400
STEP_BUDGET = 24_000


def load(run, name):
    return json.loads((Path(run)/name).read_text())


def world_target(episode, phase):
    return np.asarray(episode['beacon_xy_world'] if phase == 'OUTBOUND' else episode['home_se2_world'][:2], dtype=float)


def frames_with_samples(acquisitions, timestamps):
    """Each frame with the index of its physics sample (exact timestamp match)."""
    stamps = np.rint(timestamps*1e9).astype(np.int64)
    rows = []
    for row in acquisitions:
        index = int(np.searchsorted(stamps, row['measured_ns']))
        if stamps[index] != row['measured_ns']:
            raise ValueError('frame time without a physics sample')
        rows.append(row | dict(physical_sample_index=index))
    return rows


def arrivals(episode, mission, frames, trace, requests):
    """The physical check of each arrival the controller declared (see the module docstring)."""
    lookup = {r['frame']: r for r in frames}
    commands = {r['simulator_ns']: r['requested_command'] for r in requests}
    physics = trace['base_pose_world']
    rows = []
    for arrival in mission[-1]['arrivals']:
        end = arrival['frame']
        a, b = lookup[end-DWELL_FRAMES], lookup[end]
        target = world_target(episode, arrival['phase'])
        distance = np.linalg.norm(physics[a['physical_sample_index']:b['physical_sample_index']+1, :2]-target, axis=1)
        boundaries = physics[[lookup[i]['physical_sample_index'] for i in range(end-DWELL_FRAMES, end+1)], :3]
        speed = np.linalg.norm(np.diff(boundaries, axis=0), axis=1)/.1
        zero = all(t in commands and np.array_equal(commands[t], [0., 0., 0.]) for t in range(a['measured_ns'], b['measured_ns'], 20_000_000))
        passed = bool(np.all(distance <= ARRIVAL_RADIUS_M) and np.all(speed <= QUIET_SPEED_M_S) and zero)
        rows.append(dict(phase=arrival['phase'], frame=end, target_world_xy_m=target.tolist(), physical_radius_m=ARRIVAL_RADIUS_M,
                         dwell_seconds=1., native_minimum_distance_m=float(distance.min()),
                         native_maximum_distance_m=float(distance.max()), native_final_distance_m=float(distance[-1]),
                         native_maximum_100ms_speed_m_s=float(speed.max()), all_requested_intervals_zero=zero,
                         physical_distance_passed=bool(np.all(distance <= ARRIVAL_RADIUS_M)),
                         measured_motion_quiet=bool(np.all(speed <= QUIET_SPEED_M_S)), passed=passed))
    return rows


# -- hold classification ----------------------------------------------------------------------------------------------

def classify_hold(row):
    """Why a plan held (see the module docstring); reactive (C2) plans by their own eligibility."""
    s = row['selection']
    if 'scan_utilities' not in s and s.get('candidates') and not all('utility_m' in c for c in s['candidates']):
        return _classify_reactive_hold(row)
    checks = {r['action']: r for r in s.get('memory_forecast_candidates', [])}
    utilities = {r['action']: r['utility_m'] for r in s.get('scan_utilities', s.get('candidates', []))}
    stopping = s.get('planned_stopping_projection', {})
    stop_checks = {r['action']: r['projection_clear'] for r in stopping.get('candidates', [])}
    result = dict(frame=row['frame'], route_status=row.get('route_status'), category='insufficient_evidence',
                  observation_action_space_exclusions=[a for a in ACTIONS[1:] if a not in utilities],
                  motion_clearance_exclusions=[a for a in ACTIONS[1:] if a in utilities and a in checks
                                               and not checks[a].get('nominal_predicted_path_clear', False)],
                  stopping_projection_exclusions=[a for a in ACTIONS[1:] if a in utilities
                                                  and checks.get(a, {}).get('nominal_predicted_path_clear', False)
                                                  and stop_checks.get(a) is False],
                  clearance_modes={a: r.get('clearance_check_mode') for a, r in checks.items()}, override_reason=None)
    if set(checks) != set(ACTIONS) or not utilities or any(not math.isfinite(v) for v in utilities.values()):
        result['reason'] = 'missing candidate gates or finite recorded utilities'
    else:
        eligible = [a for a in ACTIONS if a in utilities and checks[a].get('nominal_predicted_path_clear', False)
                    and stop_checks.get(a, True)]
        moving = [a for a in eligible if a != 'hold']
        result['eligible_movement'] = moving
        turn = s.get('clearance_turn') or {}
        if stopping.get('changed') and stopping.get('before_action') != 'hold' and s['action'] == 'hold':
            result.update(category='explicit_override', override_reason='PLANNED_STOPPING_PROJECTION')
        elif turn.get('active') and turn.get('latched_turn_forecast_clear') is False:
            result.update(category='explicit_override', override_reason='LATCHED_RECOVERY_TURN_BLOCKED')
        elif not moving:
            result['category'] = 'no_eligible_movement'
        elif 'hold' in eligible and all(utilities[a] <= utilities['hold'] for a in moving):
            result.update(category='movement_lost_recorded_score_or_tie',
                          exact_tie_with_movement=any(utilities[a] == utilities['hold'] for a in moving))
        else:
            result['reason'] = 'eligible movement beats hold or hold eligibility missing; no explicit override explains it'
        result['no_eligible_movement_even_if_override'] = not moving
    result['inherited_stage_a_category'] = result['category']
    coverage, arrival = s.get('translation_footprint_coverage', {}), s.get('predictive_arrival_hold', {})
    if (s['action'] == 'hold' and coverage.get('rejected') and coverage.get('selected_action') == 'hold'
            and coverage.get('previous_action') in ('forward', 'left_arc', 'right_arc')):
        result.update(category='explicit_override', override_reason='TRANSLATION_FOOTPRINT_COVERAGE')
    elif s['action'] == 'hold' and arrival.get('changed') and arrival.get('selected_action') == 'hold' and arrival.get('eligible'):
        result.update(category='explicit_override', override_reason='PREDICTIVE_ARRIVAL_HOLD')
    return result


def _classify_reactive_hold(row):
    s = row['selection']
    candidates = {c['action']: c for c in s['candidates']}
    eligible = [a for a in ACTIONS if a in candidates and candidates[a].get('eligible')]
    moving = [a for a in eligible if a != 'hold']
    result = dict(frame=row['frame'], route_status=row.get('route_status'), category='insufficient_evidence',
                  observation_action_space_exclusions=[], motion_clearance_exclusions=[], stopping_projection_exclusions=[],
                  clearance_modes={}, override_reason=None, eligible_movement=moving, reactive_selector=True,
                  no_eligible_movement_even_if_override=not moving)
    distances = {a: candidates[a].get('normalized_command_distance_squared') for a in eligible}
    terminal = s.get('heading_first_terminal', {})
    if s['action'] == 'hold' and terminal.get('changed') and terminal.get('selected_action') == 'hold':
        result.update(category='explicit_override', override_reason='HEADING_FIRST_TERMINAL_HOLD')
    elif set(candidates) != set(ACTIONS) or any(d is None or not math.isfinite(d) for d in distances.values()):
        result['reason'] = 'reactive candidates incomplete or non-finite command distances'
    elif not moving:
        result['category'] = 'no_eligible_movement'
    elif 'hold' in eligible and all(distances['hold'] <= distances[a] for a in moving):
        result.update(category='movement_lost_recorded_score_or_tie',
                      exact_tie_with_movement=any(distances['hold'] == distances[a] for a in moving))
    result['inherited_stage_a_category'] = result['category']
    return result


OVERRIDE_LABELS = {'TRANSLATION_FOOTPRINT_COVERAGE': 'observation/view restriction (motion-dependent footprint)',
                   'PREDICTIVE_ARRIVAL_HOLD': 'planned arrival settling', 'HEADING_FIRST_TERMINAL_HOLD': 'planned arrival settling'}


# -- clearance --------------------------------------------------------------------------------------------------------

def clearance(spec, trace, first):
    """Per physics step from the first frame: (lower, upper) minimum over primitives, and the per-interval robust
    lower bound; computed in overlapping chunks of 400 steps as the original did."""
    reader = WallClearance(spec['geometry']['wall_boxes'])
    lows, highs, robust = [], [], []
    n = len(trace['timestamp_s'])
    for start in range(first, n-1, CHUNK):
        end = min(start+CHUNK+1, n)
        lower, upper, interval = reader.evaluate(trace['base_pose_world'][start:end], trace['joint_position'][start:end])
        low, high = lower.min(axis=1), upper.min(axis=1)
        if lows:
            low, high = low[1:], high[1:]
        lows.extend(low)
        highs.extend(high)
        robust.extend(interval.min(axis=1))
    return np.asarray(lows), np.asarray(highs), np.asarray(robust)


def safety(low, high, robust, contacts):
    out = {}
    for name, threshold in THRESHOLDS:
        violation = (high < threshold)|contacts
        unresolved = (low < threshold)&~violation
        out[name] = dict(threshold_m=threshold, confirmed_violation_samples=int(violation.sum()),
                         unresolved_sampled_samples=int(unresolved.sum()), all_native_samples_qualified=not unresolved.any(),
                         minimum_separation_lower_m=float(low.min()), minimum_separation_upper_m=float(high.min()),
                         interval_robustness_failure_count=int(np.count_nonzero(robust < threshold)))
    return out


# -- moving obstacles -------------------------------------------------------------------------------------------------

def obstacle_record(run, trace):
    """Closest approach and contact attribution for a run with moving obstacles (None without them)."""
    path = Path(run)/'native/obstacle_trace.npz'
    if not path.exists():
        return None
    with np.load(path) as archive:
        stamps, xy = archive['timestamp_s'], archive['xy']
    placement = load(run, 'config.json')['dynamics']['moving_obstacles']['placement']
    radius = np.array([o['radius_m'] for o in placement])
    if not len(stamps):
        return dict(count=len(placement), min_base_to_surface_m=None, contacts=[])
    index = np.clip(np.searchsorted(np.rint(trace['timestamp_s']*1e9), np.rint(stamps*1e9)), 0, len(trace['timestamp_s'])-1)
    base = trace['base_pose_world'][index, :2]
    gap = np.linalg.norm(xy-base[:, None], axis=2)-radius
    velocity = np.gradient(xy, stamps, axis=0) if len(stamps) > 1 else np.zeros_like(xy)
    contacts = []
    for event in load(run, 'native/contact_events.json'):
        hits = sorted({c['environment_object_id'] for c in event['disallowed_contacts']
                       if str(c['environment_object_id']).startswith('obstacle_')})
        for name in hits:
            k, t = int(name.split('_')[1]), event['timestamp_s']
            i = min(int(np.searchsorted(stamps, t)), len(stamps)-1)
            j = int(index[i])
            towards = xy[i, k]-trace['base_pose_world'][j, :2]
            towards = towards/max(np.linalg.norm(towards), 1e-9)
            robot_closing = float(trace['base_twist_world'][j, :2]@towards)
            obstacle_closing = float(-velocity[i, k]@towards)
            contacts.append(dict(obstacle=name, timestamp_s=t, robot_closing_m_s=robot_closing,
                                 obstacle_closing_m_s=obstacle_closing,
                                 mover='robot' if robot_closing >= obstacle_closing else 'obstacle'))
        if contacts:
            break   # the first contact stops the mission
    return dict(count=len(placement), min_base_to_surface_m=float(gap.min()),
                closest_obstacle=int(np.unravel_index(gap.argmin(), gap.shape)[1]), contacts=contacts)


# -- the record -------------------------------------------------------------------------------------------------------

def evaluate(run, *, write=True):
    run = Path(run)
    result, config = load(run, 'result.json'), load(run, 'config.json')
    if result['frames'] == 0:
        record = early_failure(run, result, config)
    else:
        record = full_evaluation(run, result, config)
    if write:
        (run/'episode_evaluation.json').write_text(json.dumps(record, allow_nan=False)+'\n')
    return record


def full_evaluation(run, result, config):
    started = time.monotonic()
    spec, episode = load(run, 'specification.json'), load(run, 'episode.json')
    planning, mission, requests = load(run, 'planning.json'), load(run, 'mission.json'), load(run, 'requests.json')
    with np.load(run/'native/physics_trace.npz', allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in ('base_pose_world', 'base_twist_world', 'joint_position',
                                                      'physics_contact', 'timestamp_s')}
    frames = frames_with_samples(load(run, 'acquisitions.json'), trace['timestamp_s'])
    lookup = {r['frame']: r for r in frames}
    first = frames[0]['physical_sample_index']
    checked = arrivals(episode, mission, frames, trace, requests)
    passed = {r['phase']: r for r in checked if r['passed']}
    selected = [r for r in planning if 'selection' in r]
    phases = {r['frame']: r['phase'] for r in mission}
    holds = [classify_hold(r) | dict(phase=phases.get(r['frame'], 'unknown')) for r in selected
             if r.get('action', r['selection']['action']) == 'hold']
    stall = {}
    for phase in sorted(set(phases.values())):
        n = sum(phases.get(r['frame']) == phase for r in selected)
        h = sum(r['phase'] == phase for r in holds)
        stall[phase] = dict(selected_plans=n, holds=h, rate=h/n if n else None)
    low, high, robust = clearance(spec, trace, first)
    contacts = trace['physics_contact'][first:].astype(bool)
    checks = safety(low, high, robust, contacts)
    np.savez_compressed(run/'native_clearance_summary_arrays.npz', timestamp_s=trace['timestamp_s'][first:],
                        separation_lower_m=low, separation_upper_m=high, interval_robust_lower_m=robust)

    def leg(start, end, success, shortest):
        length = float(np.linalg.norm(np.diff(trace['base_pose_world'][start:end+1, :2], axis=0), axis=1).sum())
        return dict(success=success, actual_path_m=length, shortest_path_m=shortest,
                    spl=float(shortest/max(length, shortest)) if success else 0.,
                    elapsed_s=float(trace['timestamp_s'][end]-trace['timestamp_s'][start]))
    logged = {r['phase']: r for r in checked}
    last = len(trace['timestamp_s'])-1
    outbound_end = lookup[logged['OUTBOUND']['frame']]['physical_sample_index'] if 'OUTBOUND' in logged else last
    return_end = lookup[logged['RETURN']['frame']]['physical_sample_index'] if 'RETURN' in logged else last
    outbound = leg(first, outbound_end, 'OUTBOUND' in passed, episode['shortest_outbound_m'])
    home = leg(outbound_end, return_end, 'RETURN' in passed, episode['shortest_return_m'])
    disallowed = int(np.count_nonzero(trace['physics_contact']))
    taxonomy = Counter()
    rejected = sum(not r['passed'] for r in checked)
    if rejected:
        taxonomy['observed arrival rejected by physical verification'] = rejected
    if disallowed:
        taxonomy['contact'] += 1
    obstacles = obstacle_record(run, trace)
    for contact in (obstacles or {}).get('contacts', []):
        taxonomy[f"obstacle contact ({contact['mover']} moving into the other)"] += 1
    if checks['hard']['confirmed_violation_samples']:
        taxonomy['hard-clearance violation'] += 1
    if (run/'failure.json').exists():
        error = load(run, 'failure.json')['reason'].lower()
        taxonomy['pose loss' if any(s in error for s in ('pose', 'tracking', 'registration')) else 'technical failure'] += 1
    for row in holds:
        if row['category'] == 'explicit_override':
            taxonomy[OVERRIDE_LABELS.get(row['override_reason'], 'blocked recovery/override')] += 1
        elif row['category'] == 'no_eligible_movement':
            for flag, label in (('observation_action_space_exclusions', 'observation/view restriction'),
                                ('motion_clearance_exclusions', 'memory clearance'),
                                ('stopping_projection_exclusions', 'stopping projection')):
                if row[flag]:
                    taxonomy[label] += 1
        elif row['category'] == 'movement_lost_recorded_score_or_tie':
            taxonomy['movement outscored'] += 1
        else:
            taxonomy['insufficient retained evidence'] += 1
    no_route = [r for r in planning if r.get('reason') == 'VIEW_BUDGET_EXHAUSTED']
    if no_route:
        taxonomy['no route on the map (scans exhausted)'] += 1
    if result['policy_steps'] >= STEP_BUDGET and 'RETURN' not in passed:
        if planning and planning[-1].get('reason') == 'VIEW_BUDGET_EXHAUSTED':
            taxonomy['budget exhaustion with no route on the map'] += 1
        else:
            taxonomy['budget exhaustion despite progress' if outbound['actual_path_m'] > .02 else 'budget exhaustion without movement'] += 1
    return dict(schema='go2nav_episode_evaluation.v1', controller=config.get('arm', config.get('controller')),
                episode_id=episode['episode_id'], role=episode['role'], beacon_success='OUTBOUND' in passed,
                home_success='RETURN' in passed, round_trip_success=all(k in passed for k in ('OUTBOUND', 'RETURN')),
                source_error=result['error'], arrivals=checked, outbound=outbound, return_leg=home,
                disallowed_contact_samples=disallowed, safety=checks, stall_by_phase=stall,
                hold_categories=dict(Counter(r['category'] for r in holds)), failure_and_stall_taxonomy=dict(taxonomy),
                hold_details=holds, wall_s=result['wall_s'], reader_wall_s=time.monotonic()-started,
                **(dict(moving_obstacles=obstacles) if obstacles is not None else {}))


def early_failure(run, result, config):
    """A mission that failed before its first frame: the settled state is the whole trace."""
    episode, spec = load(run, 'episode.json'), load(run, 'specification.json')
    leg = lambda shortest: dict(success=False, actual_path_m=0., shortest_path_m=shortest, spl=0., elapsed_s=0.)
    record = dict(schema='go2nav_episode_evaluation.v1', controller=config.get('arm', config.get('controller')),
                  episode_id=episode['episode_id'], role=episode['role'], beacon_success=False, home_success=False,
                  round_trip_success=False, source_error=result['error'], arrivals=[],
                  outbound=leg(episode['shortest_outbound_m']), return_leg=leg(episode['shortest_return_m']),
                  stall_by_phase={}, hold_categories={}, hold_details=[], wall_s=result['wall_s'])
    trace_path = run/'native/physics_trace.npz'
    if not trace_path.exists():
        return record | dict(status='STARTUP_FAILURE', failure_and_stall_taxonomy={'technical failure': 1})
    with np.load(trace_path, allow_pickle=False) as a:
        trace = {k: a[k].copy() for k in ('base_pose_world', 'joint_position', 'physics_contact')}
    lower, upper, _ = WallClearance(spec['geometry']['wall_boxes']).evaluate(trace['base_pose_world'][-2:], trace['joint_position'][-2:])
    low, high = float(lower[-1].min()), float(upper[-1].min())
    checks = {name: dict(threshold_m=threshold, confirmed_violation_samples=int(high < threshold),
                         unresolved_sampled_samples=int(low < threshold <= high), minimum_separation_lower_m=low,
                         minimum_separation_upper_m=high) for name, threshold in THRESHOLDS}
    return record | dict(disallowed_contact_samples=int(np.count_nonzero(trace['physics_contact'])), safety=checks,
                         failure_and_stall_taxonomy={'pose/perception pipeline failure': 1})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    record = evaluate(p.parse_args().run)
    print(json.dumps({k: record.get(k) for k in ('controller', 'episode_id', 'round_trip_success', 'disallowed_contact_samples',
                                                 'failure_and_stall_taxonomy')}))


if __name__ == '__main__':
    main()
