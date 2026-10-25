"""Action selection: score six forecast actions toward a waypoint, then keep the choice clear of obstacles.

**Scoring** (`score`). Each action's forecast pose at the start (300 ms) and end (700 ms) of its commitment gives its
progress toward the waypoint, plus a heading term: the reduction of the heading error to the waypoint, scaled by
min(0.35 m, waypoint distance). Contact is not predicted (its score is disabled).

**Clearance** (`clearance_check`). Each action's forecast path (eight 100-ms segments from the current pose) is
measured against the stored 1-cm obstacle cells (with obstacle prediction, moving tracks where they will be at each
segment's time: `planning.prediction`). A path is clear if it keeps the 0.45-m footprint plus a 3-cm
reserve for translations; one that starts inside the reserve may still pass if it leaves it (reserve recovery), and
a blocked translation passes if its clearance never decreases and ends higher (reserve exit). The best-scoring clear
action wins, else hold. In-place turns are re-checked with their own reserve (`reserve_turns`), and a blocked turn
passes if its clearance does not decrease (turn exit). The two exits are the reserve-exit v2 harness's; the earlier
v4 harness had neither (`exits=False`).

**Recovery and terminal stages** then refine the choice in a fixed order (`go2nav.planning.planner.Planner.select`):
a latched turn toward a blocked waypoint direction, the terminal approach near the goal, hold-relative recoveries,
stopping projection at dispatch time, an arc toward a supported view during a view recovery, unseen-floor coverage,
and memory of turns already tried here.

A selection is a dict passed through the stages; its main fields are `action`, `candidates` (score terms per action),
`scan_utilities` (when a view heading is being searched), `memory_forecast_candidates` (clearance per action) and
`clearance_turn` (the latched-turn state, when present).
"""
from copy import deepcopy
import itertools
import math

import numpy as np

from go2nav.planning import grid
from go2nav.planning.actions import (ACTIONS, COMMIT_TICKS, PERIOD_NS, PREFIX_TICKS, command, representative_turn,
                                     translating, turn_direction)

CONTACT_PENALTY_M = 1.2
SCAN_ANGULAR_SCALE_M = .35
TRANSLATION_RESERVE_M = TURN_RESERVE_M = .03
EXIT_TOLERANCE_M = .001
STOPPING_ALLOWANCE_S = .5
COVERAGE_RADIUS_M = .48
BACK_OFF_GAIN_M, BACK_OFF_REACH_M, BACK_OFF = .005, .10, 'BACKS_OFF_INSIDE_FOOTPRINT'


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def names(selection):
    """The selection's candidate names, in forecast order."""
    return [r['action'] for r in selection['candidates']]


def in_place(selection):
    """`hold` and the in-place turns among the selection's candidates."""
    return tuple(a for a in names(selection) if a == 'hold' or turn_direction(a))


def choose(selection, index):
    """Make action `index` the selection's choice (in place)."""
    action = names(selection)[index]
    selection.update(action=action, action_index=index, requested_command=command(action),
                     selected_reserve_recovery=bool(selection['memory_forecast_candidates'][index]['reserve_recovery_path_clear']),
                     selected_stepwise_recovery=bool(selection['memory_forecast_candidates'][index].get(
                         'stepwise_reserve_recovery_path_clear', False)))
    return selection


def utilities(selection):
    return {r['action']: r['utility_m'] for r in selection.get('scan_utilities', selection['candidates'])}


def best_clear(selection, allowed=None):
    """Index of the best-scoring clear action (optionally among `allowed`), else hold."""
    values = utilities(selection)
    eligible = [i for i, r in enumerate(selection['memory_forecast_candidates'])
                if r['action'] in values and r['nominal_predicted_path_clear'] and (allowed is None or r['action'] in allowed)]
    rows = selection['memory_forecast_candidates']
    return (max(eligible, key=lambda i: values[rows[i]['action']]) if eligible else names(selection).index('hold')), bool(eligible)


# -- scoring ---------------------------------------------------------------------------------------------------------

def score(prediction, goal_body, candidates=ACTIONS):
    """Score the forecasts (N, 8, 5: x, y, sin yaw, cos yaw, contact logit) of the candidates toward a waypoint in
    the body frame."""
    p, goal = np.asarray(prediction), np.asarray(goal_body, float)
    begin, end = PREFIX_TICKS-1, PREFIX_TICKS+COMMIT_TICKS-1
    start, stop = p[:, begin, :2], p[:, end, :2]
    progress = np.linalg.norm(goal-start, axis=-1)-np.linalg.norm(goal-stop, axis=-1)
    contact = np.exp(-np.logaddexp(0., -p[:, end, 4].astype(float)))
    utility = progress-CONTACT_PENALTY_M*contact
    result = dict(candidates=[dict(action=a, predicted_progress_during_commit_m=float(progress[i]),
                                   predicted_contact_by_commit_end=float(contact[i]), utility_m=float(utility[i]))
                              for i, a in enumerate(candidates)])
    scale = min(SCAN_ANGULAR_SCALE_M, float(np.linalg.norm(goal)))
    errors = []
    for index in (begin, end):
        delta = goal-p[:, index, :2]
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = np.arctan2(p[:, index, 2], p[:, index, 3])
        errors.append(np.abs(np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw))))
    alignment = scale*(errors[0]-errors[1])
    for i, row in enumerate(result['candidates']):
        row.update(position_contact_utility_m=row['utility_m'], predicted_alignment_progress_m=float(alignment[i]),
                   predicted_heading_error_at_commit_start_rad=float(errors[0][i]),
                   predicted_heading_error_at_commit_end_rad=float(errors[1][i]))
        row['utility_m'] += float(alignment[i])
    selected = max(range(len(candidates)), key=lambda i: result['candidates'][i]['utility_m'])
    action = candidates[selected]
    return result | dict(action=action, action_index=selected, requested_command=command(action),
                         waypoint_body_xy_m=goal.tolist(), alignment_scale_m=scale,
                         selection_objective='distance_and_heading_progress_minus_contact')


def scan_override(selection, prediction, scan_error):
    """When searching a view heading: hold or turn, by the forecast yaw toward the heading (0.35 m per radian)."""
    scores = []
    begin, end = PREFIX_TICKS-1, PREFIX_TICKS+COMMIT_TICKS-1
    for i, row in enumerate(selection['candidates']):
        if row['action'] != 'hold' and not turn_direction(row['action']):
            continue
        yaw = math.atan2(prediction[i, end, 2], prediction[i, end, 3])-math.atan2(prediction[i, begin, 2], prediction[i, begin, 3])
        remaining = math.atan2(math.sin(scan_error-yaw), math.cos(scan_error-yaw))
        scores.append((SCAN_ANGULAR_SCALE_M*(abs(scan_error)-abs(remaining))-CONTACT_PENALTY_M*row['predicted_contact_by_commit_end'], i))
    index = max(scores, key=lambda row: row[0])[1]
    selection.update(action=selection['candidates'][index]['action'], action_index=index,
                     requested_command=command(selection['candidates'][index]['action']), scan_heading_error_rad=scan_error,
                     scan_utilities=[dict(action=selection['candidates'][i]['action'], utility_m=float(s)) for s, i in scores])
    return selection


# -- clearance -------------------------------------------------------------------------------------------------------

def reserve_recovery_clear(distances, required):
    """Starts inside the reserve but outside the footprint, never encroaches further, ends outside the reserve."""
    if any(d is None for d in distances):
        return False
    prefix = min(distances[:3])
    return bool(.45+1e-12 < prefix <= required+1e-12 and min(distances[3:])+1e-12 >= prefix
                and distances[7] > required+1e-12)


def stepwise_recovery_clear(distances, required):
    """As reserve recovery, but only needs to gain max(1 mm, 10% of the deficit) by the end."""
    if any(d is None for d in distances):
        return False
    prefix = min(distances[:3])
    gain = max(.001, .1*(required-prefix))
    return bool(.45+1e-12 < prefix <= required+1e-12 and min(distances[3:])+1e-12 >= prefix
                and distances[-1] >= prefix+gain)


def exit_clear(distances):
    """Clearance never decreases (1 mm tolerance) and ends higher than it starts (an unbounded segment fails)."""
    if any(d is None for d in distances):
        return False
    d = [float(v) for v in distances]
    return bool(all(b >= a-EXIT_TOLERANCE_M for a, b in zip(d, d[1:])) and min(d) >= d[0]-EXIT_TOLERANCE_M and d[-1] > d[0])


def turn_hold_clear(distances):
    """Clearance never decreases (1 mm tolerance), step to step and against the start (an unbounded segment fails)."""
    if any(d is None for d in distances):
        return False
    d = [float(v) for v in distances]
    return bool(all(b >= a-EXIT_TOLERANCE_M for a, b in zip(d, d[1:])) and min(d) >= d[0]-EXIT_TOLERANCE_M)


def forecast_path(prediction, i, position, rotation):
    """The current position and action i's eight forecast positions, in the map frame (xy)."""
    return np.asarray(position)[:2]+np.vstack((np.zeros(2), prediction[i, :, :2]))@np.asarray(rotation)[:2, :2].T


def clearance_check(selection, prediction, clearance, position, rotation, exits=True):
    """Keep the choice clear of stored obstacles (see the module docstring), with reserve recovery and (`exits`) the
    reserve exit."""
    rows = []
    during = getattr(clearance, 'minimum_during', None)  # moving obstacles predicted (planning.prediction)
    step = PERIOD_NS/1e9
    for i, action in enumerate(names(selection)):
        points = forecast_path(prediction, i, position, rotation)
        minimum, distances = None, []
        for k, (begin, end) in enumerate(zip(points[:-1], points[1:])):
            value = during(begin, end, k*step, (k+1)*step) if during is not None else clearance.minimum(begin, end)
            distances.append(value)
            if value is not None:
                minimum = value if minimum is None else min(minimum, value)
        reserve = TRANSLATION_RESERVE_M if any(command(action)[:2]) else 0.
        required = .45+reserve
        full = minimum is None or minimum > required+1e-12
        recovery = bool(reserve > 0 and not full and reserve_recovery_clear(distances, required))
        rows.append(dict(action=action, minimum_predicted_path_clearance_m=minimum, prediction_error_reserve_m=reserve,
                         required_path_clearance_m=required, nominal_predicted_path_clear=full or recovery,
                         full_reserve_path_clear=full, reserve_recovery_path_clear=recovery,
                         nominal_footprint_path_clear=minimum is None or minimum > .45+1e-12,
                         segment_clearances_m=distances,
                         clearance_check_mode='FULL_RESERVE' if full else 'RESERVE_RECOVERY' if recovery else 'BLOCKED'))
    result = selection | dict(before_memory_filter_action=selection['action'], memory_forecast_candidates=rows)
    passed = []
    for row in rows if exits else ():
        passes = bool(translating(row['action']) and not row['nominal_predicted_path_clear']
                      and exit_clear(row['segment_clearances_m']))
        row['reserve_exit_path_clear'] = passes
        if passes:
            row.update(nominal_predicted_path_clear=True, clearance_check_mode='RESERVE_EXIT')
            passed.append(row['action'])
    index, eligible = best_clear(result)
    result.update(action=names(result)[index], action_index=index, requested_command=command(names(result)[index]),
                  memory_forecast_status='CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED',
                  selected_reserve_recovery=bool(rows[index]['reserve_recovery_path_clear']))
    if exits:
        result.update(reserve_exit_candidates=passed, selected_reserve_exit=bool(rows[index]['reserve_exit_path_clear']))
    return result


def reserve_turns(selection, exits=True):
    """Re-check in-place turns with a 3-cm reserve (stepwise recovery allowed), then (`exits`) the turn exit;
    reselect."""
    result = deepcopy(selection)
    for row in result['memory_forecast_candidates']:
        if not turn_direction(row['action']):
            continue
        required = .45+TURN_RESERVE_M
        full = row['minimum_predicted_path_clearance_m'] is None or row['minimum_predicted_path_clearance_m'] > required+1e-12
        recovery = not full and reserve_recovery_clear(row['segment_clearances_m'], required)
        gradual = bool(not full and stepwise_recovery_clear(row['segment_clearances_m'], required))
        row.update(prediction_error_reserve_m=TURN_RESERVE_M, required_path_clearance_m=required,
                   full_reserve_path_clear=full, reserve_recovery_path_clear=bool(recovery or gradual),
                   stepwise_reserve_recovery_path_clear=gradual, nominal_predicted_path_clear=bool(full or recovery or gradual),
                   clearance_check_mode='FULL_RESERVE' if full else 'RESERVE_RECOVERY' if recovery else
                   'STEPWISE_RESERVE_RECOVERY' if gradual else 'BLOCKED')
    index, eligible = best_clear(result)
    result['memory_forecast_status'] = 'CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED'
    choose(result, index)
    if not exits:
        return result
    passed = []
    for row in result['memory_forecast_candidates']:
        passes = bool(turn_direction(row['action']) and not row['nominal_predicted_path_clear']
                      and turn_hold_clear(row['segment_clearances_m']))
        row['turn_exit_path_clear'] = passes
        if passes:
            row.update(nominal_predicted_path_clear=True, clearance_check_mode='TURN_EXIT')
            passed.append(row['action'])
    index, eligible = best_clear(result)
    result.update(memory_forecast_status='CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED',
                  turn_exit_candidates=passed)
    choose(result, index)
    result['selected_turn_exit'] = bool(result['memory_forecast_candidates'][index].get('turn_exit_path_clear'))
    return result


def recover_turn(selection, heading, generation, latch, exits=True):
    """Latch a turn the long way round when the preferred turn toward the waypoint is blocked but the other is clear;
    follow the latch until the heading is reached. Returns (selection, latch)."""
    result = reserve_turns(selection, exits)
    if latch is not None and latch['mission_generation'] != generation:
        latch = None
    if latch is not None and 'scan_utilities' not in result:
        action = result['action']
        candidates = {r['action']: r for r in result['candidates']}
        clearance = next(r for r in result['memory_forecast_candidates'] if r['action'] == action)
        if (translating(action) and clearance['full_reserve_path_clear']
                and candidates[action]['predicted_progress_during_commit_m'] > 0.
                and candidates[action]['position_contact_utility_m'] > candidates['hold']['position_contact_utility_m']):
            result['clearance_turn'] = dict(active=False, event='FULL_RESERVE_PROGRESS_REJOINS_ROUTE',
                                            previous_target_heading_rad=latch['target_heading_rad'], released_action=action)
            return result, None
    event = None
    if latch is not None:
        error = wrap(latch['target_heading_rad']-heading)
        remaining = (latch['direction']*error)%(2*math.pi)
        crossed = remaining-latch['previous_remaining_rad'] > math.pi
        if abs(error) <= .1 or crossed:
            latch, event = None, 'MEASURED_TARGET_HEADING_REACHED'
        else:
            latch = latch | dict(previous_remaining_rad=remaining)
    preferred = result['before_memory_filter_action']
    by_action = {r['action']: r for r in result['memory_forecast_candidates']}
    turn = lambda direction: representative_turn(names(result), direction)
    if result['action'] == 'hold' and translating(preferred):
        goal = result['waypoint_body_xy_m']
        error = math.atan2(goal[1], goal[0])
        if abs(error) > .1:
            preferred = turn(1 if error > 0 else -1)
    if latch is None and event is None and turn_direction(preferred):
        alternative = turn(-turn_direction(preferred))
        error = result.get('scan_heading_error_rad')
        if error is None:
            goal = result['waypoint_body_xy_m']
            error = math.atan2(goal[1], goal[0])
        if (abs(error) > .1 and not by_action[preferred]['nominal_predicted_path_clear']
                and by_action[alternative]['nominal_predicted_path_clear']):
            direction = turn_direction(alternative)
            latch = dict(target_heading_rad=wrap(heading+error), direction=direction,
                         previous_remaining_rad=(direction*error)%(2*math.pi), mission_generation=generation,
                         blocked_preferred_action=preferred)
            event = 'CLEAR_ALTERNATIVE_TURN_LATCHED'
    if latch is not None:
        action = turn(latch['direction'])
        clear = by_action[action]['nominal_predicted_path_clear']
        other = turn(-latch['direction'])
        full_alternative = by_action[other].get('full_reserve_path_clear', False)
        if not clear and (full_alternative or by_action[other].get('stepwise_reserve_recovery_path_clear')):
            direction = -latch['direction']
            latch = latch | dict(direction=direction,
                                 previous_remaining_rad=(direction*wrap(latch['target_heading_rad']-heading))%(2*math.pi),
                                 reserve_recovery_direction_switches=latch.get('reserve_recovery_direction_switches', 0)+1)
            action, clear = other, True
            event = ('FULL_RESERVE_ALTERNATIVE_DIRECTION_SELECTED' if full_alternative
                     else 'CLEARANCE_INCREASING_RECOVERY_DIRECTION_SELECTED')
        choose(result, names(result).index(action if clear else 'hold'))
        result['clearance_turn'] = latch | dict(active=True, event=event, latched_turn_forecast_clear=clear)
    elif event is not None:
        result['clearance_turn'] = dict(active=False, event=event)
    return result, latch


# -- terminal approach -----------------------------------------------------------------------------------------------

def terminal_progress(selection):
    """Near the goal: prefer the clear translation with the best position score over turning to the heading."""
    result = deepcopy(selection)
    if (result.get('clearance_turn') or {}).get('active'):
        return result
    candidates = {r['action']: r for r in result['candidates']}
    baseline = max(candidates['hold']['position_contact_utility_m'], candidates[result['action']]['position_contact_utility_m'])
    eligible = [i for i, row in enumerate(result['memory_forecast_candidates'])
                if translating(row['action']) and row['nominal_predicted_path_clear']
                and candidates[row['action']]['predicted_progress_during_commit_m'] > 0.
                and candidates[row['action']]['position_contact_utility_m'] > baseline]
    original = result['action']
    if eligible:
        choose(result, max(eligible, key=lambda i: result['candidates'][i]['position_contact_utility_m']))
    result['terminal_position_priority'] = dict(original_action=original, selected_action=result['action'],
                                                changed=original != result['action'],
                                                original_selection_objective=result.get('selection_objective'))
    result['selection_objective'] = 'terminal_position_contact_priority_with_heading_fallback'
    return result


def predicted_arrival_hold(selection, prediction, arrival_radius_m):
    """Hold when the hold forecast already ends inside the arrival radius, nearly still (≤ 5 cm/s)."""
    if (selection.get('clearance_turn') or {}).get('active'):
        return selection
    hold = names(selection).index('hold')
    xy = np.asarray(prediction)[hold, -2:, :2]
    distances = np.linalg.norm(xy-np.asarray(selection['waypoint_body_xy_m']), axis=-1)
    speed = float(np.linalg.norm(xy[1]-xy[0])/.1)
    clear = selection['memory_forecast_candidates'][hold]['nominal_predicted_path_clear']
    eligible = bool(clear and np.isfinite(xy).all() and np.all(distances <= arrival_radius_m) and speed <= .05)
    result = deepcopy(selection)
    original = result['action']
    if eligible:
        choose(result, hold)
        result['selection_objective'] = 'predicted_quiet_arrival_hold'
    result['predictive_arrival_hold'] = dict(eligible=eligible, original_action=original, selected_action=result['action'],
                                             changed=original != result['action'])
    return result


def require_arrival_entry(selection, prediction, arrival_radius_m):
    """Undo the terminal translation preference unless that translation's forecast ends inside the arrival radius."""
    result = deepcopy(selection)
    priority = result.get('terminal_position_priority') or {}
    considered = bool(priority.get('changed') and result['action'] == priority.get('selected_action'))
    changed = False
    if considered:
        endpoint = np.asarray(prediction)[names(result).index(result['action']), 6, :2]
        distance = float(np.linalg.norm(endpoint-np.asarray(result['waypoint_body_xy_m'])))
        if not np.isfinite(distance) or distance > arrival_radius_m:
            choose(result, names(result).index(priority['original_action']))
            result['selection_objective'] = priority['original_selection_objective']
            changed = True
    result['arrival_entry_terminal_priority'] = dict(considered=considered, changed=changed)
    return result


# -- recoveries relative to holding ----------------------------------------------------------------------------------

def _finite(values):
    return len(values) == 8 and all(v is not None and math.isfinite(v) for v in values)


def recover_against_hold(selection, inside=False):
    """When everything but hold is blocked: allow a move that stays outside the footprint, never closer than hold
    gets, and ends further from obstacles than hold does by max(1 mm, 10% of its reserve deficit). With `inside` (a
    lattice), a robot already inside the footprint (the continuous policy creeps about 1 mm/s on a zero command) backs
    off: the translation never closer than hold gets that ends furthest out, at least `BACK_OFF_GAIN_M` beyond hold."""
    rows = selection.get('memory_forecast_candidates', [])
    if selection.get('action') != 'hold':
        return selection
    hold = next(row for row in rows if row['action'] == 'hold')['segment_clearances_m']
    if not _finite(hold):
        return selection
    if min(hold) <= .45:  # inside: also while scanning, whose utilities leave out every translation
        return _back_off(selection, rows, hold) if inside else selection
    if any(row['nominal_predicted_path_clear'] for row in rows if row['action'] != 'hold'):
        return selection
    eligible = []
    for i, row in enumerate(rows):
        if row['action'] == 'hold':
            continue
        path = row['segment_clearances_m']
        gain = max(.001, .1*max(0., row['required_path_clearance_m']-min(hold)))
        if _finite(path) and min(path) > .45 and min(path) >= min(hold) and path[-1] >= hold[-1]+gain:
            eligible.append((min(path), path[-1], i, gain))
    if not eligible:
        return selection
    _, _, index, _ = max(eligible)
    result = deepcopy(selection)
    row = result['memory_forecast_candidates'][index]
    row.update(nominal_predicted_path_clear=True, reserve_recovery_path_clear=True,
               clearance_check_mode='IMPROVES_ON_HOLD_WITHIN_NOMINAL_FOOTPRINT')
    result.update(memory_forecast_status='HOLD_RELATIVE_CLEARANCE_RECOVERY',
                  clearance_turn=dict(active=False, event='HOLD_RELATIVE_RECOVERY_RESELECTS_EACH_PLAN'))
    return choose(result, names(result).index(row['action']))


def _back_off(selection, rows, hold):
    eligible = [(row['segment_clearances_m'][-1], min(row['segment_clearances_m']), i) for i, row in enumerate(rows)
                if translating(row['action']) and _finite(row['segment_clearances_m'])
                and min(row['segment_clearances_m']) >= min(hold) and row['segment_clearances_m'][-1] >= hold[-1]+BACK_OFF_GAIN_M]
    if not eligible:
        return selection
    _, _, index = max(eligible)
    result = deepcopy(selection)
    row = result['memory_forecast_candidates'][index]
    row.update(nominal_predicted_path_clear=True, reserve_recovery_path_clear=True, clearance_check_mode=BACK_OFF)
    result.update(memory_forecast_status='INSIDE_FOOTPRINT_BACK_OFF',
                  clearance_turn=dict(active=False, event='INSIDE_FOOTPRINT_BACK_OFF'))
    return choose(result, names(result).index(row['action']))


def heading_release(selection, latch):
    """Whether a latched turn would be released for the preferred turn (the release itself is suppressed: the latch
    is kept until its measured heading is reached). Returns (selection, latch) unchanged in effect."""
    turn = selection.get('clearance_turn') or {}
    preferred, current = selection.get('before_memory_filter_action'), selection['action']
    if (latch is None or not turn.get('active') or not turn_direction(preferred) or not turn_direction(current)
            or preferred == current):
        return selection, latch
    if current != representative_turn(names(selection), latch['direction']):
        return selection, latch
    clearance = next(r for r in selection['memory_forecast_candidates'] if r['action'] == preferred)
    if not clearance['full_reserve_path_clear']:
        return selection, latch
    values = utilities(selection)
    if not all(math.isfinite(values[a]) for a in (preferred, current, 'hold')) or values[preferred] <= values[current]:
        return selection, latch
    if 'scan_utilities' in selection:
        improving = values[preferred] > values['hold']
    else:
        candidate = next(r for r in selection['candidates'] if r['action'] == preferred)
        improving = candidate['predicted_heading_error_at_commit_end_rad'] < candidate['predicted_heading_error_at_commit_start_rad']
    if not improving:
        return selection, latch
    return selection | dict(suppressed_early_heading_release=dict(previous_action=current, selected_action=preferred)), latch


def reposition_survey(selection):
    """While a latched view turn goes the long way: prefer a clear translation that improves on holding."""
    turn = selection.get('clearance_turn') or {}
    preferred = selection.get('before_memory_filter_action')
    if ('scan_utilities' not in selection or not turn.get('active') or not turn_direction(preferred)
            or not turn_direction(selection['action']) or selection['action'] == preferred):
        return selection
    rows = {r['action']: r for r in selection['memory_forecast_candidates']}
    hold = rows['hold']['segment_clearances_m']
    if not _finite(hold) or min(hold) <= .45:
        return selection
    eligible = []
    order = names(selection)
    for action in (a for a in order if translating(a)):
        row = rows[action]
        path = row['segment_clearances_m']
        gain = max(.001, .1*max(0., row['required_path_clearance_m']-min(hold)))
        if (row['full_reserve_path_clear'] and _finite(path) and min(path) > row['required_path_clearance_m']+1e-12
                and min(path) >= min(hold) and path[-1] >= hold[-1]+gain):
            eligible.append((min(path), path[-1], -order.index(action), gain))
    if not eligible:
        return selection
    _, _, negative_index, _ = max(eligible)
    result = deepcopy(selection)
    choose(result, -negative_index)
    result['clearance_turn'] = dict(active=False, event='FULL_RESERVE_SURVEY_REPOSITION', previous_latch=turn)
    result['survey_clearance_reposition'] = dict(previous_action=selection['action'], selected_action=result['action'])
    return result


def recover_heading_against_hold(selection):
    """When everything but hold is blocked: allow the preferred turn if it never comes closer to obstacles than holding
    does, ends at least as far from them, and scores better than holding (without a scan, it must also reduce the
    predicted heading error)."""
    preferred = selection.get('before_memory_filter_action')
    rows = selection.get('memory_forecast_candidates', [])
    if (selection.get('action') != 'hold' or not turn_direction(preferred)
            or any(r['nominal_predicted_path_clear'] for r in rows if r['action'] != 'hold')):
        return selection
    by_action = {r['action']: r for r in rows}
    hold, turn = by_action['hold']['segment_clearances_m'], by_action[preferred]['segment_clearances_m']
    if (not _finite(hold) or not _finite(turn) or min(hold) <= .45 or min(turn) < min(hold) or turn[-1] < hold[-1]):
        return selection
    values = utilities(selection)
    if (not all(a in values and math.isfinite(values[a]) for a in ('hold', preferred))
            or values[preferred] <= values['hold']):
        return selection
    if 'scan_utilities' not in selection:
        candidate = next(r for r in selection['candidates'] if r['action'] == preferred)
        start = candidate['predicted_heading_error_at_commit_start_rad']
        end = candidate['predicted_heading_error_at_commit_end_rad']
        if not math.isfinite(start) or not math.isfinite(end) or end >= start:
            return selection
    result = deepcopy(selection)
    index = names(result).index(preferred)
    result['memory_forecast_candidates'][index].update(nominal_predicted_path_clear=True, reserve_recovery_path_clear=True,
                                                       clearance_check_mode='HEADING_PROGRESS_WITHIN_HOLD_RELATIVE_MINIMUM')
    result.update(memory_forecast_status='HOLD_RELATIVE_HEADING_RECOVERY',
                  clearance_turn=dict(active=False, event='HOLD_RELATIVE_HEADING_RESELECTS_EACH_PLAN'),
                  hold_relative_heading_recovery=dict(applied=True, selected_action=preferred))
    return choose(result, index)


# -- dispatch-time stopping and floor coverage -----------------------------------------------------------------------

def stopping_projection(selection, prediction, clearance, position, rotation, *, pulse=False, admissible=False):
    """A translation must also stop clear if dispatched from any forecast pose with obstacles seen then: from each
    pose up to expiry, the requested-speed distance to expiry plus 0.5 s must keep the footprint. Otherwise align or hold
    (`admissible`, for a lattice: the best clear candidate whose projection is clear, translations included, as the
    dynamic window approach takes the best admissible velocity)."""
    position, rotation = np.asarray(position, float), np.asarray(rotation, float)
    checks = []
    for i, action in enumerate(names(selection)):
        motion = np.asarray(command(action), float)
        if not any(motion[:2]):
            checks.append(dict(action=action, translating=False, projection_clear=True))
            continue
        expiry_tick, clear = 4 if pulse else 7, True
        for tick in range(1, expiry_tick):
            forecast = prediction[i, tick-1]
            yaw = math.atan2(forecast[2], forecast[3])
            c, s = math.cos(yaw), math.sin(yaw)
            turn = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            q = position+rotation@np.r_[forecast[:2], 0.]
            Q = turn@rotation
            horizon = (expiry_tick-tick)*.1+STOPPING_ALLOWANCE_S
            endpoint = q+Q@(motion*np.array([horizon, horizon, 0.]))
            minimum = clearance.minimum(q[:2], endpoint[:2])
            clear &= minimum is None or minimum > grid.FOOTPRINT_RADIUS_M+1e-12
        checks.append(dict(action=action, translating=True, projection_clear=clear))
    result = deepcopy(selection)
    result['planned_stopping_projection'] = dict(candidates=checks, before_action=selection['action'], changed=False)
    selected = next(row for row in checks if row['action'] == selection['action'])
    backing_off = selection.get('memory_forecast_status') == 'INSIDE_FOOTPRINT_BACK_OFF'  # starts inside: its own test
    if not selected['translating'] or selected['projection_clear'] or backing_off:
        return result
    allowed = tuple(r['action'] for r in checks if r['projection_clear']) if admissible else in_place(selection)
    index, _ = best_clear(selection, allowed=allowed)
    choose(result, index)
    result['planned_stopping_projection'].update(changed=True, after_action=result['action'])
    return result


def footprint_extension(prediction, position, rotation, floor, candidates=ACTIONS):
    """Per candidate: the map cells within 0.48 m of its forecast path that are not observed floor, beyond those the
    current pose and holding already touch."""
    paths = np.asarray(position)[:2]+np.concatenate((np.zeros((len(candidates), 1, 2)), prediction[:, :, :2]),
                                                    axis=1)@np.asarray(rotation)[:2, :2].T
    low = np.floor((paths.min(axis=(0, 1))-COVERAGE_RADIUS_M)/grid.CELL_M).astype(int)
    high = np.floor((paths.max(axis=(0, 1))+COVERAGE_RADIUS_M)/grid.CELL_M).astype(int)
    low, high = np.maximum(low, -grid.HALF_CELLS), np.minimum(high, grid.HALF_CELLS-1)
    cells = np.array(list(itertools.product(range(low[0], high[0]+1), range(low[1], high[1]+1))), dtype=int).reshape(-1, 2)
    unknown = np.array([tuple(c) not in floor for c in cells], dtype=bool)
    current = (grid.segment_cell_distances(paths[0, 0], paths[0, 0], cells) <= COVERAGE_RADIUS_M)&unknown
    masks = [np.logical_or.reduce([grid.segment_cell_distances(a, b, cells) <= COVERAGE_RADIUS_M
                                   for a, b in zip(path, path[1:])])&unknown for path in paths]
    baseline = current|masks[candidates.index('hold')]
    return [dict(action=action, new_unknown_cells=int((mask&~baseline).sum()), added_cells=cells[mask&~baseline].tolist(),
                 leaves_map_bounds=bool((path.min(0)-COVERAGE_RADIUS_M < -grid.MAP_HALF_WIDTH_M).any()
                                        or (path.max(0)+COVERAGE_RADIUS_M > grid.MAP_HALF_WIDTH_M).any()))
            for action, path, mask in zip(candidates, paths, masks)]


def require_floor_coverage(selection, prediction, snapshot, position, rotation, occupied_observed=True, admissible=False):
    """A translation must not sweep unobserved floor: otherwise align or hold, and return the nearest unobserved cell
    it would have swept (to be viewed). Returns (selection, cell or None). Observed occupied cells count as observed
    (the coverage fix of 2 October); with `occupied_observed=False`, the earlier rule, only observed floor does (the
    view target is still an unobserved cell). With `admissible` (a lattice), the best clear, stopping-clear candidate
    that sweeps no unobserved floor is taken instead, translations included (no view target then). A back-off from inside
    the footprint (`recover_against_hold`) reaching at most `BACK_OFF_REACH_M` is exempt."""
    if not translating(selection['action']):
        return selection, None
    observed = snapshot.floor|snapshot.occupied
    order = names(selection)
    needed = sorted({order.index('hold'), order.index(selection['action'])})  # the baseline and the choice only
    rows = footprint_extension(np.asarray(prediction)[needed], position, rotation,
                               observed if occupied_observed else snapshot.floor, [order[i] for i in needed])
    selected = next(r for r in rows if r['action'] == selection['action'])
    if selected['new_unknown_cells'] == 0 and not selected['leaves_map_bounds']:
        return selection | dict(translation_footprint_coverage=dict(rejected=False)), None
    reach = float(np.linalg.norm(np.asarray(prediction)[order.index(selection['action']), -1, :2]))
    if selection.get('memory_forecast_status') == 'INSIDE_FOOTPRINT_BACK_OFF' and reach <= BACK_OFF_REACH_M:
        # a few centimetres into space the body has just left (the cameras never see behind): less risk than creeping on
        return selection | dict(translation_footprint_coverage=dict(rejected=False, back_off_exempt=True,
                                                                     new_unknown_cells=selected['new_unknown_cells'])), None
    values = utilities(selection)
    stopping = {r['action']: r['projection_clear'] for r in selection['planned_stopping_projection']['candidates']}
    if admissible:
        action = _covered_alternative(selection, prediction, position, rotation,
                                      observed if occupied_observed else snapshot.floor, stopping, values)
        if action is not None:
            result = choose(deepcopy(selection), order.index(action))
            result['translation_footprint_coverage'] = dict(rejected=True, previous_action=selection['action'],
                                                            selected_action=action)
            return result, None
    eligible = [r['action'] for r in selection['memory_forecast_candidates']
                if r['action'] in in_place(selection) and r['nominal_predicted_path_clear']
                and stopping[r['action']] and r['action'] in values]
    action = max(eligible, key=lambda a: values[a]) if eligible else 'hold'
    result = deepcopy(selection)
    choose(result, order.index(action))
    result['translation_footprint_coverage'] = dict(rejected=True, previous_action=selection['action'], selected_action=action)
    targets = [tuple(c) for c in selected['added_cells'] if tuple(c) not in observed]
    target = min(targets, key=lambda c: (float(np.linalg.norm(grid.centre(c)-np.asarray(position)[:2])), c)) if targets else None
    return result, target


def _covered_alternative(selection, prediction, position, rotation, observed, stopping, values):
    """The best-scoring clear, stopping-clear translation (other than the choice) sweeping no unobserved floor, if it
    scores above every clear in-place candidate; else None."""
    order = names(selection)
    rows = [r for r in selection['memory_forecast_candidates']
            if r['nominal_predicted_path_clear'] and stopping[r['action']] and r['action'] in values]
    best_still = max((values[r['action']] for r in rows if r['action'] in in_place(selection)), default=-math.inf)
    hold = order.index('hold')
    for r in sorted(rows, key=lambda r: -values[r['action']]):
        action = r['action']
        if values[action] <= best_still:
            return None
        if not translating(action) or action == selection['action']:
            continue
        i = order.index(action)
        row = footprint_extension(np.asarray(prediction)[[hold, i]], position, rotation, observed, ['hold', action])[1]
        if row['new_unknown_cells'] == 0 and not row['leaves_map_bounds']:
            return action
    return None


# -- memory of turns tried at this place -----------------------------------------------------------------------------

def recover_view_with_arc(selection, prediction, view_body, max_distance_m):
    """During a view recovery (`perception.visual_support`), when the search holds because neither turn is clear: take
    an arc toward the view's heading instead, if one is already clear. `view_body`: the supported view's position in
    the body frame. The arc must be clear under the full or recovery reserve and its stopping projection, turn the
    right way, reduce the heading error by more than hold does, and keep within `max_distance_m` of the view."""
    if selection['action'] != 'hold' or 'scan_utilities' not in selection:
        return selection
    rows = {r['action']: r for r in selection['memory_forecast_candidates']}
    if (any(r['nominal_predicted_path_clear'] for a, r in rows.items() if turn_direction(a))
            or not rows['hold']['nominal_footprint_path_clear']):
        return selection
    stopping = {r['action']: r['projection_clear'] for r in selection['planned_stopping_projection']['candidates']}
    prediction, view = np.asarray(prediction, float), np.asarray(view_body, float)
    error = selection['scan_heading_error_rad']
    if np.linalg.norm(view) > max_distance_m or abs(error) <= .1:
        return selection

    def heading_gain(index):
        yaw = math.atan2(prediction[index, 6, 2], prediction[index, 6, 3])
        yaw -= math.atan2(prediction[index, 2, 2], prediction[index, 2, 3])
        return abs(error)-abs(wrap(error-yaw))

    order = names(selection)
    hold_gain = heading_gain(order.index('hold'))
    eligible = []
    for action in (a for a in order if translating(a) and command(a)[2]):
        i, row = order.index(action), rows[action]
        if not (row['nominal_predicted_path_clear'] and row['nominal_footprint_path_clear']
                and row['clearance_check_mode'] in ('FULL_RESERVE', 'RESERVE_RECOVERY') and stopping[action]):
            continue
        if command(action)[2]*error <= 0:
            continue
        gain = heading_gain(i)
        path = np.column_stack((prediction[i, :, :2], np.zeros(8)))
        distance = float(np.linalg.norm(path-view, axis=1).max())
        if gain <= max(0., hold_gain) or distance > max_distance_m:
            continue
        eligible.append((gain, -i, distance))
    if not eligible:
        return selection
    gain, negative_index, distance = max(eligible)
    result = deepcopy(selection)
    choose(result, -negative_index)
    result['measured_view_arc_recovery'] = dict(applied=True, previous_action='hold', selected_action=result['action'],
                                                predicted_heading_gain_rad=gain, hold_heading_gain_rad=hold_gain,
                                                maximum_predicted_distance_from_reference_m=distance)
    return result


class RouteTurnMemory:
    """Remembers turns toward the waypoint that ended in a view recovery, and tries the other direction next time.

    The *attempt* is the turn being made toward the waypoint at this place (within 20 cm and 0.3 rad). If a view
    recovery starts (`weak_trigger`) after it has turned at least 0.025 rad, the attempt is recorded as failed (the
    last 16 are kept). When the preferred turn has failed here and the other direction has not and is clear, the other
    direction is latched (*active*) until the waypoint is within 0.1 rad, the place changes or the turn passes the
    target; while latched, that turn is chosen if clear, else hold. A latched direction that is no longer clear is
    released. A mission phase change clears everything; a translation or a view search clears the attempt and latch."""

    def __init__(self):
        self.generation = 0
        self.attempt = self.active = None
        self.failed = []
        self.last_trigger = -1

    @staticmethod
    def nearby(record, position, target):
        return np.linalg.norm(position-record['position']) <= .20 and abs(wrap(target-record['target_heading_rad'])) <= .30

    def select(self, selection, position, heading, generation, weak_trigger):
        released = None
        if self.active is not None and self.generation == generation and weak_trigger is None \
                and 'scan_utilities' not in selection:
            action = representative_turn(names(selection), self.active['direction'])
            row = next(r for r in selection['memory_forecast_candidates'] if r['action'] == action)
            if not row['nominal_predicted_path_clear']:
                released = dict(action=action, target_heading_rad=self.active['target_heading_rad'])
                self.active = self.attempt = None
        result = self._select(selection, position, heading, generation, weak_trigger)
        return result if released is None else result | dict(released_visual_route_turn_memory=released)

    def _select(self, selection, position, heading, generation, weak_trigger):
        position = np.asarray(position, float)[:2]
        if generation != self.generation:
            self.attempt = self.active = None
            self.failed = []
            self.generation = generation
        if weak_trigger is not None:
            if weak_trigger > self.last_trigger:
                self.last_trigger = weak_trigger
                attempt = self.attempt
                if attempt is not None and np.linalg.norm(position-attempt['position']) <= .20 \
                        and attempt['direction']*wrap(heading-attempt['start_heading_rad']) >= .025:
                    self.failed = (self.failed+[attempt | dict(trigger_ns=weak_trigger)])[-16:]
                self.attempt = self.active = None
            return selection
        if 'scan_utilities' in selection:
            self.attempt = self.active = None
            return selection
        error = math.atan2(*selection['waypoint_body_xy_m'][::-1])
        target = wrap(heading+error)
        if self.active is not None:
            remaining = (self.active['direction']*wrap(self.active['target_heading_rad']-heading))%(2*math.pi)
            crossed = remaining-self.active['previous_remaining_rad'] > math.pi
            if not self.nearby(self.active, position, target) or abs(error) <= .10 or crossed:
                self.active = None
            else:
                self.active['previous_remaining_rad'] = remaining
        by_action = {r['action']: r for r in selection['memory_forecast_candidates']}
        if translating(selection['action']):
            self.attempt = self.active = None
            return selection
        failed = {r['direction'] for r in self.failed if self.nearby(r, position, target)}
        preferred = selection.get('before_memory_filter_action')
        if self.active is None and turn_direction(preferred) and abs(error) > .10:
            direction = -turn_direction(preferred)
            alternative = representative_turn(names(selection), direction)
            if turn_direction(preferred) in failed and direction not in failed and by_action[alternative]['nominal_predicted_path_clear']:
                self.active = dict(position=position.copy(), target_heading_rad=target, direction=direction,
                                   previous_remaining_rad=(direction*error)%(2*math.pi))
        result = selection
        if self.active is not None:
            action = representative_turn(names(selection), self.active['direction'])
            eligible = by_action[action]['nominal_predicted_path_clear']
            result = deepcopy(selection)
            choose(result, names(result).index(action if eligible else 'hold'))
            result['visual_route_turn_memory'] = dict(active=True, direction=self.active['direction'],
                                                      target_heading_rad=self.active['target_heading_rad'],
                                                      failed_directions=sorted(failed), selected_turn_forecast_clear=eligible)
        action = result['action']
        if turn_direction(action):
            if (self.attempt is None or self.attempt['direction'] != turn_direction(action)
                    or not self.nearby(self.attempt, position, target)):
                self.attempt = dict(position=position.copy(), target_heading_rad=target, direction=turn_direction(action),
                                    start_heading_rad=heading)
        elif self.active is None:
            self.attempt = None
        return result
