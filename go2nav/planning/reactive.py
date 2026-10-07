"""C2: the model-free reactive selector.

Each plan picks the action primitive nearest to an instantaneous feedback command toward the waypoint: forward speed
0.2·max(0, cos e) and yaw rate clip(4.5·e, ±0.45) for the heading error e to the waypoint (when scanning toward a view
heading: no translation, e is the view-heading error). Distances are normalised by 0.2 m/s and 0.45 rad/s.

An action is eligible if the rule allows it (translations only when heading for a waypoint) and its dispatched
command path is clear: the forecasting controllers' clearance check (`selection.clearance_check` and
`selection.reserve_turns`, with their reserves, recoveries and exits) on the kinematic path of the commands as they
would be dispatched (the committed prefix, the action for four ticks, one in pulse mode, then a stop; unicycle
integration at 10 ms). No eligible action: hold. No forecast is used anywhere.

In the terminal approach (translation pulses), the heading-first rule replaces the choice while the robot's current
position is clear of the 0.45-m footprint: hold within the arrival radius, else forward if the waypoint is within
0.1 rad of straight ahead, else turn toward it, provided that action is eligible.

Under the earlier v4 harness (`select_v4`) there is no path check: every action is ineligible (hold) unless the
current stored clearance exceeds the 0.45-m footprint, and the feedback command is zero then.
"""
from copy import deepcopy
import math

import numpy as np

from go2nav.planning import selection as sel
from go2nav.planning.actions import ACTIONS, HORIZONS, command, command_sequences

MAX_FORWARD, MAX_YAW = .20, .45
HEADING_TOLERANCE_RAD = .1
HEADING_GAIN = MAX_YAW/HEADING_TOLERANCE_RAD
SUBSTEPS, TICK_S = 10, .1
FOOTPRINT_M = .45
HEADING_FIRST_RULE = 'terminal_heading_first'


def nominal_paths(prefix, pulse):
    """(6, 8, 3) body-frame x, y, yaw at 0.1 .. 0.8 s of each action's dispatched commands (unicycle, 10-ms steps)."""
    sequences = command_sequences(prefix, pulse=pulse)
    dt = TICK_S/SUBSTEPS
    out = np.zeros((len(ACTIONS), HORIZONS, 3))
    for i in range(len(ACTIONS)):
        x = y = yaw = 0.
        for k, (vx, vy, wz) in enumerate(sequences[i]):
            for _ in range(SUBSTEPS):
                mid = yaw+.5*wz*dt
                x += (vx*math.cos(mid)-vy*math.sin(mid))*dt
                y += (vx*math.sin(mid)+vy*math.cos(mid))*dt
                yaw += wz*dt
            out[i, k] = (x, y, yaw)
    return out


def path_rows(prefix, pulse, clearance, position, rotation):
    """The clearance check's per-action rows for the dispatched command paths."""
    neutral = dict(action='hold', candidates=[dict(action=a, utility_m=0.) for a in ACTIONS])
    checked = sel.clearance_check(neutral, nominal_paths(prefix, pulse), clearance, position, rotation)
    return sel.reserve_turns(checked)['memory_forecast_candidates']


def select(goal_body, scan_error, current_clearance_m, rows):
    """The nearest eligible primitive to the feedback command (see the module docstring)."""
    goal = np.asarray(goal_body, float)
    translation = scan_error is None and np.linalg.norm(goal) > 1e-8
    error = math.atan2(goal[1], goal[0]) if scan_error is None else scan_error
    desired = [0., 0., 0.]
    if translation or scan_error is not None:
        desired[0] = MAX_FORWARD*max(0., math.cos(error)) if translation else 0.
        desired[2] = float(np.clip(HEADING_GAIN*error, -MAX_YAW, MAX_YAW))
    clear = {r['action']: bool(r['nominal_predicted_path_clear']) for r in rows}
    candidates = []
    for action in ACTIONS:
        c = command(action)
        allowed = not any(c[:2]) or translation
        candidates.append(dict(action=action, requested_command=list(c), rule_eligible=bool(allowed),
                               nominal_path_clear=clear[action], eligible=bool(allowed and clear[action]),
                               normalized_command_distance_squared=float(((c[0]-desired[0])/MAX_FORWARD)**2
                                                                         + ((c[2]-desired[2])/MAX_YAW)**2)))
    eligible = [i for i, row in enumerate(candidates) if row['eligible']]
    index = (min(eligible, key=lambda i: candidates[i]['normalized_command_distance_squared'])
             if eligible else ACTIONS.index('hold'))
    return dict(action=ACTIONS[index], action_index=index, requested_command=list(command(ACTIONS[index])),
                candidates=candidates, waypoint_body_xy_m=goal.tolist(), scan_heading_error_rad=scan_error,
                desired_instantaneous_command=desired, current_stored_clearance_m=current_clearance_m,
                current_nominal_disk_clear=current_clearance_m is None or current_clearance_m > FOOTPRINT_M+1e-12,
                path_rows=rows, rule='nearest_primitive_among_nominal_path_clear')


def select_v4(goal_body, scan_error, current_clearance_m):
    """The v4 harness's rule: the nearest allowed primitive to the feedback command, all held inside the footprint."""
    goal = np.asarray(goal_body, float)
    clear = current_clearance_m is None or current_clearance_m > FOOTPRINT_M+1e-12
    translation = scan_error is None and np.linalg.norm(goal) > 1e-8
    error = math.atan2(goal[1], goal[0]) if scan_error is None else scan_error
    desired = [0., 0., 0.]
    if clear and (translation or scan_error is not None):
        desired[0] = MAX_FORWARD*max(0., math.cos(error)) if translation else 0.
        desired[2] = float(np.clip(HEADING_GAIN*error, -MAX_YAW, MAX_YAW))
    candidates = []
    for action in ACTIONS:
        c = command(action)
        candidates.append(dict(action=action, requested_command=list(c), eligible=bool(clear and (not any(c[:2]) or translation)),
                               normalized_command_distance_squared=float(((c[0]-desired[0])/MAX_FORWARD)**2
                                                                         + ((c[2]-desired[2])/MAX_YAW)**2)))
    eligible = [i for i, row in enumerate(candidates) if row['eligible']]
    index = (min(eligible, key=lambda i: candidates[i]['normalized_command_distance_squared'])
             if eligible else ACTIONS.index('hold'))
    return dict(action=ACTIONS[index], action_index=index, requested_command=list(command(ACTIONS[index])),
                candidates=candidates, waypoint_body_xy_m=goal.tolist(), scan_heading_error_rad=scan_error,
                desired_instantaneous_command=desired, current_stored_clearance_m=current_clearance_m,
                current_nominal_disk_clear=clear, rule='nearest_six_action_primitive_to_current_waypoint_feedback')


def heading_first_terminal(selection, arrival_radius_m):
    """The terminal rule (see the module docstring); the selection unchanged when it does not apply."""
    if selection['scan_heading_error_rad'] is not None or not selection['current_nominal_disk_clear']:
        return selection
    x, y = selection['waypoint_body_xy_m']
    error = math.atan2(y, x)
    if math.hypot(x, y) <= arrival_radius_m:
        action = 'hold'
    elif abs(error) <= HEADING_TOLERANCE_RAD:
        action = 'forward'
    else:
        action = 'left_turn' if error > 0 else 'right_turn'
    if not next(r for r in selection['candidates'] if r['action'] == action)['eligible']:
        return selection
    result = deepcopy(selection)
    result.update(action=action, action_index=ACTIONS.index(action), requested_command=list(command(action)),
                  rule=HEADING_FIRST_RULE, heading_first_terminal=dict(previous_action=selection['action'], selected_action=action,
                                                                             changed=action != selection['action'],
                                                                             measured_heading_error_rad=error))
    return result
