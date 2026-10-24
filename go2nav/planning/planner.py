"""The planner: from the map, the registered pose and a motion forecast, the action to commit for one plan.

One plan is made every fourth frame (2.5 Hz), in three steps:

1. **Route** (`route`). In order: the initial panorama (eight 45° views at the start); the route on the observed map
   (`routing.propose`, clearance-preferred, with the fine goal-cell route) toward the active mission target; a view
   recovery after a dispatch veto (turn 45° toward the vetoed command's yaw side); frontier visits (outbound only:
   approach a viewpoint and turn to see the frontier's unobserved neighbour); a view recovery while visual support is
   low (turn to the supported view's heading, `perception.visual_support`; it has priority over everything above
   and below, and a recovery during the initial panorama ends the panorama); coverage views (see floor that a
   translation was refused for sweeping unobserved).
2. **Objective**. A view heading to turn to; or the look-ahead target on the route (`routing.clear_route_target`) as
   a body-frame waypoint; or, with no route, a scan through five headings. Near the goal (within 10 cm of the exact goal
   point, widened by the committed prefix's travel), the *terminal approach* applies, and translations are 100-ms pulses.
3. **Selection** (`select`). Score the forecast toward the waypoint (`selection.score`), or toward the view heading;
   then the clearance check and the recovery and terminal stages, in the order given in `select`.

`select_reactive` replaces step 3 for the model-free C2 (`planning.reactive`).

`margin_m` and `unknown` (`planning.caution`) make the forecast clearance checks cautious (calibrated-margin
experiment).

`coverage_fix` (default on) counts observed occupied cells as observed in the translation coverage rule
(`selection.require_floor_coverage`); it was off in the experiments before 2 October.

`harness` is the navigation harness version: 'reserve_exit_v2' (the final one: reserve exit, turn exit and C2's
dispatched-path check) or 'v4' (the earlier one, without them; `selection.clearance_check`, `reactive.select_v4`).

`recovery=True` adds the development recovery fixes (the "recovery on" condition of the experiments):
- *terminal burst*: after 15 turn choices in a row within 10 cm of the waypoint, three plans are chosen by position
  progress alone (C2: after 15 heading-first turns the same way, one forward pulse);
- *latch timeout*: a latched turn that makes no heading progress in 10 plans, or flips three times, is released;
- *deadlock escape*: after five holds with nothing clear, a turn whose forecast keeps at least 0.42 m (or the current
  clearance less 3 mm), toward the waypoint side (C2: `_reactive_escape`); while armed, the dispatch check is relaxed
  for it (`NavigationController._escape_dispatch`);
- *stall watchdog*: 30 s within 15 cm while routing to a frontier (outbound) ends the frontier visit and excludes the
  frontier's surroundings (1 m) for 90 s, unless that leaves no route;
- *scripted backup*: five 400-ms reverse steps at 0.10 m/s, on observed floor with 0.40 m of stored clearance. It
  is requested when a deadlock has no clear turn or move, and by the first stall at a place before any exclusion.
  The plan is recorded as a hold carrying the reverse command.

`deadlock_escape=True` (for moving obstacles) adds only the latch timeout and the deadlock escape, without the other
recovery fixes: a robot left just inside the turn clearance of a wall (both turns blocked, a latched turn waiting)
otherwise holds for the rest of the mission (docs/dynamic/STATUS.md, R7c). `terminal_burst=True` adds only the terminal
burst: on the continuous-command policy a turn in place moves the base, so near a waypoint the robot can circle it,
turning for the rest of the mission (docs/dynamic/STATUS.md, C1 refit). `obstacle_prediction=True` checks the
forecast paths against moving tracks (`tracks`, set by the controller each map update) where constant velocity puts
them (`planning.prediction`).
"""
import math

import numpy as np

from go2nav.planning import caution, grid, prediction as predicted, reactive, routing, selection as sel
from go2nav.planning.actions import ACTIONS, TRANSLATIONS, TURNS, command, turn_direction
from go2nav.planning.grid import centre
from go2nav.perception import tracks
from go2nav.perception.rigid import UnportedBranch
from go2nav.perception.visual_support import MAX_VIEW_DISTANCE_M

TERMINAL_APPROACH_RADIUS_M = .10
VIEW_RECOVERY = 'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'
SCAN_OFFSETS = (math.pi/4, math.pi/2, -math.pi/4, -math.pi/2, math.pi)
VETO_VIEW_ANGLE_RAD = math.pi/4
STALL_RADIUS_M, STALL_NS = .15, 30_000_000_000
STALL_EXCLUSION_RADIUS_M, STALL_EXCLUSION_NS = 1.0, 90_000_000_000
LATCH_STALL_DECISIONS, LATCH_MAX_SWITCHES, LATCH_COOLDOWN = 10, 3, 15
ESCAPE_AFTER, ESCAPE_FLOOR_M = 5, .42
TERMINAL_SPIN_TURNS, TERMINAL_BURST_PLANS = 15, 3
REACTIVE_ESCAPE_FLOOR_M, REACTIVE_ESCAPE_GAIN_M, REACTIVE_ESCAPE_S, ESCAPE_SLACK_M = .40, .005, .4, .003
BACKUP_SPEED_M_S, BACKUP_PLANS, BACKUP_STEP_S = .10, 5, .4
BACKUP_FLOOR_M, BACKUP_REPEAT_RADIUS_M = .40, .30
ROUTES = ('OBSERVED_FLOOR_ROUTE_TO_FRONTIER', 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL')
HARNESSES = ('reserve_exit_v2', 'v4')


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class Planner:
    """Planning state and the route, objective and selection steps (see the module docstring)."""

    def __init__(self, *, arrival_radius_m, recovery=False, margin_m=0., unknown=None, harness='reserve_exit_v2',
                 coverage_fix=True, deadlock_escape=False, terminal_burst=False, obstacle_prediction=False,
                 candidates=ACTIONS):
        if harness not in HARNESSES:
            raise ValueError(f'unknown harness {harness!r}')
        self.arrival_radius_m, self.recovery, self.harness = arrival_radius_m, recovery, harness
        self.deadlock_escape, self.terminal_burst_on = deadlock_escape, terminal_burst
        self.obstacle_prediction, self.tracks = obstacle_prediction, []  # set by the controller each map update
        self.candidates = tuple(candidates)  # the forecast candidates (`planning.actions`)
        self.exits, self.coverage_fix = harness == 'reserve_exit_v2', coverage_fix
        self.margin_m, self.unknown = margin_m, unknown  # cautious clearance (planning.caution)
        self.route_status = None
        self.panorama, self.panorama_ended = routing.InitialPanorama(), False
        self.frontier_visits = routing.CommittedFrontierVisits()
        self.coverage_views = routing.CoverageViews()
        self.coverage_target, self.coverage_view_status, self.coverage_generation = None, None, 0
        self.view_recovery = None          # set by a translation veto at dispatch
        self.view_body = None              # the supported view's body-frame position during a visual view recovery
        self.weak_trigger = None           # that recovery's start time (for the route-turn memory)
        self.camera_view_request, self.last_view_trigger = None, -1
        self.scan_target, self.scan_index = None, 0
        self.clearance_turn = None         # the latched long-way turn (selection.recover_turn)
        self.route_turn_memory = sel.RouteTurnMemory()
        self.terminal_approach = self.exact_terminal_target = False
        self.terminal_target_distance = None
        self.pulse = False
        # Recovery fixes.
        self.stall_window = None
        self.latch_track, self.latch_cooldown = None, 0
        self.deadlock_holds, self.escape_armed = 0, None  # armed: 'turn' or 'reactive'
        self.terminal_turns, self.terminal_burst = 0, 0
        self.terminal_spin = (None, 0)
        self.backup, self.backup_request, self.backup_sites = None, None, []
        self.backup_plan = False            # this plan carries the reverse command
        if recovery:
            self.frontier_visits.excluded = routing.StickyExclusions()

    # -- route ------------------------------------------------------------------------------------------------------

    def _base_route(self, snapshot, q, goal_map, phase):
        excluded = self.frontier_visits.excluded if phase == 'OUTBOUND' else ()
        route = routing.propose(snapshot.floor, snapshot.occupied, snapshot.fine_occupied, q[:2], goal_map,
                                excluded_frontiers=excluded)
        route = routing.prefer_clearance(route, snapshot.floor, snapshot.occupied)
        return routing.fine_goal_route(snapshot, q[:2], goal_map, route)

    def _veto_view(self, route, heading, generation):
        recovery = self.view_recovery
        if recovery is None:
            return route
        if recovery['mission_generation'] != generation:
            raise UnportedBranch('veto view recovery across a mission phase change')
        if recovery['target_heading_rad'] is None:
            recovery['target_heading_rad'] = wrap(heading+recovery['view_angle_rad'])
        target = recovery['target_heading_rad']
        if abs(wrap(target-heading)) <= .1:
            self.view_recovery = None
            return route
        self.scan_target = target
        return route | dict(route_cells=[], status='TRANSLATION_VETO_REQUIRES_NEW_VIEW')

    def _frontier_route(self, snapshot, q, Q, goal_map, now_ns, phase, generation):
        route = self._veto_view(self._base_route(snapshot, q, goal_map, phase), math.atan2(Q[1, 0], Q[0, 0]), generation)
        if route['status'] in ('TRANSLATION_VETO_REQUIRES_NEW_VIEW', 'WAITING_FOR_POST_VETO_VIEW'):
            return route
        if phase != 'OUTBOUND':
            self.frontier_visits.visit = None
            return route
        action, visit = self.frontier_visits.advance(snapshot, q[:2], math.atan2(Q[1, 0], Q[0, 0]), goal_map, now_ns, route)
        if action == 'replan':
            return self._veto_view(self._base_route(snapshot, q, goal_map, phase), math.atan2(Q[1, 0], Q[0, 0]), generation)
        if action == 'view':
            return route | dict(route_cells=[], status='FRONTIER_STANDOFF_REQUIRES_VIEW', view_heading_rad=visit['heading_rad'],
                                frontier_visit=visit)
        return route

    def route(self, snapshot, position, rotation, goal, now_ns, *, phase, generation, view=None):
        """The route for a registered pose (initial-body frame) toward the mission's active goal (initial-body xy).
        `view`: the active supported view while visual support is low (`perception.visual_support`), else None."""
        def propose():
            return self._route(snapshot, position, rotation, goal, now_ns, phase=phase, generation=generation, view=view)
        route = propose()
        if self.recovery:
            route = self._stall_watchdog(route, propose, snapshot, np.asarray(snapshot.map_from_initial)@position, now_ns, phase)
        self.route_status = route.get('status')
        return route

    def _route(self, snapshot, position, rotation, goal, now_ns, *, phase, generation, view):
        B = np.asarray(snapshot.map_from_initial)
        q, Q = B@position, B@rotation
        goal_map = (B@np.r_[goal, 0.])[:2]
        self.terminal_approach = self.exact_terminal_target = False
        self.terminal_target_distance = None
        self.frontier_visits.context = (q, Q)
        heading = math.atan2(Q[1, 0], Q[0, 0])
        if view is not None and self.panorama.state is not None and not self.panorama.complete:
            self.panorama_ended = True  # its remaining views stay unobserved
        if self.panorama.complete:
            route = self._frontier_route(snapshot, q, Q, goal_map, now_ns, phase, generation)
        elif self.panorama_ended:
            route = self._frontier_route(snapshot, q, Q, goal_map, now_ns, phase, generation) | dict(
                initial_survey=dict(self.panorama.state))
        else:
            survey = self.panorama.advance(snapshot, q[:2], heading, now_ns)
            if survey['complete']:
                route = self._frontier_route(snapshot, q, Q, goal_map, now_ns, phase, generation) | dict(initial_survey=survey)
            else:
                route = dict(status='INITIAL_PANORAMA_REQUIRES_VIEW', route_cells=[], view_heading_rad=survey['heading_rad'])
        route = self._visual_view(route, view, B, position, rotation)
        route = self._coverage(route, snapshot, q, Q, goal_map, now_ns, generation)
        route = self._interrupted_camera_view(route, view, snapshot, q)
        self.weak_trigger = view['trigger_ns'] if route['status'] == VIEW_RECOVERY else None
        return route

    def _visual_view(self, route, view, B, position, rotation):
        """While visual support is low: turn to the supported view's heading instead of following the route."""
        self.view_body = None
        if view is None:
            return route
        target = B@view['rotation']
        self.clearance_turn = None
        self.view_body = rotation.T@(view['position']-position)
        return route | dict(route_cells=[], view_heading_rad=math.atan2(target[1, 0], target[0, 0]), status=VIEW_RECOVERY)

    def _interrupted_camera_view(self, route, view, snapshot, q):
        """A frontier or coverage view that a new view recovery interrupts is given up
        (`routing.retire_interrupted_view`); the next plan after the recovery chooses another viewpoint."""
        if route['status'] != VIEW_RECOVERY:
            self.camera_view_request = None
            if 'view_heading_rad' in route:
                name = ('coverage_views' if self.coverage_view_status == 'view' else
                        'frontier_visits' if 'frontier_visit' in route else None)
                visit = None if name is None else getattr(self, name).visit
                if visit is not None:
                    self.camera_view_request = (name, visit['started_ns'])
            return route
        if view['trigger_ns'] <= self.last_view_trigger:
            return route
        self.last_view_trigger = view['trigger_ns']
        request, self.camera_view_request = self.camera_view_request, None
        if request is not None:
            name, started = request
            visits = getattr(self, name)
            if visits.visit is not None and visits.visit['started_ns'] == started \
                    and routing.retire_interrupted_view(visits, snapshot, q):
                route = route | dict(interrupted_camera_viewpoints={name: True})
        return route

    def _coverage(self, route, snapshot, q, Q, goal_map, now_ns, generation):
        if self.coverage_generation != generation:
            self.coverage_views.visit, self.coverage_target, self.coverage_generation = None, None, generation
        self.coverage_view_status = None
        target = self.coverage_target
        if target is None:
            return route
        if target in snapshot.floor|snapshot.occupied:
            self.coverage_views.visit, self.coverage_target, self.coverage_view_status = None, None, 'REQUESTED_PATCH_OBSERVED'
            return route
        if route['status'] == VIEW_RECOVERY:
            self.coverage_view_status = 'WEAK_VIEW_RECOVERY_HAS_PRIORITY'
            return route
        query = route | dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', target_map_xy_m=centre(target).tolist())
        self.coverage_views.context = (q, Q)
        if self.coverage_views.visit is None:
            view = self.coverage_views._choose(snapshot, query, q, Q, target)
            if view is None:
                self.coverage_view_status = 'NO_REACHABLE_UNTRIED_COVERAGE_VIEWPOINT'
                return route
            self.coverage_views.visit = dict(unknown_neighbour=list(target), started_ns=now_ns, aligned_ns=None,
                                             heading_rad=view['view_heading_rad'], camera_viewpoint=view)
        status, view = self.coverage_views.advance(snapshot, q[:2], math.atan2(Q[1, 0], Q[0, 0]), goal_map, now_ns, query)
        self.coverage_view_status = status
        if status == 'view':
            self.clearance_turn = None
            return query | dict(route_cells=[], view_heading_rad=view['heading_rad'],
                                status='FOOTPRINT_EXTENSION_REQUIRES_OBSERVED_VIEW')
        if status == 'route':
            query.pop('view_heading_rad', None)
            return query
        return route

    def _stall_watchdog(self, route, propose, snapshot, q, now_ns, phase):
        """The stall fix (see the module docstring). `propose` recomputes the route after an exclusion."""
        visits = self.frontier_visits
        visits.excluded.expire(now_ns)
        if phase != 'OUTBOUND' or route.get('status') != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER':
            self.stall_window = None
            return route
        q = q[:2]
        window = self.stall_window
        if window is None or np.linalg.norm(q-window['start_xy']) > STALL_RADIUS_M:
            self.stall_window = dict(start_ns=now_ns, start_xy=q)
            return route
        if now_ns-window['start_ns'] < STALL_NS:
            return route
        self.stall_window = dict(start_ns=now_ns, start_xy=q)
        if not any(np.linalg.norm(q-site) <= BACKUP_REPEAT_RADIUS_M for site in self.backup_sites):
            self.backup_sites.append(q)
            self._request_backup('stall')
            return route
        target = tuple(route['route_cells'][-1])
        if visits.visit is not None:
            visit = visits.visit
            tried = visits.attempted.setdefault(tuple(visit['unknown_neighbour']), set())
            viewpoint = np.asarray(visit['camera_viewpoint']['viewpoint_map_xy_m'])
            tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-viewpoint) <= .10)
            target = tuple(visit['target_cell'])
            visits._finish('DEV_NO_PROGRESS_STALL')
        cells = {c for c in snapshot.floor if np.linalg.norm(centre(c)-centre(target)) <= STALL_EXCLUSION_RADIUS_M} | {target}
        for cell in cells:
            visits.excluded.sticky[cell] = now_ns+STALL_EXCLUSION_NS
        visits.excluded.update(cells)
        rerouted = propose()
        if rerouted.get('status') not in ROUTES:  # never retire the last reachable frontier
            for cell in cells:
                visits.excluded.sticky.pop(cell, None)
                visits.excluded.discard(cell)
            rerouted = propose()
        return rerouted | dict(stall_reroute=dict(frontier_cell=list(target), excluded_cells=len(cells)))

    # -- objective --------------------------------------------------------------------------------------------------

    def objective(self, route, snapshot, position, rotation, prefix):
        """(goal_body, scan_error) for this plan, or None when the scan headings are exhausted."""
        B = np.asarray(snapshot.map_from_initial)
        Q, q = B@rotation, B@position
        heading = math.atan2(Q[1, 0], Q[0, 0])
        scan_error = None
        if 'view_heading_rad' in route:
            scan_error = wrap(route['view_heading_rad']-heading)
            goal_body = np.zeros(2)
        elif route['route_cells']:
            target = self._route_target(route, snapshot, q[:2])
            goal_body = (Q.T@np.r_[target-q[:2], 0.])[:2]
        else:
            if self.scan_target is None:
                self.scan_target = SCAN_OFFSETS[self.scan_index]
            scan_error = wrap(self.scan_target-heading)
            if abs(scan_error) <= .1:
                self.scan_index += 1
                if self.scan_index >= len(SCAN_OFFSETS):
                    return None
                self.scan_target = SCAN_OFFSETS[self.scan_index]
                scan_error = wrap(self.scan_target-heading)
            goal_body = np.zeros(2)
        radius = TERMINAL_APPROACH_RADIUS_M+.1*float(np.linalg.norm(np.asarray(prefix, float)[:, :2], axis=1).sum())
        if scan_error is None and self.exact_terminal_target:
            self.terminal_approach = self.terminal_target_distance <= radius
        self.pulse = bool(self.terminal_approach and scan_error is None)
        return goal_body, scan_error

    def _route_target(self, route, snapshot, position):
        points = [centre(c) for c in route['route_cells']]
        if route.get('status') == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL' and 'goal_map_xy_m' in route:
            points.append(np.asarray(route['goal_map_xy_m'], float))
        cautious = self._cautious(grid.fine_clearance(snapshot.fine_occupied), snapshot, position)
        target = routing.clear_route_target(points, position, snapshot.fine_occupied, clearance=cautious)
        exact = bool(route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
                     and np.linalg.norm(target-np.asarray(route['goal_map_xy_m'])) < 1e-9)
        self.terminal_approach = bool(exact and np.linalg.norm(target-np.asarray(position)) <= TERMINAL_APPROACH_RADIUS_M)
        self.exact_terminal_target = exact
        self.terminal_target_distance = float(np.linalg.norm(target-np.asarray(position)))
        return target

    # -- selection --------------------------------------------------------------------------------------------------

    def select(self, prediction, goal_body, scan_error, snapshot, position, rotation, *, generation):
        """The action for a forecast (6, 8, 5), in the stages' fixed order (each may revise the previous choice)."""
        B = np.asarray(snapshot.map_from_initial)
        q, Q = B@position, B@rotation
        heading = math.atan2(Q[1, 0], Q[0, 0])
        clearance = grid.fine_clearance(snapshot.fine_occupied)
        terminal = self.terminal_approach
        result = sel.score(prediction, goal_body, self.candidates)
        if self.recovery or self.terminal_burst_on:
            result = self._terminal_position_burst(result, goal_body)
        if scan_error is not None:
            result = sel.scan_override(result, prediction, scan_error)
        checked = self._cautious(clearance, snapshot, q)
        moving = tracks.moving(self.tracks) if self.obstacle_prediction else []
        if moving:
            stored = grid.FineClearance(predicted.without(snapshot.fine_occupied, moving))
            checked = predicted.PredictedClearance(self._cautious(stored, snapshot, q), moving, snapshot.measured_ns)
            plain = sel.clearance_check(result, prediction, self._cautious(clearance, snapshot, q), q, Q, exits=self.exits)
        result = sel.clearance_check(result, prediction, checked, q, Q, exits=self.exits)
        if moving:  # what the check would have chosen without the prediction (a diagnostic only)
            result['unpredicted_check_action'] = plain['action']
        result, self.clearance_turn = sel.recover_turn(result, heading, generation, self.clearance_turn, exits=self.exits)
        if terminal:
            result = sel.terminal_progress(result)
            result = sel.predicted_arrival_hold(result, prediction, self.arrival_radius_m)
        recovered = sel.recover_against_hold(result)
        if recovered is not result:
            self.clearance_turn = None
        result = recovered
        if terminal:
            result = sel.require_arrival_entry(result, prediction, self.arrival_radius_m)
        else:
            result, self.clearance_turn = sel.heading_release(result, self.clearance_turn)
            repositioned = sel.reposition_survey(result)
            if repositioned is not result:
                self.clearance_turn = None
            result = sel.recover_heading_against_hold(repositioned)
            if result is not repositioned:
                self.clearance_turn = None
        result = sel.stopping_projection(result, prediction, clearance, q, Q, pulse=self.pulse)
        if result['planned_stopping_projection']['changed']:
            self.clearance_turn = None
        if self.view_body is not None:
            arc = sel.recover_view_with_arc(result, prediction, self.view_body, MAX_VIEW_DISTANCE_M)
            if arc is not result:
                self.clearance_turn = None
            result = arc
        result, target = sel.require_floor_coverage(result, prediction, snapshot, q, Q, occupied_observed=self.coverage_fix)
        if result.get('translation_footprint_coverage', {}).get('rejected'):
            self.clearance_turn = None
            if self.coverage_target is None and target is not None:
                self.coverage_target, self.coverage_view_status = target, 'TRANSLATION_REQUIRES_COVERAGE'
        if not terminal:
            result = self.route_turn_memory.select(result, q, heading, generation, self.weak_trigger)
        if self.recovery:
            result = self._terminal_spin_guard(result)
            result = self._latch_timeout(result, heading)
            result = self._deadlock_escape(result)
            result = self._scripted_backup(result, snapshot, q, Q)
        elif self.deadlock_escape:
            result = self._latch_timeout(result, heading)
            result = self._deadlock_escape(result)
        if self.obstacle_prediction:
            result = result | dict(obstacle_tracks=[t.summary(snapshot.measured_ns) for t in self.tracks],
                                   predicted_tracks=[t.id for t in moving])
        return result

    def select_reactive(self, goal_body, scan_error, snapshot, position, rotation, prefix):
        """C2's action (`planning.reactive`; no forecast), then the recovery stages that apply to it."""
        B = np.asarray(snapshot.map_from_initial)
        q, Q = B@position, B@rotation
        clearance = grid.fine_clearance(snapshot.fine_occupied)
        if self.harness == 'v4':
            result = reactive.select_v4(goal_body, scan_error, clearance.minimum(q[:2], q[:2]))
        else:
            rows = reactive.path_rows(prefix, self.pulse, self._cautious(clearance, snapshot, q), q, Q)
            result = reactive.select(goal_body, scan_error, clearance.minimum(q[:2], q[:2]), rows)
        if self.pulse:
            result = reactive.heading_first_terminal(result, self.arrival_radius_m)
        if self.recovery:
            result = self._terminal_spin_guard(result)
            result = self._reactive_escape(result, clearance, q, Q)
            result = self._scripted_backup(result, snapshot, q, Q)
        return result

    def _cautious(self, clearance, snapshot, position):
        """The clearance seen by the forecast checks: shifted by the margin, with never-observed cells if pessimistic."""
        if not self.margin_m and self.unknown is None:
            return clearance
        context = self.unknown.context(snapshot, position, self.route_status) if self.unknown is not None else None
        return caution.CautiousClearance(clearance, self.margin_m, context) if self.margin_m or context is not None else clearance

    def _terminal_position_burst(self, result, goal_body):
        """The terminal burst (see the module docstring): the score's heading term dropped for three plans."""
        if not np.any(np.asarray(goal_body, float)) or float(np.linalg.norm(goal_body)) >= TERMINAL_APPROACH_RADIUS_M:
            self.terminal_turns, self.terminal_burst = 0, 0
            return result
        if self.terminal_burst == 0:
            self.terminal_turns = self.terminal_turns+1 if turn_direction(result['action']) else 0
            if self.terminal_turns < TERMINAL_SPIN_TURNS:
                return result
            self.terminal_burst, self.terminal_turns = TERMINAL_BURST_PLANS, 0
        self.terminal_burst -= 1
        for row in result['candidates']:
            row['utility_m'] = row['position_contact_utility_m']
        index = max(range(len(result['candidates'])), key=lambda i: result['candidates'][i]['utility_m'])
        action = result['candidates'][index]['action']
        result.update(action=action, action_index=index, requested_command=command(action), terminal_position_burst=True)
        return result

    def _terminal_spin_guard(self, result):
        """After 15 consecutive heading-first terminal turns the same way, a forward pulse instead (if eligible)."""
        if result.get('rule') != reactive.HEADING_FIRST_RULE or result['action'] not in TURNS:
            self.terminal_spin = (None, 0)
            return result
        direction, count = self.terminal_spin
        count = count+1 if direction == result['action'] else 1
        self.terminal_spin = (result['action'], count)
        forward = next(r for r in result['candidates'] if r['action'] == 'forward')
        if count < TERMINAL_SPIN_TURNS or not forward['eligible']:
            return result
        result.update(action='forward', action_index=ACTIONS.index('forward'), requested_command=list(command('forward')),
                      terminal_spin_break=dict(consecutive_same_direction_turns=count, turn_direction=direction))
        self.terminal_spin = (None, 0)
        return result

    def _reactive_escape(self, result, clearance, q, Q):
        """C2: after five holds with the current position inside the 0.45-m footprint (but at least 0.40 m clear of
        stored obstacles), move away: the translation nearest the feedback command whose 0.4-s path keeps the current
        clearance (3-mm slack) and ends 5 mm clearer, else a turn toward the clearest of 16 points 10 cm around. While
        armed, the dispatch check is relaxed for the escape (`NavigationController._escape_dispatch`)."""
        if result['action'] != 'hold' or result['current_nominal_disk_clear'] is not False:
            self.deadlock_holds, self.escape_armed = 0, None
            return result
        self.deadlock_holds += 1
        current = result['current_stored_clearance_m']
        if self.deadlock_holds < ESCAPE_AFTER or current is None or current < REACTIVE_ESCAPE_FLOOR_M:
            return result
        p, yaw = np.asarray(q, float)[:2], math.atan2(Q[1][0], Q[0][0])
        desired = {r['action']: r['normalized_command_distance_squared'] for r in result['candidates']}
        moves = []
        for action in TRANSLATIONS:
            end = requested_endpoint(p, yaw, command(action), REACTIVE_ESCAPE_S)
            path, final = clearance.minimum(p, end), clearance.minimum(end, end)
            if path is not None and path >= current-ESCAPE_SLACK_M and final >= current+REACTIVE_ESCAPE_GAIN_M:
                moves.append((desired[action], action))
        if moves:
            action = min(moves)[1]
        else:
            headings = np.linspace(-math.pi, math.pi, 16, endpoint=False)
            gains = [clearance.minimum(c, c) for c in (p+.1*np.array([math.cos(yaw+h), math.sin(yaw+h)]) for h in headings)]
            best = int(np.argmax(gains))
            if gains[best] < current+REACTIVE_ESCAPE_GAIN_M:
                self._request_backup('reactive_deadlock_no_move')
                return result
            action = 'left_turn' if headings[best] > 0 else 'right_turn'
        result.update(action=action, action_index=ACTIONS.index(action), requested_command=list(command(action)),
                      reactive_escape=dict(consecutive_holds=self.deadlock_holds, current_clearance_m=current))
        self.escape_armed = 'reactive'
        return result

    def _latch_timeout(self, result, heading):
        latch = self.clearance_turn
        active = (result.get('clearance_turn') or {}).get('active')
        if self.latch_cooldown > 0:
            self.latch_cooldown -= 1
            if active:
                self.clearance_turn, self.latch_track = None, None
                result = self._unlatched(result, 'DEV_LATCH_SUPPRESSED_COOLDOWN')
            return result
        if latch is not None and active:
            remaining = abs(wrap(latch['target_heading_rad']-heading))
            key = (latch['target_heading_rad'], latch['mission_generation'])
            track = self.latch_track
            if track is None or track['key'] != key:
                track = dict(key=key, best=remaining, stalled=0, switches=latch.get('reserve_recovery_direction_switches', 0))
            if remaining < track['best']-.05:
                track.update(best=remaining, stalled=0)
            else:
                track['stalled'] += 1
            switches = latch.get('reserve_recovery_direction_switches', 0)-track['switches']
            if track['stalled'] >= LATCH_STALL_DECISIONS or switches >= LATCH_MAX_SWITCHES:
                self.clearance_turn, self.latch_track, self.latch_cooldown = None, None, LATCH_COOLDOWN
                result = self._unlatched(result, 'DEV_LATCH_TIMEOUT_RELEASED')
            else:
                self.latch_track = track
        return result

    @staticmethod
    def _unlatched(result, event):
        index, _ = sel.best_clear(result)
        sel.choose(result, index)
        result['clearance_turn'] = dict(active=False, event=event)
        return result

    def _deadlock_escape(self, result):
        """The deadlock escape of the forecasting controllers (see the module docstring)."""
        rows = result.get('memory_forecast_candidates')
        if not rows or result['action'] != 'hold':
            self.deadlock_holds, self.escape_armed = 0, None
            return result
        values = sel.utilities(result)
        movable = [r for r in rows if r['action'] != 'hold' and r['action'] in values and r['nominal_predicted_path_clear']]
        if movable or (result.get('clearance_turn') or {}).get('active'):
            self.deadlock_holds, self.escape_armed = 0, None
            return result
        self.deadlock_holds += 1
        if self.deadlock_holds < ESCAPE_AFTER:
            return result
        current = next(r['minimum_predicted_path_clearance_m'] for r in rows if r['action'] == 'hold')
        floor = max(ESCAPE_FLOOR_M, (current or ESCAPE_FLOOR_M)-ESCAPE_SLACK_M)
        options = [i for i, r in enumerate(rows) if turn_direction(r['action']) and r['action'] in values
                   and r['minimum_predicted_path_clearance_m'] is not None and r['minimum_predicted_path_clearance_m'] >= floor]
        if not options:
            self._request_backup('deadlock_no_clear_turn')
            return result
        goal = result.get('waypoint_body_xy_m')
        side = 1 if goal is None or math.atan2(goal[1], goal[0]) > 0 else -1
        index = max(options, key=lambda i: (turn_direction(rows[i]['action']) == side,
                                            rows[i]['minimum_predicted_path_clearance_m']))
        sel.choose(result, index)
        self.escape_armed = 'turn'
        result['deadlock_escape'] = dict(consecutive_holds=self.deadlock_holds, current_clearance_m=current,
                                         clearance_floor_m=floor)
        return result

    def _request_backup(self, reason):
        if self.backup is None and self.backup_request is None:
            self.backup_request = reason

    def _scripted_backup(self, result, snapshot, q, Q):
        """One step of the scripted backup (see the module docstring), or the selection unchanged. A step that would
        leave observed floor, or lacks clearance (0.40 m, and no more than 1 cm below the current), ends the script."""
        self.backup_plan = False
        if self.backup is None and self.backup_request is None:
            return result
        if self.backup is None:
            self.backup, self.backup_request = dict(reason=self.backup_request, step=0), None
        clearance = grid.fine_clearance(snapshot.fine_occupied)
        p, yaw = np.asarray(q, float)[:2], math.atan2(Q[1][0], Q[0][0])
        end = p-BACKUP_SPEED_M_S*BACKUP_STEP_S*np.array([math.cos(yaw), math.sin(yaw)])
        current, path = clearance.minimum(p, p), clearance.minimum(p, end)
        on_floor = all(tuple(c) in snapshot.floor for c in grid.segment_cells(p, end))
        clear = path is None or (path >= BACKUP_FLOOR_M and (current is None or path >= current-.01))
        detail = dict(reason=self.backup['reason'], step=self.backup['step']+1, current_clearance_m=current,
                      path_clearance_m=path, on_observed_floor=on_floor)
        if not (on_floor and clear):
            result['backup_aborted'] = detail
            self.backup = None
            return result
        order = [r['action'] for r in result['candidates']] if 'candidates' in result else ACTIONS  # C2: no candidates
        result.update(action='hold', action_index=order.index('hold'),
                      requested_command=[-BACKUP_SPEED_M_S, 0., 0.],
                      backup=detail)
        self.backup_plan = True
        self.backup['step'] += 1
        if self.backup['step'] >= BACKUP_PLANS:
            self.backup = None
        return result

    def veto(self, plan, now_ns, generation):
        """A dispatched translation was vetoed: look 45° to the side of its yaw (left for straight) unless already."""
        if any(plan.command[:2]) and self.view_recovery is None:
            self.view_recovery = dict(trigger_ns=now_ns, mission_generation=generation, target_heading_rad=None,
                                      view_angle_rad=(-1 if plan.command[2] < 0 else 1)*VETO_VIEW_ANGLE_RAD)


def requested_endpoint(p, yaw, command, seconds, steps=20):
    """Map-frame endpoint of one constant command (unicycle, Euler steps)."""
    x, y, heading = float(p[0]), float(p[1]), yaw
    dt = seconds/steps
    for _ in range(steps):
        x += (command[0]*math.cos(heading)-command[1]*math.sin(heading))*dt
        y += (command[0]*math.sin(heading)+command[1]*math.cos(heading))*dt
        heading += command[2]*dt
    return np.array([x, y])
