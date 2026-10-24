"""The navigation controller: perception, mission, planning and dispatch for one mission.

`on_frame` takes each 10 Hz sensor frame and runs, in order (all synchronous, at the frame's time):
1. visual odometry (`perception.visual_odometry`) and floor registration (`perception.registration`); then the
   visual support (`perception.visual_support`): when a view recovery starts, every plan made before it is vetoed,
   and plans turn to the supported view until it is released;
2. the mission (`planning.mission`) on the registered position, with the command requested over the last interval;
   while it requires holding, every queued plan is vetoed;
   *floor re-acquisition*: a frame whose floor measurement the registration rejects has no pose. The mission holds,
   every queued plan is vetoed, nothing is mapped or planned from the frame, and the robot holds until four
   consecutive frames are registered again; planning resumes no earlier than 400 ms after the rejected frame;
3. every fourth frame: the routing map (`perception.mapping`), then from frame 10 a plan (`planning.planner` with the
   controller's forecaster), committed in the ledger (`planning.dispatch`);
4. the depth obstacles for dispatch (`perception.obstacles`).

`request(now_ns)` is called at every 50 Hz policy tick and returns the command to send: the current plan's command if
it passes the dispatch check, the mission and the committed-prefix check; otherwise zero, with the reason.

With no forecaster the controller is the model-free C2: the planner's reactive selection replaces forecasting and
scoring (`Planner.select_reactive`).

Each plan is logged in `planning`: the frame and time, the selected action, the route, the selection record and
`forecast_xy_m`, the scored forecast's body-frame positions (6 candidates x 8 ticks of 100 ms; None for C2).
"""
from dataclasses import dataclass, replace

import numpy as np

from go2nav.perception.mapping import RoutingMap
from go2nav.perception.obstacles import DepthObstacleObserver
from go2nav.perception.registration import FloorRegistration
from go2nav.perception.visual_odometry import VisualOdometry
from go2nav.perception.visual_support import SupportedView
from go2nav.planning import grid
from go2nav.planning.actions import COMMIT_TICKS, PERIOD_NS
from go2nav.planning.dispatch import PASSED, CommitmentLedger, ScheduledCommand, dispatch_check
from go2nav.planning.mission import RoundTripMission
from go2nav.planning.planner import Planner

MAP_PERIOD_FRAMES = 4
FIRST_PLAN_FRAME = 10
CONTEXT_WINDOW_NS = 1_500_000_000
ESCAPE_DISK_M = dict(turn=.42, reactive=.40)  # by the armed escape (`Planner.escape_armed`)
BACKUP_DISK_M, BACKUP_SLACK_M = .40, .005
VETOES = ('CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED')
POSE_WINDOW_FRAMES = 256
FLOOR_HOLD = 'FLOOR_REACQUISITION_HOLD'
VIEW_RECOVERY_HOLD = 'WAITING_FOR_POST_VISUAL_RECOVERY_PLAN'
FLOOR_STREAK_FRAMES, FLOOR_PLAN_DELAY_NS = 4, 400_000_000
FORGET_UNSEEN_NS = 30_000_000_000  # map clearing on: walls kept when no route is left (a full scan takes about 20 s)


@dataclass
class SensorFrame:
    """One 10 Hz sensor frame, as the robot would deliver it."""
    frame: int
    measured_ns: int
    rgb: np.ndarray                 # primary camera (480, 640, 3) uint8
    depth: tuple                    # primary (depth_m float32, valid bool)
    auxiliary_rgb: np.ndarray       # auxiliary camera
    auxiliary_depth: tuple
    gyro_rates: np.ndarray          # (51, 3) 500 Hz body rates over the last 100 ms
    specific_force: np.ndarray      # (20, 3) 50 Hz accelerometer history
    policy: dict                    # the policy observation packet (images, body sensors, applied commands)
    history: tuple                  # the last four policy packets (t-300 .. t ms)


@dataclass
class PlanContext:
    """What a forecaster may use: the four latest policy packets, those at t-1000, t-500 and t ms, the recent registered
    poses (by frame) and the plans committed so far."""
    frame: int
    measured_ns: int
    history: tuple
    native: tuple
    poses: dict
    commitments: tuple


class NavigationController:
    """See the module docstring."""

    def __init__(self, forecaster, *, goal_xy, home_xy, arrival_radius_m=.02, navigation_frames=4800, recovery=False,
                 margin_m=0., unknown=None, harness='reserve_exit_v2', coverage_fix=True, map_decay=False,
                 deadlock_escape=False, terminal_burst=False):
        self.forecaster = forecaster
        self.odometry, self.registration = VisualOdometry(), FloorRegistration()
        self.routing_map, self.obstacles = RoutingMap(clear_seen_free=bool(map_decay)), DepthObstacleObserver()
        self.forget_floor_first = map_decay == 'floor'
        self.planner = Planner(arrival_radius_m=arrival_radius_m, recovery=recovery, margin_m=margin_m, unknown=unknown,
                               harness=harness, coverage_fix=coverage_fix, deadlock_escape=deadlock_escape,
                               terminal_burst=terminal_burst)
        self.mission = RoundTripMission(goal_xy, home_xy, arrival_radius_m=arrival_radius_m, navigation_frames=navigation_frames)
        self.ledger = CommitmentLedger()
        self.plans, self.rejected, self.served = [], {}, set()
        self.backup_plans = set()  # observation times of the plans carrying a scripted reverse command
        self.goal, self.generation = np.asarray(goal_xy, float), 0
        self.mission_state = self.terminal = None
        self.latest_map = self.latest_obstacles = None
        self.policy_packets, self.registered_poses = {}, {}
        self.supported_view, self.view, self.view_plan_minimum_ns = SupportedView(), None, -1
        self.floor_waiting, self.floor_streak, self.floor_plan_minimum_ns = False, 0, -1
        self.planning, self.mission_rows, self.poses = [], [], []

    # -- frames ---------------------------------------------------------------------------------------------------

    def on_frame(self, frame):
        self.policy_packets[frame.measured_ns] = frame.policy
        self.policy_packets = {t: p for t, p in self.policy_packets.items() if t >= frame.measured_ns-CONTEXT_WINDOW_NS}
        requested = self.ledger.preceding_request(frame.frame, frame.measured_ns)
        visual = self.odometry.observe(frame.rgb, frame.depth, frame.auxiliary_rgb, frame.auxiliary_depth,
                                       frame.gyro_rates, frame.specific_force, frame.measured_ns)
        registered = self.registration.observe(frame.depth, frame.auxiliary_depth, visual, frame.specific_force)
        if registered is None:
            state = self.mission.unavailable(frame=frame.frame)
            self.floor_waiting, self.floor_streak = True, 0
            self.floor_plan_minimum_ns = frame.measured_ns+FLOOR_PLAN_DELAY_NS
        else:
            self.registered_poses[frame.frame] = registered
            self.registered_poses = {f: p for f, p in self.registered_poses.items() if f > frame.frame-POSE_WINDOW_FRAMES}
            self._visual_support(visual, registered, frame.measured_ns)
            state = self.mission.advance(registered.position, frame=frame.frame, requested=requested)
            self.poses.append(dict(frame=frame.frame, visual=visual, registered=registered))
            self.floor_streak += 1
            if self.floor_streak >= FLOOR_STREAK_FRAMES:
                self.floor_waiting = False
        self.mission_rows.append(state | dict(measured_ns=frame.measured_ns, preceding_request=list(requested),
                                              floor_reacquisition_hold=self.floor_waiting))
        self.mission_state, self.terminal = state, state['terminal']
        self.goal = np.asarray(state['active_goal_initial_body_xy_m'], float)
        if state['phase_transition'] is not None:
            self.generation += 1
            self.planner.scan_target, self.planner.scan_index = None, 0
        if self.floor_waiting or state['hold_required']:
            for plan in self.plans:
                self.rejected[plan.observed_ns] = FLOOR_HOLD if self.floor_waiting else 'MISSION_SETTLING_OR_TERMINAL'
                self.ledger.veto(plan, frame.measured_ns)
        if registered is not None and frame.frame%MAP_PERIOD_FRAMES == 0:
            self.latest_map = self.routing_map.update(frame.frame, frame.measured_ns, registered.position, registered.rotation,
                                                      frame.depth, frame.auxiliary_depth, frame.specific_force)
            if self.latest_map.cleared and self.planning and self.planning[-1].get('reason') == 'VIEW_BUDGET_EXHAUSTED':
                self.planner.scan_target, self.planner.scan_index = None, 0  # freed cells: look around again
            if frame.frame >= MAP_PERIOD_FRAMES:
                self._plan(frame, registered)
        self.latest_obstacles = self.obstacles.observe(frame.depth, frame.auxiliary_depth, frame.gyro_rates,
                                                       frame.specific_force, frame.measured_ns)

    def _visual_support(self, visual, registered, now_ns):
        """Advance the view recovery with this frame; when one starts, veto every plan made before it."""
        self.view = self.supported_view.advance(visual.feature_counts, registered.position, registered.rotation, now_ns)
        if self.view is None or self.view['trigger_ns'] <= self.view_plan_minimum_ns:
            return
        self.view_plan_minimum_ns = trigger = self.view['trigger_ns']
        for plan in self.plans:
            if plan.observed_ns < trigger and plan.expires_ns > now_ns:
                self.rejected[plan.observed_ns] = VIEW_RECOVERY_HOLD
                self.ledger.veto(plan, now_ns)

    def _plan(self, frame, registered):
        now = frame.measured_ns
        if frame.frame < FIRST_PLAN_FRAME:
            self.planning.append(dict(frame=frame.frame, reason='NATIVE_DENSE_CONTEXT_WARMUP'))
            return
        if self.floor_waiting or now < self.floor_plan_minimum_ns:
            self.planning.append(dict(frame=frame.frame, reason=FLOOR_HOLD, measured_ns=now))
            return
        if self.mission_state is None or self.mission_state['hold_required']:
            self.planning.append(dict(frame=frame.frame, reason='MISSION_HOLD', measured_ns=now))
            return
        snapshot, generation = self.latest_map, self.generation
        prefix = self.ledger.prefix_at(now)
        phase = self.mission_state['phase']
        route = self.planner.route(snapshot, registered.position, registered.rotation, self.goal.copy(), now,
                                   phase=phase, generation=generation, view=self.view)
        objective = self.planner.objective(route, snapshot, registered.position, registered.rotation, prefix)
        if objective is None:
            forgotten = 0
            if self.routing_map.clear_seen_free:  # no route after a full look around: forget walls not seen lately
                # ('floor': first only cells once seen as floor; every stale cell only when none of those is left)
                if self.forget_floor_first:
                    forgotten = len(self.routing_map.forget(now-FORGET_UNSEEN_NS, floor_only=True))
                if not forgotten:
                    forgotten = len(self.routing_map.forget(now-FORGET_UNSEEN_NS))
                if forgotten:
                    self.planner.scan_target, self.planner.scan_index = None, 0
            self.planning.append(dict(frame=frame.frame, reason='VIEW_BUDGET_EXHAUSTED', forgotten_cells=forgotten))
            return
        goal_body, scan_error = objective
        forecast_xy = None
        if self.forecaster is None:
            selected = self.planner.select_reactive(goal_body, scan_error, snapshot, registered.position, registered.rotation,
                                                    prefix)
        else:
            context = PlanContext(frame.frame, now, frame.history,
                                  tuple(self.policy_packets[now-d] for d in (1_000_000_000, 500_000_000, 0)),
                                  dict(self.registered_poses), tuple(self.ledger.commitments))
            prediction = self.forecaster.forecast(context, prefix, pulse=self.planner.pulse)
            selected = self.planner.select(prediction, goal_body, scan_error, snapshot, registered.position,
                                           registered.rotation, generation=generation)
            forecast_xy = np.asarray(prediction)[:, :, :2].tolist()
        plan = ScheduledCommand.prepare(selected['action'], observed_ns=now,
                                        pulse=self.planner.pulse and selected['action'] in ('forward', 'left_arc', 'right_arc'))
        if self.planner.backup_plan:  # recorded as a hold; a translation, so a pulse in the terminal approach
            plan = replace(plan, command=tuple(selected['requested_command']),
                           expires_ns=plan.dispatch_ns+(1 if self.planner.pulse else COMMIT_TICKS)*PERIOD_NS)
            self.backup_plans.add(plan.observed_ns)
        self.planning.append(dict(frame=frame.frame, measured_ns=now, map_frame=snapshot.frame, action=selected['action'],
                                  route_status=route['status'], committed_prefix=prefix, selection=selected, route=route,
                                  forecast_xy_m=forecast_xy))
        self.ledger.commit(plan, now, prefix)
        self.plans.append(plan)

    # -- 50 Hz dispatch -------------------------------------------------------------------------------------------

    def request(self, now_ns):
        """The command to send at a 50 Hz tick: dict(requested_command, reason)."""
        self.plans = [p for p in self.plans if p.expires_ns > now_ns]
        eligible = [p for p in self.plans if p.dispatch_ns <= now_ns]
        plan = eligible[-1] if eligible else None
        self.rejected = {k: v for k, v in self.rejected.items() if any(p.observed_ns == k for p in self.plans)}
        latched = None if plan is None else self.rejected.get(plan.observed_ns)
        if latched is not None:
            requested, reason = [0., 0., 0.], 'COMMAND_WINDOW_VETO_LATCHED'
        else:
            requested, reason = dispatch_check(plan, self.latest_obstacles, now_ns=now_ns)
            if plan is not None and reason != PASSED:
                self.rejected[plan.observed_ns] = reason
        if self.mission_state is None or self.mission_state['hold_required']:
            requested, reason = [0., 0., 0.], 'MISSION_SETTLING_OR_TERMINAL'
        if self.floor_waiting:
            requested, reason = [0., 0., 0.], FLOOR_HOLD
        if reason == PASSED and plan.observed_ns < self.view_plan_minimum_ns:
            requested, reason = [0., 0., 0.], VIEW_RECOVERY_HOLD
        if plan is not None:
            if not self.ledger.prefix_was_requested(plan):
                requested, reason = [0., 0., 0.], 'COMMITTED_PREFIX_NOT_EXECUTED'
            if reason != PASSED:
                self.rejected[plan.observed_ns] = reason
                self.ledger.veto(plan, now_ns)
        self.ledger.record_request(now_ns, requested)
        if reason in ('CURRENT_STOPPING_MARGIN_VETO', 'CURRENT_OBSERVED_OBSTACLE_VETO') and plan is not None:
            self.planner.veto(plan, now_ns, self.generation)
        response = dict(requested_command=requested, reason=reason)
        if self.planner.escape_armed and reason in VETOES:
            response = self._escape_dispatch(response, plan, now_ns)
        self.backup_plans = {ns for ns in self.backup_plans if any(p.observed_ns == ns for p in self.plans)}
        if response['reason'] in VETOES and plan is not None and plan.observed_ns in self.backup_plans:
            response = self._backup_dispatch(response, plan, now_ns)
        return response

    def _escape_dispatch(self, response, plan, now_ns):
        """While a deadlock escape is armed, a vetoed escape command is sent anyway if a smaller disk (0.42 m for the
        forecasting controllers' turn, 0.40 m for C2's move or turn) swept to its endpoint is clear and the endpoint
        is no nearer the observed obstacles. Only C2's escape may translate. The ledger keeps the veto and the zero
        request; only the command sent changes, and the window is no longer latched.

        As in the original, these tests read the 1-cm body obstacle cells with the 5-cm map cell size (cell (i, j)
        taken as the 5-cm square at 5i, 5j cm), kept so that missions reproduce."""
        armed, current = self.planner.escape_armed, self.latest_obstacles
        if plan is None or current is None or not any(plan.command):
            return response
        translating = any(plan.command[:2])
        if translating and armed != 'reactive':
            return response
        cells = sorted(current.cells)
        start, endpoint = np.zeros(2), np.zeros(2)
        if translating:
            seconds = (plan.expires_ns-now_ns)/1e9
            endpoint = np.array([plan.command[0]*seconds, plan.command[1]*seconds])
        clear = grid.disk_connector(start, endpoint, cells, radius_m=ESCAPE_DISK_M[armed], cell_m=grid.CELL_M)
        away = (not translating
                or grid.nearest_cell_distance(endpoint, cells, grid.CELL_M) >= grid.nearest_cell_distance(start, cells, grid.CELL_M))
        if not clear or not away:
            return response
        self.rejected.pop(plan.observed_ns, None)
        return dict(requested_command=plan.request(now_ns=now_ns, allowed=True), reason=PASSED, escape_dispatch=True)

    def _backup_dispatch(self, response, plan, now_ns):
        """A vetoed scripted-backup command is sent anyway if a 0.40-m disk swept to its endpoint is clear of the
        observed obstacles and the endpoint is no more than 5 mm nearer to them (read as in `_escape_dispatch`)."""
        current = self.latest_obstacles
        if current is None:
            return response
        cells = sorted(current.cells)
        start = np.zeros(2)
        endpoint = np.array([plan.command[0]*(plan.expires_ns-now_ns)/1e9, 0.])
        clear = grid.disk_connector(start, endpoint, cells, radius_m=BACKUP_DISK_M, cell_m=grid.CELL_M)
        away = (grid.nearest_cell_distance(endpoint, cells, grid.CELL_M)
                >= grid.nearest_cell_distance(start, cells, grid.CELL_M)-BACKUP_SLACK_M)
        if not clear or not away:
            return response
        self.rejected.pop(plan.observed_ns, None)
        return dict(requested_command=plan.request(now_ns=now_ns, allowed=True), reason=PASSED, backup_dispatch=True)
