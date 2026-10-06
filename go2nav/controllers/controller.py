"""The navigation controller: perception, mission, planning and dispatch for one mission.

`on_frame` takes each 10 Hz sensor frame and runs, in order (all synchronous, at the frame's time):
1. visual odometry (`perception.visual_odometry`) and floor registration (`perception.registration`); visual support
   (the frame's corner counts) is checked for the low-support recovery;
2. the mission (`planning.mission`) on the registered position, with the command requested over the last interval;
   while it requires holding, every queued plan is vetoed;
3. every fourth frame: the routing map (`perception.mapping`), then from frame 10 a plan (`planning.planner` with the
   controller's forecaster), committed in the ledger (`planning.dispatch`);
4. the depth obstacles for dispatch (`perception.obstacles`).

`request(now_ns)` is called at every 50 Hz policy tick and returns the command to send: the current plan's command if
it passes the dispatch check, the mission and the committed-prefix check; otherwise zero, with the reason.

With no forecaster the controller is the model-free C2: the planner's reactive selection replaces forecasting and
scoring (`Planner.select_reactive`).
"""
from dataclasses import dataclass

import numpy as np

from go2nav.perception.mapping import RoutingMap
from go2nav.perception.obstacles import DepthObstacleObserver
from go2nav.perception.registration import FloorRegistration
from go2nav.perception.rigid import UnportedBranch
from go2nav.perception.visual_odometry import VisualOdometry
from go2nav.planning import grid
from go2nav.planning.dispatch import PASSED, CommitmentLedger, ScheduledCommand, dispatch_check
from go2nav.planning.mission import RoundTripMission
from go2nav.planning.planner import Planner

MAP_PERIOD_FRAMES = 4
FIRST_PLAN_FRAME = 10
CONTEXT_WINDOW_NS = 1_500_000_000
ESCAPE_DISPATCH_RADIUS_M = .40
POSE_WINDOW_FRAMES = 256
STRONG_FEATURES, LOW_FEATURES = 96, 48


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

    def __init__(self, forecaster, *, goal_xy, home_xy, arrival_radius_m=.02, navigation_frames=4800, recovery=False):
        self.forecaster = forecaster
        self.odometry, self.registration = VisualOdometry(), FloorRegistration()
        self.routing_map, self.obstacles = RoutingMap(), DepthObstacleObserver()
        self.planner = Planner(arrival_radius_m=arrival_radius_m, recovery=recovery)
        self.mission = RoundTripMission(goal_xy, home_xy, arrival_radius_m=arrival_radius_m, navigation_frames=navigation_frames)
        self.ledger = CommitmentLedger()
        self.plans, self.rejected, self.served = [], {}, set()
        self.goal, self.generation = np.asarray(goal_xy, float), 0
        self.mission_state = self.terminal = None
        self.latest_map = self.latest_obstacles = None
        self.policy_packets, self.registered_poses = {}, {}
        self.support_good = None
        self.planning, self.mission_rows, self.poses = [], [], []

    # -- frames ---------------------------------------------------------------------------------------------------

    def on_frame(self, frame):
        self.policy_packets[frame.measured_ns] = frame.policy
        self.policy_packets = {t: p for t, p in self.policy_packets.items() if t >= frame.measured_ns-CONTEXT_WINDOW_NS}
        requested = self.ledger.preceding_request(frame.frame, frame.measured_ns)
        visual = self.odometry.observe(frame.rgb, frame.depth, frame.auxiliary_rgb, frame.auxiliary_depth,
                                       frame.gyro_rates, frame.specific_force, frame.measured_ns)
        registered = self.registration.observe(frame.depth, frame.auxiliary_depth, visual, frame.specific_force)
        self.registered_poses[frame.frame] = registered
        self.registered_poses = {f: p for f, p in self.registered_poses.items() if f > frame.frame-POSE_WINDOW_FRAMES}
        self._visual_support(visual)
        state = self.mission.advance(registered.position, frame=frame.frame, requested=requested)
        self.mission_rows.append(state | dict(measured_ns=frame.measured_ns, preceding_request=list(requested)))
        self.poses.append(dict(frame=frame.frame, visual=visual, registered=registered))
        self.mission_state, self.terminal = state, state['terminal']
        self.goal = np.asarray(state['active_goal_initial_body_xy_m'], float)
        if state['phase_transition'] is not None:
            self.generation += 1
            self.planner.scan_target, self.planner.scan_index = None, 0
        if state['hold_required']:
            for plan in self.plans:
                self.rejected[plan.observed_ns] = 'MISSION_SETTLING_OR_TERMINAL'
                self.ledger.veto(plan, frame.measured_ns)
        if frame.frame%MAP_PERIOD_FRAMES == 0:
            self.latest_map = self.routing_map.update(frame.frame, frame.measured_ns, registered.position, registered.rotation,
                                                      frame.depth, frame.auxiliary_depth, frame.specific_force)
            if frame.frame >= MAP_PERIOD_FRAMES:
                self._plan(frame, registered)
        self.latest_obstacles = self.obstacles.observe(frame.depth, frame.auxiliary_depth, frame.gyro_rates,
                                                       frame.specific_force, frame.measured_ns)

    def _visual_support(self, visual):
        """Remember the last pose with strong corner support; very weak support afterwards would start the visual
        recovery (never reached in the traced missions)."""
        strength = max(visual.feature_counts)
        if strength >= STRONG_FEATURES:
            self.support_good = visual.frame
        if strength < LOW_FEATURES and self.support_good is not None:
            raise UnportedBranch('low visual support recovery')

    def _plan(self, frame, registered):
        now = frame.measured_ns
        if frame.frame < FIRST_PLAN_FRAME:
            self.planning.append(dict(frame=frame.frame, reason='NATIVE_DENSE_CONTEXT_WARMUP'))
            return
        if self.mission_state is None or self.mission_state['hold_required']:
            self.planning.append(dict(frame=frame.frame, reason='MISSION_HOLD', measured_ns=now))
            return
        snapshot, generation = self.latest_map, self.generation
        prefix = self.ledger.prefix_at(now)
        phase = self.mission_state['phase']
        route = self.planner.route(snapshot, registered.position, registered.rotation, self.goal.copy(), now,
                                   phase=phase, generation=generation)
        objective = self.planner.objective(route, snapshot, registered.position, registered.rotation, prefix)
        if objective is None:
            self.planning.append(dict(frame=frame.frame, reason='VIEW_BUDGET_EXHAUSTED'))
            return
        goal_body, scan_error = objective
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
        plan = ScheduledCommand.prepare(selected['action'], observed_ns=now,
                                        pulse=self.planner.pulse and selected['action'] in ('forward', 'left_arc', 'right_arc'))
        self.planning.append(dict(frame=frame.frame, measured_ns=now, map_frame=snapshot.frame, action=selected['action'],
                                  route_status=route['status'], committed_prefix=prefix, selection=selected, route=route))
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
        if self.planner.escape_armed and reason in ('CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'):
            response = self._escape_dispatch(response, plan, now_ns)
        return response

    def _escape_dispatch(self, response, plan, now_ns):
        """While C2's reactive escape is armed, a vetoed escape command is sent anyway if a 0.40-m disk swept to its
        endpoint is clear and the endpoint is no nearer the observed obstacles. The ledger keeps the veto and the zero
        request; only the command sent changes, and the window is no longer latched.

        As in the original, these two tests read the 1-cm body obstacle cells with the 5-cm map cell size (cell (i, j)
        taken as the 5-cm square at 5i, 5j cm), kept so that missions reproduce."""
        current = self.latest_obstacles
        if plan is None or current is None or not any(plan.command):
            return response
        translating = any(plan.command[:2])
        cells = sorted(current.cells)
        start, endpoint = np.zeros(2), np.zeros(2)
        if translating:
            seconds = (plan.expires_ns-now_ns)/1e9
            endpoint = np.array([plan.command[0]*seconds, plan.command[1]*seconds])
        clear = grid.disk_connector(start, endpoint, cells, radius_m=ESCAPE_DISPATCH_RADIUS_M, cell_m=grid.CELL_M)
        away = (not translating
                or grid.nearest_cell_distance(endpoint, cells, grid.CELL_M) >= grid.nearest_cell_distance(start, cells, grid.CELL_M))
        if not clear or not away:
            return response
        self.rejected.pop(plan.observed_ns, None)
        return dict(requested_command=plan.request(now_ns=now_ns, allowed=True), reason=PASSED, escape_dispatch=True)
