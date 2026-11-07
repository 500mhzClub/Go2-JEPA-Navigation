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
3. every fourth frame: the routing map (`perception.mapping`; with `mover_trails`, the confirmed tracks' trails are
   marked as moving obstacles' after the tracker's update), then from frame 10 a plan (`planning.planner` with the
   controller's forecaster), committed in the ledger (`planning.dispatch`);
4. the depth obstacles for dispatch (`perception.obstacles`).

`request(now_ns)` is called at every 50 Hz policy tick and returns the command to send: the current plan's command if
it passes the dispatch check, the mission and the committed-prefix check; otherwise zero, with the reason.

**Module oracles** (the failure analysis; `controllers.truth`, given the mission's `GroundTruth` as `truth`):
`pose_oracle` replaces step 1 with the true pose, visual odometry running beside it in shadow to log its outages
(`controllers.pose_shadow`). `pose_carry` ('model' or 'c1') carries the frames visual odometry cannot measure
(`perception.carried`) instead of ending the mission; carried frames skip the visual support. `map_oracle` ('obstacles', 'full' or 'static') masks the depth
images the map is built from and replaces the map's cells; `prediction_oracle` gives the planner every obstacle's true
future positions in place of the tracker's; `walker_mode='truth'` gives it every walker's true position, no speeds.

With no forecaster the controller is the model-free C2: the planner's reactive selection replaces forecasting and
scoring (`Planner.select_reactive`).

Each plan is logged in `planning`: the frame and time, the selected action, the route, the selection record and
`forecast_xy_m`, the scored forecast's body-frame positions (6 candidates x 8 ticks of 100 ms; None for C2).
"""
from dataclasses import dataclass, replace

import numpy as np

from go2nav.controllers import truth as truth_oracles
from go2nav.controllers.pose_shadow import VisualShadow
from go2nav.controllers.walkers import ModelTracker
from go2nav.perception.carried import MAX_CARRIED_PATH_M, CarriedOdometry, Carrier
from go2nav.perception.mapping import RoutingMap
from go2nav.perception.obstacles import DepthObstacleObserver
from go2nav.perception.registration import FloorRegistration
from go2nav.perception.tracks import ObstacleTracker
from go2nav.perception.visual_odometry import VisualOdometry
from go2nav.perception.visual_support import SupportedView
from go2nav.planning import grid
from go2nav.planning.actions import ACTIONS, COMMIT_TICKS, PERIOD_NS, translating
from go2nav.planning.dispatch import PASSED, CommitmentLedger, ScheduledCommand, dispatch_check
from go2nav.planning.mission import RoundTripMission
from go2nav.planning.planner import Planner

MAP_PERIOD_FRAMES = 4
FIRST_PLAN_FRAME = 10
FRAME_NS = 100_000_000
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
                 deadlock_escape=False, terminal_burst=False, obstacle_prediction=False, footprint_clearing=False,
                 candidates=ACTIONS, truth=None, pose_oracle=False, map_oracle=None, prediction_oracle=False,
                 mover_trails=False, give_way=False, close_up=False, walker_finder=None, walker_mode='shadow',
                 walker_memory=None, pose_carry=None):
        self.forecaster = forecaster
        self.odometry, self.registration = VisualOdometry(), FloorRegistration()
        if pose_carry is not None:  # 'model' or 'c1' (`perception.carried`)
            self.odometry = CarriedOdometry(Carrier(pose_carry, forecaster), max_path_m=MAX_CARRIED_PATH_M)
        self.pose_carry = pose_carry
        self.routing_map = RoutingMap(clear_seen_free=bool(map_decay), clear_footprint=footprint_clearing,
                                      mover_trails=mover_trails)
        self.obstacles = DepthObstacleObserver()
        self.forget_floor_first = map_decay == 'floor'
        self.planner = Planner(arrival_radius_m=arrival_radius_m, recovery=recovery, margin_m=margin_m, unknown=unknown,
                               harness=harness, coverage_fix=coverage_fix, deadlock_escape=deadlock_escape,
                               terminal_burst=terminal_burst,
                               obstacle_prediction='oracle' if prediction_oracle else obstacle_prediction,
                               candidates=candidates, give_way=give_way, close_up=close_up)
        self.walker_finder, self.walker_mode = walker_finder, walker_mode
        self.walker_memory, self.memory_log = walker_memory, []   # `controllers.memory.MemoryRunner`
        self.model_tracker, self.walker_shadow = ModelTracker(held=walker_mode == 'held') if walker_finder is not None else None, []
        truth_tracks = walker_mode == 'truth'
        if truth_tracks:
            self.model_tracker = ModelTracker(held=True)
        driving = truth_tracks or (walker_finder is not None and walker_mode in ('live', 'held'))  # these replace the hand tracker's
        self.tracker = ObstacleTracker() if obstacle_prediction and not prediction_oracle and not driving else None
        if mover_trails and self.tracker is None and not driving:
            raise ValueError('mover trails need the obstacle tracker (obstacle_prediction)')
        if give_way and self.tracker is None and not driving:
            raise ValueError('the give-way rule needs the obstacle tracker (obstacle_prediction)')
        if (pose_oracle or map_oracle or prediction_oracle or truth_tracks) and truth is None:
            raise ValueError('module oracles need the ground truth')
        self.truth, self.pose_oracle, self.prediction_oracle = truth, pose_oracle, prediction_oracle
        self.pose_shadow = VisualShadow(truth) if pose_oracle else None  # visual odometry beside the true pose
        self.map_oracle = truth_oracles.MapOracle(truth, map_oracle) if map_oracle else None
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
        if self.pose_oracle:
            visual, registered = None, self.truth.registered(frame.frame, frame.measured_ns)
            self.pose_shadow.observe(frame)
        else:
            visual = self.odometry.observe(frame) if self.pose_carry else \
                self.odometry.observe(frame.rgb, frame.depth, frame.auxiliary_rgb, frame.auxiliary_depth,
                                      frame.gyro_rates, frame.specific_force, frame.measured_ns)
            registered = self.registration.observe(frame.depth, frame.auxiliary_depth, visual, frame.specific_force)
        if registered is None:
            state = self.mission.unavailable(frame=frame.frame)
            self.floor_waiting, self.floor_streak = True, 0
            self.floor_plan_minimum_ns = frame.measured_ns+FLOOR_PLAN_DELAY_NS
        else:
            self.registered_poses[frame.frame] = registered
            self.registered_poses = {f: p for f, p in self.registered_poses.items() if f > frame.frame-POSE_WINDOW_FRAMES}
            if visual is not None and visual.camera != 'carried':
                self._visual_support(visual, registered, frame.measured_ns)
            state = self.mission.advance(registered.position, frame=frame.frame, requested=requested)
            if self.walker_memory is not None:
                self.walker_memory.step(frame.frame, frame.policy, registered.position, registered.rotation)
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
            depth, auxiliary_depth = frame.depth, frame.auxiliary_depth
            if self.map_oracle is not None:
                depth, auxiliary_depth = self.map_oracle.mask(depth, auxiliary_depth)
            self.latest_map = self.routing_map.update(frame.frame, frame.measured_ns, registered.position, registered.rotation,
                                                      depth, auxiliary_depth, frame.specific_force)
            if self.map_oracle is not None:
                self.latest_map = self.map_oracle.apply(self.latest_map, registered.position, registered.rotation)
            if self.tracker is not None:
                self.planner.tracks = self.tracker.update(self.latest_map)
                if self.routing_map.mover_trails and self.walker_mode not in ('live', 'held'):
                    self.routing_map.mark_mover(set().union(*(t.trail for t in self.planner.tracks if t.confirmed)))
            if self.prediction_oracle:
                self.planner.tracks = truth_oracles.oracle_tracks(self.truth, self.latest_map, registered.position,
                                                                  registered.rotation)
            if self.walker_memory is not None:
                self._remembered_walkers(frame, registered)
            if self.walker_finder is not None:
                self._model_walkers(frame)
            elif self.walker_mode == 'truth':
                self._true_walkers(frame, registered)
            if self.latest_map.cleared and self.planning and self.planning[-1].get('reason') == 'VIEW_BUDGET_EXHAUSTED':
                self.planner.scan_target, self.planner.scan_index = None, 0  # freed cells: look around again
            if frame.frame >= MAP_PERIOD_FRAMES:
                self._plan(frame, registered)
        self.latest_obstacles = self.obstacles.observe(frame.depth, frame.auxiliary_depth, frame.gyro_rates,
                                                       frame.specific_force, frame.measured_ns)

    def _model_walkers(self, frame):
        """Advance the world model's tracks with this frame's history; live, the planner takes them for its own."""
        packets, first = frame.history, frame.frame-len(frame.history)+1
        registered = [self.registered_poses.get(first+k) for k in range(len(packets))]
        detected = self.walker_finder.detect(packets)
        tracks = self.model_tracker.update(detected, [p['image']['measured_ns'] for p in packets],
                                           [r and (r.position, r.rotation) for r in registered],
                                           self.latest_map.map_from_initial)
        self.walker_shadow.append(dict(frame=frame.frame, measured_ns=frame.measured_ns, detections=detected[-1].round(3).tolist(),
                                       tracks=[t.summary(frame.measured_ns)|dict(confirmed=t.confirmed, fitted=np.round(t.fitted, 3).tolist())
                                               for t in tracks]))
        if self.walker_mode in ('live', 'held'):
            self.planner.tracks = tracks
            if self.routing_map.mover_trails:
                self.routing_map.mark_mover(set().union(*(t.trail for t in tracks if t.confirmed)))

    def _remembered_walkers(self, frame, registered):
        """The walkers the memory remembers out of view, for the planner (`Planner.remembered` or `remembered_walkers`)."""
        remembered = self.walker_memory.remembered(registered.position, registered.rotation, self.latest_map.map_from_initial)
        if self.walker_memory.use == 'veto':
            self.planner.remembered_walkers = tuple(tuple(c[:2]) for c, _ in remembered)
        else:
            self.planner.remembered = self.walker_memory.cells(remembered)
        self.memory_log.append(dict(frame=frame.frame, measured_ns=frame.measured_ns,
                                    walkers=[dict(map_xy_m=np.round(c, 3).tolist(), peak=round(p, 3)) for c, p in remembered]))

    def _true_walkers(self, frame, registered):
        """The tracks of every walker where it truly is, whether or not it can be seen, as the model's would be with
        perfect detections: taken relative to the true pose, placed through the estimated one (`controllers.truth`)."""
        rows = self.truth.walkers_in_body()
        pose = (registered.position, registered.rotation)
        times = [frame.measured_ns-k*FRAME_NS for k in (3, 2, 1, 0)]  # the model's update: four frames, all detected
        self.planner.tracks = self.model_tracker.update([rows]*4, times, [pose]*4, self.latest_map.map_from_initial)
        if self.routing_map.mover_trails:
            self.routing_map.mark_mover(set().union(*(t.trail for t in self.planner.tracks if t.confirmed)))

    def carried_outages(self):
        """The carried outages of visual odometry (`pose_carry`), the one still open last."""
        open_ = self.odometry.open_outage()
        return self.odometry.outages+([open_] if open_ else [])

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
            lattice = {} if self.planner.candidates == ACTIONS else dict(candidates=self.planner.candidates)
            prediction = self.forecaster.forecast(context, prefix, pulse=self.planner.pulse, **lattice)
            selected = self.planner.select(prediction, goal_body, scan_error, snapshot, registered.position,
                                           registered.rotation, generation=generation)
            forecast_xy = np.asarray(prediction)[:, :, :2].tolist()
        plan = ScheduledCommand.prepare(selected['action'], observed_ns=now,
                                        pulse=self.planner.pulse and translating(selected['action']))
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
