"""The round-trip mission: reach the goal, then return home, judged only from the registered visual pose.

Each frame (10 Hz) the mission compares the registered position with its active target (the goal, then home). Arrival
is confirmed when the robot has been within the arrival radius for 10 consecutive intervals in which the requested
command was zero and the measured motion was quiet (at most 5 cm/s between frames, at both ends of the interval). A
fast interval, or a requested motion at its start, restarts the count. The robot holds while within the radius, during
the first 3 frames, and once the mission is over.
"""
import numpy as np

WARMUP_FRAMES = 3
ARRIVAL_QUIET_INTERVALS = 10
MAX_QUIET_SPEED_M_S = .05


class RoundTripMission:
    """See the module docstring. `advance` returns this frame's mission state (dict)."""

    def __init__(self, goal_xy, home_xy=(0., 0.), *, arrival_radius_m, navigation_frames, require_return=True):
        self.outbound, self.home = np.asarray(goal_xy, float), np.asarray(home_xy, float)
        self.arrival_radius_m, self.navigation_frames, self.require_return = arrival_radius_m, navigation_frames, require_return
        self.phase, self.terminal, self.frame = 'OUTBOUND', None, -1
        self.quiet, self.was_within = 0, False
        self.previous_position, self.previous_quiet, self.previous_stopped = None, False, False
        self.arrivals = []

    def target(self):
        return (self.outbound if self.phase == 'OUTBOUND' else self.home).copy()

    def advance(self, position, *, frame, requested):
        """`position`: the registered position (initial-body frame); `requested`: the request of the interval just ended."""
        if self.terminal is not None:
            return self.last
        p = np.asarray(position, float)
        command = np.asarray(requested, float)
        # Both interval boundaries must be stopped (zero request) and measured quiet for the dwell to count.
        boundary = self.previous_stopped
        if not boundary:
            self.previous_quiet = False
        if not self.previous_quiet:
            self.quiet, self.was_within = 0, False
        speed = None if self.previous_position is None else float(np.linalg.norm(p-self.previous_position)/.1)
        quiet = speed is not None and speed <= MAX_QUIET_SPEED_M_S
        if not quiet:
            self.quiet, self.was_within = 0, False
        target, phase = self.target(), self.phase
        distance = float(np.linalg.norm(p[:2]-target))
        within = distance <= self.arrival_radius_m
        hold, transition, arrival = frame < WARMUP_FRAMES or within, None, False
        if frame >= WARMUP_FRAMES:
            self.quiet = self.quiet+1 if within and self.was_within and np.array_equal(command, np.zeros(3)) else 0
            if within and self.quiet >= ARRIVAL_QUIET_INTERVALS:
                arrival = True
                self.arrivals.append(dict(phase=phase, frame=frame, distance_m=distance))
                if phase == 'OUTBOUND' and self.require_return:
                    self.phase, self.quiet, transition = 'RETURN', 0, 'OUTBOUND_TO_RETURN'
                else:
                    self.terminal = 'OBSERVED_ROUND_TRIP_CANDIDATE' if self.require_return else 'OBSERVED_GOAL_CANDIDATE'
        if frame >= WARMUP_FRAMES+self.navigation_frames and self.terminal is None:
            self.terminal = 'MISSION_TICK_BUDGET_EXHAUSTED'
        self.was_within = within if transition is None else False
        self.frame = frame
        self.previous_position = p.copy()
        self.previous_quiet = quiet
        self.previous_stopped = bool(np.array_equal(command, np.zeros(3)) and quiet and transition is None)
        self.last = dict(frame=frame, phase=self.phase, phase_transition=transition, evaluated_phase=phase,
                         active_goal_initial_body_xy_m=self.target().tolist(), observed_goal_distance_m=distance,
                         quiet_intervals=self.quiet, hold_required=hold or self.terminal is not None,
                         arrival_confirmed_this_frame=arrival, terminal=self.terminal, observed_interval_speed_m_s=speed)
        return self.last
