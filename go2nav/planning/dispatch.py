"""Plans in time: scheduled commands, the commitment ledger, and the 50 Hz dispatch check.

A plan made from the frame at t commands its action from t+300 ms for 400 ms (a terminal translation pulse: 100 ms).
The **ledger** records every committed plan and every 20-ms request actually sent, so that later forecasts are
conditioned on the commands that will really execute: the *prefix* of a frame is the committed command of each of its
next three 100-ms ticks (zero if none, or if that plan was vetoed before the frame), and a plan is only dispatched if
the prefix it assumed was actually requested.

The **dispatch check** runs at every 50 Hz tick against the latest depth obstacles (1-cm body cells): with
obstacles older than 250 ms the robot holds; otherwise the 0.45-m disk swept by the requested motion until the
plan expires must stay clear, and a translation must also stop clear (the motion until expiry plus the obstacle age
plus 0.5 s). A failed check vetoes the plan for the rest of its window.
"""
from dataclasses import dataclass

import numpy as np

from go2nav.planning import grid
from go2nav.planning.actions import COMMIT_TICKS, PERIOD_NS, PREFIX_TICKS, command

TICK_NS = 20_000_000
MAX_OBSERVATION_AGE_NS = 250_000_000
STOPPING_ALLOWANCE_S = .5
PASSED = 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'


@dataclass(frozen=True)
class ScheduledCommand:
    observed_ns: int
    dispatch_ns: int
    expires_ns: int
    command: tuple

    @classmethod
    def prepare(cls, action, *, observed_ns, pulse=False):
        dispatch = observed_ns+PREFIX_TICKS*PERIOD_NS
        return cls(observed_ns, dispatch, dispatch+(1 if pulse else COMMIT_TICKS)*PERIOD_NS, tuple(command(action)))

    def request(self, *, now_ns, allowed):
        return list(self.command_at(now_ns)) if self.dispatch_ns <= now_ns < self.expires_ns and allowed else [0., 0., 0.]

    def command_at(self, ns):
        """The command the plan asks for at a time in its window (one command for the whole window)."""
        return self.command

    def displacement(self, now_ns, seconds=None):
        """The body-frame displacement (2,) of the commands from now: until the window ends, or for `seconds` at
        the current command."""
        command = self.command_at(now_ns)
        span = (self.expires_ns-now_ns)/1e9 if seconds is None else seconds
        return np.array([command[0]*span, command[1]*span])


@dataclass(frozen=True)
class TapeCommand(ScheduledCommand):
    """A plan whose command changes every 100 ms (PLAN P24 2b: a planner tape's first ticks). `command` is the first
    tick's; `commands` holds one a tick."""
    commands: tuple = ()

    @classmethod
    def from_tape(cls, commands, *, observed_ns):
        dispatch = observed_ns+PREFIX_TICKS*PERIOD_NS
        commands = tuple(tuple(float(v) for v in c) for c in commands)
        return cls(observed_ns, dispatch, dispatch+len(commands)*PERIOD_NS, commands[0], commands)

    def command_at(self, ns):
        tick = min(max((ns-self.dispatch_ns)//PERIOD_NS, 0), len(self.commands)-1)
        return self.commands[int(tick)]

    def displacement(self, now_ns, seconds=None):
        if seconds is not None:
            return super().displacement(now_ns, seconds)
        out, t = np.zeros(2), max(now_ns, self.dispatch_ns)
        while t < self.expires_ns:
            end = min(self.expires_ns, self.dispatch_ns+((t-self.dispatch_ns)//PERIOD_NS+1)*PERIOD_NS)
            out += np.array(self.command_at(t)[:2])*(end-t)/1e9
            t = end
        return out


@dataclass(frozen=True)
class Commitment:
    plan: ScheduledCommand
    completed_ns: int  # when planning finished (the frame time: planning is synchronous)
    prefix: tuple


class CommitmentLedger:
    """Committed plans, vetoes and sent requests (see the module docstring)."""

    def __init__(self):
        self.commitments, self.vetoes, self.requests = [], {}, {}

    def prefix_at(self, observed_ns):
        prefix = []
        for slot in range(PREFIX_TICKS):
            ns = observed_ns+slot*PERIOD_NS
            candidates = [c for c in self.commitments if c.completed_ns <= observed_ns
                          and c.plan.dispatch_ns <= ns < c.plan.expires_ns]
            if len(candidates) > 1:
                raise ValueError('overlapping committed windows')
            value = (0., 0., 0.)
            if candidates:
                veto = self.vetoes.get(candidates[0].plan.observed_ns)
                if veto is None or veto > observed_ns:
                    value = candidates[0].plan.command_at(ns)
            prefix.append(tuple(value))
        return tuple(prefix)

    def commit(self, plan, completed_ns, prefix):
        self.commitments.append(Commitment(plan, completed_ns, tuple(tuple(c) for c in prefix)))

    def veto(self, plan, now_ns):
        self.vetoes.setdefault(plan.observed_ns, now_ns)

    def prefix_was_requested(self, plan):
        matches = [c for c in self.commitments if c.plan.observed_ns == plan.observed_ns]
        if len(matches) != 1:
            return False
        prefix = matches[0].prefix
        return all(self.requests.get(ns) == prefix[(ns-plan.observed_ns)//PERIOD_NS]
                   for ns in range(plan.observed_ns, plan.dispatch_ns, TICK_NS))

    def record_request(self, now_ns, requested):
        self.requests[now_ns] = tuple(requested)
        self.requests = {ns: c for ns, c in self.requests.items() if ns >= now_ns-2_000_000_000}

    def preceding_request(self, frame, frame_ns):
        """The request driving the 100 ms before a frame: the first nonzero of its five requests, else zero."""
        if frame == 0:
            return (0., 0., 0.)
        rows = [self.requests[ns] for ns in range(frame_ns-PERIOD_NS, frame_ns, TICK_NS)]
        return next((row for row in rows if any(row)), (0., 0., 0.))


def _connector_clear(end_xy, cells):
    return grid.disk_connector(np.zeros(2), end_xy, sorted(cells), cell_m=grid.FINE_CELL_M)


def dispatch_check(plan, obstacles, *, now_ns):
    """The command to send now for a plan (or hold), and the reason (see the module docstring). While the forward
    camera has no depth returns, a translating plan is held (the downward camera's obstacles allow turns only)."""
    requested, reason = _obstacle_check(plan, obstacles, now_ns=now_ns)
    if obstacles is not None and obstacles.returns[0] == 0 and plan is not None and any(plan.command_at(now_ns)[:2]):
        return [0., 0., 0.], 'PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO'
    return requested, reason


def _obstacle_check(plan, obstacles, *, now_ns):
    """The swept disk to the commands' endpoint until the window ends, and for a translation the stopping margin at
    the current command (a tape's command changes each 100 ms: `TapeCommand`)."""
    if plan is None:
        return [0., 0., 0.], 'NO_ON_TIME_PLAN'
    if obstacles is None or not 0 <= now_ns-obstacles.measured_ns <= MAX_OBSERVATION_AGE_NS:
        return [0., 0., 0.], 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'
    endpoint = plan.displacement(now_ns)
    allowed = _connector_clear(endpoint, obstacles.cells)
    requested = plan.request(now_ns=now_ns, allowed=allowed)
    if not allowed:
        return requested, 'CURRENT_OBSERVED_OBSTACLE_VETO'
    if any(plan.command_at(now_ns)[:2]):
        horizon = (plan.expires_ns-now_ns)/1e9+(now_ns-obstacles.measured_ns)/1e9+STOPPING_ALLOWANCE_S
        if not _connector_clear(plan.displacement(now_ns, horizon), obstacles.cells):
            return [0., 0., 0.], 'CURRENT_STOPPING_MARGIN_VETO'
    return requested, PASSED
