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
from go2nav.perception.rigid import UnportedBranch

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
        return list(self.command) if self.dispatch_ns <= now_ns < self.expires_ns and allowed else [0., 0., 0.]


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
                    value = candidates[0].plan.command
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
    """The command to send now for a plan (or hold), and the reason (see the module docstring)."""
    if plan is None:
        return [0., 0., 0.], 'NO_ON_TIME_PLAN'
    if obstacles is None or not 0 <= now_ns-obstacles.measured_ns <= MAX_OBSERVATION_AGE_NS:
        return [0., 0., 0.], 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'
    if obstacles.returns[0] == 0:
        raise UnportedBranch('primary depth unavailable (translation veto)')
    seconds = (plan.expires_ns-now_ns)/1e9
    endpoint = np.array([plan.command[0]*seconds, plan.command[1]*seconds])
    allowed = _connector_clear(endpoint, obstacles.cells)
    requested = plan.request(now_ns=now_ns, allowed=allowed)
    if not allowed:
        return requested, 'CURRENT_OBSERVED_OBSTACLE_VETO'
    if any(plan.command[:2]):
        horizon = (plan.expires_ns-now_ns)/1e9+(now_ns-obstacles.measured_ns)/1e9+STOPPING_ALLOWANCE_S
        endpoint = np.array([plan.command[0]*horizon, plan.command[1]*horizon])
        if not _connector_clear(endpoint, obstacles.cells):
            return [0., 0., 0.], 'CURRENT_STOPPING_MARGIN_VETO'
    return requested, PASSED
