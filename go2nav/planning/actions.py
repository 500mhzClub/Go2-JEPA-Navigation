"""The six candidate actions, their command tapes, and the timing of a plan.

A plan is made from a camera frame at time t. The commands already committed for t .. t+300 ms (the **prefix**,
three 100-ms ticks) are known; the chosen action is dispatched at t+300 ms for 400 ms (or 100 ms for a terminal
translation pulse). Forecasts cover eight 100-ms horizons: the prefix, the action's ticks, then zero.

Commands are (forward m/s, lateral m/s, yaw rad/s); lateral is always zero.
"""
import numpy as np

ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
TURNS = {'left_turn': 1, 'right_turn': -1}
PERIOD_NS = 100_000_000
PREFIX_TICKS, COMMIT_TICKS, HORIZONS = 3, 4, 8
COMMAND_SCALE = np.array([.3, 1., .5])  # command normalisation of the learned models


def command(action):
    """The first command of an action's tape (what a plan dispatches)."""
    if action == 'hold':
        return (0., 0., 0.)
    if action == 'forward':
        return (.20, 0., 0.)
    if action in ('left_arc', 'right_arc'):
        return (.16, 0., .45 if action == 'left_arc' else -.45)
    return (0., 0., .45 if action == 'left_turn' else -.45)


def command_sequences(prefix, *, pulse=False):
    """(6, 8, 3): per action, the prefix, then the action for four ticks (one for a translation pulse), then zero."""
    commands = np.zeros((6, HORIZONS, 3), dtype=float)
    commands[:, :PREFIX_TICKS] = np.asarray(prefix, float)
    for i, action in enumerate(ACTIONS):
        count = 1 if pulse and action in TRANSLATIONS else COMMIT_TICKS
        commands[i, PREFIX_TICKS:PREFIX_TICKS+count] = command(action)
    return commands


def movement(tape, past):
    """The movement type of an (8, 3) command tape after the (15, 3) commands of the 1.5 s before it: hold;
    rest_start (forward motion after 1 s without any command); turn (no forward motion); cruise or arc_steady (one
    constant translation, without or with yaw); otherwise switch. The decoder's training mix and C1A use these types."""
    tape, past = np.asarray(tape), np.asarray(past)
    forward, yaw = tape[:, 0] > 0, tape[:, 2] != 0
    if not tape.any():
        return 'hold'
    if not past[-10:].any() and forward.any():
        return 'rest_start'
    if not forward.any():
        return 'turn'
    if np.all(tape == tape[0]):
        return 'arc_steady' if yaw.any() else 'cruise'
    return 'switch'
