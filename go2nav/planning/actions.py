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
