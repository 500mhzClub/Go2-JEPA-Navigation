"""The candidate actions, their command tapes, and the timing of a plan.

A plan is made from a camera frame at time t. The commands already committed for t .. t+300 ms (the **prefix**,
three 100-ms ticks) are known; the chosen action is dispatched at t+300 ms for 400 ms (or 100 ms for a terminal
translation pulse). Forecasts cover eight 100-ms horizons: the prefix, the action's ticks, then zero.

Commands are (forward m/s, lateral m/s, yaw rad/s).

**Candidates.** The six primitives are named (`ACTIONS`). A command lattice (`lattice`) names each candidate by its
command (`cmd+0.20+0.00+0.45`), so a name always gives its command (`command`) and category: hold (zero), translation
(any forward or lateral speed; an arc if it also turns) or in-place turn (yaw only), with its direction
(`turn_direction`). The selection rules use these categories, not the names, so they apply to any candidate set. Rules
that take "the" turn one way (latched and recovery turns) use the in-place turn nearest the primitives' yaw rate
(`representative_turn`). Every candidate set contains `hold`.
"""
import numpy as np

ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
TURNS = {'left_turn': 1, 'right_turn': -1}
PERIOD_NS = 100_000_000
PREFIX_TICKS, COMMIT_TICKS, HORIZONS = 3, 4, 8
COMMAND_SCALE = np.array([.3, 1., .5])  # command normalisation of the learned models
# yaw to the primitives' 0.45 rad/s, the rate the 0.45-m footprint was validated at (lv0's ±0.6 clipped walls and
# obstacles); no lateral: both cameras look ahead, so a sideways step enters space no camera watches, where a moving
# obstacle can have walked in (lv0 and lv1's obstacle contacts)
LATTICE_AXES = dict(forward=(-.10, 0., .15, .30, .45), lateral=(0.,), yaw=(-.45, -.3, 0., .3, .45))


def command(action):
    """The command of an action's tape (what a plan dispatches): a primitive's, or a lattice name's."""
    if action.startswith('cmd'):
        return tuple(float(action[i:i+5]) for i in (3, 8, 13))
    if action == 'hold':
        return (0., 0., 0.)
    if action == 'forward':
        return (.20, 0., 0.)
    if action in ('left_arc', 'right_arc'):
        return (.16, 0., .45 if action == 'left_arc' else -.45)
    return (0., 0., .45 if action == 'left_turn' else -.45)


def name(vx, vy, yaw):
    """A lattice candidate's name (zero is `hold`)."""
    return 'hold' if not (vx or vy or yaw) else f'cmd{vx+0.:+.2f}{vy+0.:+.2f}{yaw+0.:+.2f}'  # +0. makes -0. read +0.00


def lattice(forward, lateral, yaw):
    """The candidate names of a command lattice (every combination; `hold` first)."""
    names = [name(float(a), float(b), float(c)) for a in forward for b in lateral for c in yaw]
    return tuple(['hold']+[n for n in names if n != 'hold'])


LATTICE = lattice(**LATTICE_AXES)  # 25 candidates, forward to the continuous policy's trained 0.45 m/s


def translating(action):
    """Whether a candidate (or None) moves the base."""
    return action is not None and any(command(action)[:2])


def turn_direction(action):
    """+1 / -1 for an in-place turn left / right, else 0 (also for None)."""
    if action is None:
        return 0
    vx, vy, yaw = command(action)
    return 0 if vx or vy or not yaw else (1 if yaw > 0 else -1)


def representative_turn(candidates, direction):
    """The in-place turn one way nearest the primitives' yaw rate, or None."""
    turns = [a for a in candidates if turn_direction(a) == direction]
    return min(turns, key=lambda a: (round(abs(abs(command(a)[2])-.45), 6), a)) if turns else None  # ties: the slower


def command_sequences(prefix, *, pulse=False, candidates=ACTIONS):
    """(N, 8, 3): per candidate, the prefix, then its command for four ticks (one for a translation pulse), then
    zero."""
    commands = np.zeros((len(candidates), HORIZONS, 3), dtype=float)
    commands[:, :PREFIX_TICKS] = np.asarray(prefix, float)
    for i, action in enumerate(candidates):
        count = 1 if pulse and translating(action) else COMMIT_TICKS
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
