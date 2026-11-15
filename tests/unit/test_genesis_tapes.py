"""P24 stage 2b: the built map as a layout, tapes through the dispatch, and the tapes decision (docs/dynamic/PLAN.md)."""
import numpy as np
import pytest

from go2nav.perception.map_geometry import MapGeometry
from go2nav.perception.mapping import CELL_M, MapSnapshot
from go2nav.planning.dispatch import CommitmentLedger, ScheduledCommand, TapeCommand, dispatch_check

FROZEN = dict(GO2NAV_EXACT='1', GO2NAV_RECOVER='1', GO2NAV_WALKER_PRICE='300', GO2NAV_W_STILL='0')


def corridor(walker_at=None):
    """A map of a corridor along x (walls at |y| = 0.6 m, floor seen between), optionally a walker standing on floor
    seen before."""
    cell = lambda x, y: (int(np.floor(x/CELL_M)), int(np.floor(y/CELL_M)))
    walls = {cell(x, y) for x in np.arange(-1., 3., CELL_M) for y in (-.62, .62)}
    floor = {cell(x, y) for x in np.arange(-1., 3., CELL_M) for y in np.arange(-.55, .56, CELL_M)}
    occupied = set(walls)
    if walker_at is not None:
        occupied |= {cell(walker_at[0]+dx, walker_at[1]+dy) for dx in np.arange(-.15, .16, CELL_M)
                     for dy in np.arange(-.15, .16, CELL_M) if dx*dx+dy*dy <= .15**2}
    return MapSnapshot(frame=0, measured_ns=0, floor=frozenset(floor), occupied=frozenset(occupied),
                       position_map=(0., 0., 0.), map_from_initial=tuple(map(tuple, np.eye(3))), floor_height=0.,
                       primary_current_floor_cells=0, auxiliary_current_floor_cells=0, fine_occupied=frozenset(),
                       current_floor=frozenset(), current_occupied=frozenset(), current_fine_occupied=frozenset())


def test_the_map_s_walls_are_its_occupied_cells_never_seen_as_floor():
    g = MapGeometry(corridor(walker_at=(1.5, 0.)), robot_xy=(0., 0.))
    assert abs(float(g.wall_distance(np.array([[0., 0.]]))[0])-.6) < .05            # the walls 0.6 m to either side
    assert float(g.wall_distance(np.array([[0., .6]]))[0]) < .03
    assert abs(float(g.wall_distance(np.array([[1.5, 0.]]))[0])-.6) < .05           # the walker stands on seen floor
    assert g.visible((0., 0.), (2.5, 0.)) and not g.visible((0., 0.), (0., 1.2))
    assert g.low[0] <= -1. and g.high[0] >= 2.9 and g.low[1] <= -.6 and g.high[1] >= .6 and g.neighbours == {}


def test_a_tape_s_commands_change_each_tick_and_a_plain_plan_s_do_not():
    plain = ScheduledCommand.prepare('forward', observed_ns=0)
    now = plain.dispatch_ns+50_000_000
    assert np.allclose(plain.displacement(now), np.array(plain.command[:2])*(plain.expires_ns-now)/1e9)
    commands = [(.1, 0., 0.), (.2, 0., .1), (.3, 0., 0.), (0., 0., .4)]
    tape = TapeCommand.from_tape(commands, observed_ns=0)
    assert tape.expires_ns-tape.dispatch_ns == 400_000_000 and tape.command == commands[0]
    assert [tape.command_at(tape.dispatch_ns+k*100_000_000+20_000_000) for k in range(4)] == commands
    assert np.allclose(tape.displacement(tape.dispatch_ns), [(.1+.2+.3)*.1, 0.])
    assert tape.request(now_ns=tape.dispatch_ns+150_000_000, allowed=True) == list(commands[1])
    ledger = CommitmentLedger()
    ledger.commit(tape, 0, ((0., 0., 0.),)*3)
    assert ledger.prefix_at(tape.dispatch_ns) == tuple(commands[:3])
    obstacles = type('Obstacles', (), dict(measured_ns=tape.dispatch_ns, cells=frozenset(), returns=(100, 100)))()
    assert dispatch_check(tape, obstacles, now_ns=tape.dispatch_ns)[1] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'


def test_the_planner_without_sideways_commands_turns_onto_a_target_beside_it(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.planning.sampling import Planner
    g = MapGeometry(corridor(), robot_xy=(0., 0.))
    planner = Planner(g, horizon_s=6.4, terminal='walkers', model='frozen', cells=False, lateral=False, arrive_m=.015)
    start = dict(xy=np.array([0., 0.]), yaw=0., executed=np.zeros(3), walkers=[], walkers_xy=np.zeros((0, 2)))
    requested, names = planner.proposals(start, np.array([.1, .25]))
    assert np.abs(requested[..., 1]).max() == 0. and 'hold' in names
    target = requested[names.index('target@0.4')]
    assert target[0, 2] > 0                                                           # it turns left, to the target


def test_the_final_approach_is_never_slower_than_the_go2_executes_and_stops_inside_the_arrival(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.planning.sampling import MIN_APPROACH_M_S, Planner
    g = MapGeometry(corridor(), robot_xy=(0., 0.))
    planner = Planner(g, horizon_s=6.4, terminal='walkers', model='frozen', cells=False, lateral=False, arrive_m=.015)
    start = dict(xy=np.array([0., 0.]), yaw=0., executed=np.zeros(3), walkers=[], walkers_xy=np.zeros((0, 2)))
    near = planner.follow(start, np.array([.03, 0.]), [(None, 0, False, .4)])[0]
    assert np.isclose(near[0, 0], MIN_APPROACH_M_S)                                   # 3 cm out: 0.06 m/s, raised
    inside = planner.follow(start, np.array([.012, 0.]), [(None, 0, False, .4)])[0]
    assert not inside[0].any()                                                        # inside the arrival: hold


def test_the_decision_plans_a_tape_turns_to_look_and_keeps_off_floor_not_yet_seen(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.perception.range import BEAMS
    from go2nav.planning.genesis_tapes import TapesDecision
    decision = TapesDecision('ring_velocity', arrival_radius_m=.02)
    snapshot = corridor()
    decision.bind(snapshot, (0., 0.))
    for t in range(4):
        decision.observe(t*.1, (np.zeros(BEAMS), np.zeros(BEAMS, bool)), (0., 0.), 0., (0., 0., 0.))
    prefix = ((0., 0., 0.),)*3
    commands, record = decision.plan(xy=np.zeros(2), yaw=0., executed=(0., 0., 0.), prefix=prefix,
                                     target=np.array([2., 0.]))
    assert commands.shape == (4, 3) and commands[:, 0].min() > 0 and np.abs(commands[:, 1]).max() == 0.
    assert record['kind'] == 'tape' and record['blocked'] == 0
    look, record = decision.plan(xy=np.zeros(2), yaw=0., executed=(0., 0., 0.), prefix=prefix,
                                 target=np.zeros(2), look=1.)
    assert record['kind'] == 'look' and np.allclose(look[:, 2], .45) and not look[:, :2].any()
    unseen = lambda body, names: [[] if n == 'hold' else [(99, 99)] for n in names]   # every move sweeps unseen floor
    commands, record = decision.plan(xy=np.zeros(2), yaw=0., executed=(0., 0., 0.), prefix=prefix,
                                     target=np.array([2., 0.]), uncovered=unseen)
    assert record['chosen'] == 'hold' and not commands.any() and record['coverage'] == [(99, 99)]


def test_the_decision_needs_the_frozen_rule():
    from go2nav.planning.genesis_tapes import TapesDecision
    decision = TapesDecision('ring_velocity', arrival_radius_m=.02)
    with pytest.raises(RuntimeError):
        decision.bind(corridor(), (0., 0.))
