"""The stand-off driver and the stand-off units (data.lewm_corpus_v2; docs/dynamic/PLAN.md P14 items 3-4)."""
import collections
import math

import numpy as np

from go2nav.data import lewm_corpus_v2 as v2
from go2nav.mazes.sets import MazeSet


def corridor(length=6.):
    """A straight passage 1.3 m wide along x, closed at x = -1 and x = length."""
    boxes = [dict(centre_xyz=[length/2-.5, side*.69, .7], size_xyz=[length+1.1, .08, 1.4], yaw_rad=0.) for side in (1, -1)]
    boxes += [dict(centre_xyz=[x, 0., .7], size_xyz=[.08, 1.46, 1.4], yaw_rad=0.) for x in (-1.04, length+.04)]
    return dict(geometry=dict(wall_boxes=boxes))


def pose_of(x, y, yaw):
    return np.array([x, y, .3, 0., 0., math.sin(yaw/2), math.cos(yaw/2)])


def drive(driver, start, ticks):
    """Integrate the driver's commands kinematically; returns the commands, segment codes and poses."""
    x, y, h = start
    commands, codes, poses = [], [], []
    for _ in range(ticks):
        pose = pose_of(x, y, h)
        command, code = driver.next(pose)
        vx, vy, wz = command
        x, y = x+.1*(vx*math.cos(h)-vy*math.sin(h)), y+.1*(vx*math.sin(h)+vy*math.cos(h))
        h += .1*wz
        commands.append(command), codes.append(code), poses.append((x, y, h))
    return np.array(commands), codes, np.array(poses)


class Retreating(v2.StandoffDriver):
    WEIGHTS = dict(standoff=1.)
    RESOLUTIONS = dict(retreat=1.)
    WAIT_S = (2., 2.0001)


def test_a_stand_off_approaches_waits_then_retreats_without_touching():
    walker = np.array([3., 0.])
    for seed in range(6):
        driver = Retreating(corridor(), np.random.default_rng(seed), movers=lambda: [walker])
        commands, codes, poses = drive(driver, (0., 0., 0.), 300)
        assert set(codes) == {0}                                             # stand-offs only
        still = np.all(np.abs(commands) < 1e-12, axis=1)
        runs = [len(list(g)) for k, g in __import__('itertools').groupby(still) if k]
        assert max(runs) >= 19                                               # the 2-s wait
        first_wait = int(np.argmax(still))
        backing = commands[first_wait:, 0] < 0
        assert backing.any()                                                 # then it backs away
        distance = np.linalg.norm(poses[:, :2]-walker, axis=1)
        assert distance[first_wait+int(np.argmax(backing))+15:].max() > distance[first_wait]+.1
        clearance = [driver.clearance(p[:2]) for p in poses]
        assert min(clearance) >= .40                                         # never at the walls or the walker


def test_the_retreat_steers_back_along_its_own_path():
    driver = Retreating(corridor(), np.random.default_rng(0))
    driver.trail = [np.array([-.5, -.6]), np.array([-.3, -.3]), np.array([0., 0.])]
    gen = driver._resolve_retreat(pose_of(0., 0., 0.))
    command = next(gen)
    assert command[0] < 0 and command[2] > 0                                 # rear swings right, towards the path


def test_stand_off_units_draw_their_walkers():
    spec = MazeSet('validation').maze(10)
    episode = MazeSet('validation').episode(10, 0)
    behaviours, counts = collections.Counter(), collections.Counter()
    for maze in range(300):
        placement, record = v2.unit_obstacles(spec, episode['home_se2_world'], episode['beacon_xy_world'], maze, 'standoff')
        behaviours[record['behaviour']] += 1
        counts[record['count']] += 1
        if placement and record['behaviour'] == 'reactive':
            assert all(o['behaviour'] == 'reactive' for o in placement)
        if placement and record['behaviour'] == 'non_yielding':
            assert all(o.get('polite') is False for o in placement)
    assert set(counts) <= {1, 2, 3}
    assert abs(behaviours['reactive']/300-.4) < .08 and abs(behaviours['polite']/300-.2) < .07
    _, record = v2.unit_obstacles(spec, episode['home_se2_world'], episode['beacon_xy_world'], 3, 'giveway')
    assert 'behaviour' not in record                                         # the other kinds draw as before


def test_kinds_sets_and_seeds():
    assert v2.unit_set('standoff_b') == v2.FAMILY_B_SET and v2.unit_set('standoff') == v2.unit_set('free')
    assert v2.unit_seed(5, 'free') == v2.v1.keyed_seed(v2.VERSION, v2.v1.SET, 5, 'free')   # unchanged for v2's kinds
    assert v2.unit_seed(5, 'standoff') != v2.unit_seed(5, 'standoff_b')
    assert v2.FRAME_CAPS['standoff'] == v2.FRAME_CAPS['standoff_b'] == 900


def test_free_starts_sit_in_a_cell_of_the_layout_s_own_pitch():
    from go2nav.data import lewm_corpus as v1
    from go2nav.mazes import grid
    from go2nav.sim.obstacles import pitch_of
    for spec in (dict(evaluation_layout=dict(pitch_m=1.6), geometry={}), dict(evaluation_layout=dict(pitch_m=1.1), geometry={}),
                 dict(geometry={})):
        for seed in range(20):
            xy = np.asarray(v1.free_spec(spec, seed)['geometry']['spawn_se2_world'][:2])
            cell = np.round(xy/pitch_of(spec))
            assert np.abs(xy-cell*pitch_of(spec)).max() <= min(.15, pitch_of(spec)/2-.49)+1e-9
    tree = v1.free_spec(dict(geometry={}), 7)['geometry']['spawn_se2_world']   # the tree mazes' starts are unchanged
    rng = np.random.default_rng(7)
    cell = np.asarray(grid.SQUARE.cells[int(rng.integers(len(grid.SQUARE.cells)))], float)
    assert np.allclose(tree[:2], cell*grid.PITCH_M+rng.uniform(-.15, .15, 2))


def t_junction():
    """A passage 1.3 m wide along x (x -2 to 4) with a branch to +y at x = 0 (to y = 2.5)."""
    box = lambda cx, cy, sx, sy: dict(centre_xyz=[cx, cy, .7], size_xyz=[sx, sy, 1.4], yaw_rad=0.)
    boxes = [box(1., -.69, 6.1, .08), box(-1.335, .69, 1.37, .08), box(2.335, .69, 3.37, .08),
             box(-.69, 1.6, .08, 1.9), box(.69, 1.6, .08, 1.9), box(0., 2.54, 1.46, .08),
             box(-2.04, 0., .08, 1.46), box(4.04, 0., .08, 1.46)]
    return dict(geometry=dict(wall_boxes=boxes))


class GivingWay(v2.StandoffDriver):
    WEIGHTS = dict(standoff=1., idle=1e-6)            # idle: once out of the walker's sight
    RESOLUTIONS = dict(give_way=1.)
    WAIT_S = (1., 1.0001)

    def _idle(self):
        yield
        for _ in range(10):
            yield (0., 0., 0.)


def test_giving_way_backs_to_the_junction_and_turns_out_of_the_passage():
    walker = np.array([3.2, 0.])
    for seed in range(6):
        driver = GivingWay(t_junction(), np.random.default_rng(seed), movers=lambda: [walker])
        commands, codes, poses = drive(driver, (1.0, 0., 0.), 300)
        assert (commands[:, 0] <= -.249).sum() > 15                          # backed off at 0.25 m/s
        into = poses[:, 1].max()
        assert into > .65 and abs(poses[np.argmax(poses[:, 1]), 0]) < .35     # into the branch, out of the passage
        assert min(driver.clearance(p[:2]) for p in poses) >= .40


def test_a_driver_stuck_at_a_walker_gives_way_but_a_hold_is_left_alone():
    walker = np.array([1.62, 0.])
    driver = GivingWay(t_junction(), np.random.default_rng(0), movers=lambda: [walker])
    driver.trail = [np.array([x, 0.]) for x in np.arange(-.5, 1.01, .1)]
    pose = pose_of(1., 0., 0.)
    driver.recent.extend([np.array([1., 0.])]*driver.STUCK_TICKS)
    driver.commanded.extend([True]*driver.STUCK_TICKS)
    command, _ = driver.next(pose)
    assert command[0] < 0                                                    # it backs off to give way
    held = GivingWay(t_junction(), np.random.default_rng(0), movers=lambda: [walker])
    held.recent.extend([np.array([1., 0.])]*held.STUCK_TICKS)
    held.commanded.extend([False]*held.STUCK_TICKS)
    held.next(pose)
    assert held.unstuck == 0


def test_stuck_three_times_on_one_spot_is_pinned():
    walker = np.array([1.62, 0.])
    driver = GivingWay(t_junction(), np.random.default_rng(0), movers=lambda: [walker])
    pose = pose_of(1., 0., 0.)
    for attempt in range(3):
        assert not driver.pinned
        driver.recent.extend([np.array([1., 0.])]*driver.STUCK_TICKS)
        driver.commanded.extend([True]*driver.STUCK_TICKS)
        driver.next(pose)
    assert driver.pinned and driver.unstuck == 3


def test_cornered_in_a_dead_end_is_pinned_at_the_first_stuck():
    walker = np.array([.32, 0.])                                             # at the mouth of the corridor's closed end
    driver = GivingWay(corridor(), np.random.default_rng(0), movers=lambda: [walker])
    driver.recent.extend([np.array([-.3, 0.])]*driver.STUCK_TICKS)
    driver.commanded.extend([True]*driver.STUCK_TICKS)
    driver.next(pose_of(-.3, 0., 0.))
    assert driver.pinned and driver.unstuck == 1
    open_ = GivingWay(corridor(), np.random.default_rng(0), movers=lambda: [np.array([-.5, 0.])])
    assert not open_._cornered(np.array([.3, 0.]))                           # the way past the walker's back is open
