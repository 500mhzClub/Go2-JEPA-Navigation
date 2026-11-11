"""The fast ring world (sim.kinematic's scan, perception.ring_walkers, planning.sampling's ring models; PLAN P20)."""
import numpy as np

from go2nav.eval.mission import episode_spec
from go2nav.perception import ring_walkers as rw
from go2nav.perception.range import AZIMUTHS
from go2nav.planning import sampling
from go2nav.sim import kinematic as kin
from go2nav.sim import obstacles as moving


def world(walkers=2):
    spec, episode = episode_spec('validation', 12, 0)
    seed = moving.placement_seed('validation', 12, 0, walkers)
    return kin.World(spec, episode['home_se2_world'], moving.place_obstacles(spec, episode, walkers, seed, polite=False))


def test_the_scan_reads_the_walls_and_a_walker_in_front_of_them():
    w = world()
    ranges, valid = w.scan()
    walls = w.geometry.wall_ranges(w.xy, w.yaw)
    seen = valid & np.isfinite(walls)
    assert seen.mean() > .9 and np.abs(ranges[seen]-walls[seen]).max() < .06        # no walker in sight: the walls, 1 cm of noise
    ahead = rw.mount_of(w.xy, w.yaw)+1.*np.array([np.cos(w.yaw), np.sin(w.yaw)])
    assert w.geometry.wall_distance(ahead) > moving.RADIUS_M                          # the test's premise: free floor 1 m ahead
    w.walkers[0].xy = ahead
    ranges, valid = w.scan()
    front = np.abs(AZIMUTHS).argmin()
    assert abs(ranges[front]-(1.-moving.RADIUS_M)) < .04
    found = rw.detect(ranges, valid, walls, w.xy, w.yaw)
    assert len(found) == 1 and np.linalg.norm(found[0]-ahead) < .02


def test_the_tracker_gives_a_walker_its_velocity_and_forgets_one_seen_gone():
    w = world()
    origin, heading = rw.mount_of(w.xy, w.yaw), np.array([np.cos(w.yaw), np.sin(w.yaw)])
    walls, tracker = w.geometry.wall_ranges(w.xy, w.yaw), rw.Tracker()
    w.walkers[1].xy = np.array([50., 50.])                                            # out of the maze
    for k in range(8):                                                                # walking away at 0.3 m/s
        w.walkers[0].xy, w.frame = origin+(1.+.03*k)*heading, k
        tracker.update(.1*k, rw.detect(*w.scan(), walls, w.xy, w.yaw), origin, w.geometry)
    at, velocity = tracker.state()
    assert len(at) == 1 and np.linalg.norm(at[0]-w.walkers[0].xy) < .02 and np.linalg.norm(velocity[0]-.3*heading) < .05
    w.walkers[0].xy = np.array([50., 50.])
    for k in range(8, 8+rw.EMPTY_FRAMES):
        assert len(tracker.state()[0]) == 1                                           # at rest where last seen, until seen empty
        w.frame = k
        tracker.update(.1*k, rw.detect(*w.scan(), walls, w.xy, w.yaw), origin, w.geometry)
    assert len(tracker.state()[0]) == 0


def test_a_ring_model_plans_from_what_the_ring_showed():
    w = world()
    model = sampling.make_model('ring_velocity')
    origin, heading = rw.mount_of(w.xy, w.yaw), np.array([np.cos(w.yaw), np.sin(w.yaw)])
    w.walkers[1].xy = np.array([50., 50.])
    for k in range(6):
        w.walkers[0].xy, w.frame = origin+(1.+.03*k)*heading, k
        model.observe(.1*k, w.scan(), w.xy, w.yaw, w.geometry)
    paths = model.walkers(None, np.zeros((3, 5, 2)), np.zeros((3, 5)), np.full(3, 5), w.geometry)
    assert paths.shape == (3, 5, 1, 2)
    assert np.linalg.norm(paths[0, 4, 0]-(w.walkers[0].xy+.5*.3*heading)) < .05      # carried on for half a second
    still = sampling.make_model('ring_frozen')
    still.tracker = model.tracker
    assert np.allclose(still.walkers(None, np.zeros((1, 5, 2)), np.zeros((1, 5)), np.full(1, 5), w.geometry)[0, 4, 0],
                       model.tracker.state()[0][0])


def test_a_random_tape_is_random_but_keeps_off_the_walls():
    from go2nav.data import ring_corpus as rc
    from go2nav.planning.fields import Fields
    from go2nav.planning.sampling import FLOOR_M, integrate
    w = world(0)
    fields, start = Fields(w.geometry), dict(xy=w.xy, yaw=w.yaw, executed=w.executed)
    rng = np.random.default_rng(0)
    tapes = [rc.random_tape(rng, 15, start, fields) for _ in range(40)]
    moving_ = [t for t in tapes if np.abs(t).sum()]
    assert 20 < len(moving_) < 40 and len({t.tobytes() for t in moving_}) == len(moving_)
    for tape in moving_:
        xy, yaw = integrate(start, tape[None])
        assert fields.wall_clearance(kin.body_points(xy, yaw)).min() >= FLOOR_M
