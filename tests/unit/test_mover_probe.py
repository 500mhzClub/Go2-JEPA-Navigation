"""The mover-forecast probe's labels (go2nav.eval.mover_probe)."""
import math

import numpy as np

from go2nav.eval import mover_probe as mp
from go2nav.sim.obstacles import PITCH_M

LAYOUT = dict(evaluation_layout=dict(cells=[[i, 0] for i in range(8)], edges=[[[i, 0], [i+1, 0]] for i in range(7)]))
OPEN = lambda xy: 10.


def pose(x, y, yaw):
    return np.array([x, y, .3, 0., 0., math.sin(yaw/2), math.cos(yaw/2)])


def test_body_frame_and_view():
    p = pose(1., 1., math.pi/2)
    assert np.allclose(mp.body_frame(p, [1., 2.]), [1., 0.]) and np.allclose(mp.body_frame(p, [0., 1.]), [0., 1.])
    movers = [(1., 4.), (1., 2.5), (1., -1.), (3., 1.2)]          # ahead, nearer ahead, behind, off to the side
    assert mp.visible(p, movers, lambda a, b: True) == 1
    assert mp.visible(p, movers, lambda a, b: b[1] > 3.) == 0      # the wall hides the nearer one
    assert mp.visible(p, movers[2:], lambda a, b: True) is None


def test_contrast_tapes_follow_the_rate_limit():
    tape = mp.contrast_tape((.3, 0., 0.), (0., 0., .5), 3)
    assert tape.shape == (3, 5, 3)
    assert np.allclose(tape[:, 0], [[.25, 0., .15], [.3, 0., 0.], [.3, 0., 0.]])
    assert (tape == tape[:, :1]).all()


def walk(placement, robot_path):
    """The walkers stepped against a robot path given per frame (linear between frames), as a session records them."""
    state, recorded = mp.walkers(LAYOUT, placement), []
    for i in range(len(robot_path)):
        recorded.append([w.xy.copy() for w in state])
        if i+1 < len(robot_path):
            for k in range(mp.SUBSTEPS):
                for w in state:
                    w.step(mp.PHYSICS_DT_S, robot_path[i]+(robot_path[i+1]-robot_path[i])*k/mp.SUBSTEPS)
    return np.asarray(recorded)


def placement():
    """A walker at cell 3 whose first leg heads straight for cell 2 (x = 0 side), without a pause."""
    for seed in range(100):
        record = [dict(start_cell=[3, 0], speed_m_s=.3, walk_seed=seed)]
        walker = mp.walkers(LAYOUT, record)[0]
        if walker.target == (2, 0) and not walker.pause_left:
            return record


def test_replay_reproduces_the_recorded_walk_and_flags_a_different_robot():
    poses = np.array([pose(3*PITCH_M-1.5+.005*i, 0., 0.) for i in range(120)])   # in its way, edging closer
    recorded = walk(placement(), poses[:, :2])
    error, kept = mp.replay(LAYOUT, placement(), poses, recorded, keep={50})
    assert error.max() < 1e-9 and np.allclose(kept[50][0].xy, recorded[50, 0])
    error, _ = mp.replay(LAYOUT, placement(), np.array([pose(-5., 0., 0.)]*120), recorded)
    assert error.max() > mp.REPLAY_TOLERANCE_M                         # a robot far away does not stop it


def test_a_polite_mover_responds_to_an_advance_and_a_far_one_does_not():
    state = mp.walkers(LAYOUT, [dict(start_cell=[4, 0], speed_m_s=.3, walk_seed=0)])
    state[0].target, state[0].pause_left, state[0].came_from = (3, 0), 0., (5, 0)    # walking towards x = 0
    hold, advance = (mp.contrast_tape(mp.CONTRAST[k], (0., 0., 0.), 32) for k in ('hold', 'advance'))
    near, far = pose(4*PITCH_M-1.5, 0., 0.), pose(-6., 0., 0.)
    a, b = (mp.branch(state, near, tape, OPEN)[0][:, 0] for tape in (advance, hold))
    assert np.linalg.norm(a[-1]-b[-1]) > mp.EVENT_M                  # it stops sooner for an approaching robot
    a, b = (mp.branch(state, far, tape, OPEN)[0][:, 0] for tape in (advance, hold))
    assert np.allclose(a, b)
    assert np.allclose(state[0].xy, np.asarray((4, 0))*PITCH_M)        # the branch works on a copy


def test_the_kinematic_robot_stops_short_of_walls_and_obstacles():
    from go2nav.sim.obstacles import RADIUS_M, ROBOT_REACH_M
    state = mp.walkers(LAYOUT, [dict(start_cell=[2, 0], speed_m_s=.3, walk_seed=0)])
    state[0].pause_left = 1e9                                          # it stands still at x = 2 pitches
    start, tape = pose(2*PITCH_M-1.5, 0., 0.), mp.contrast_tape((.3, 0., 0.), (.3, 0., 0.), 32)
    _, robot = mp.branch(state, start, tape, OPEN)
    assert robot[0, 0] > start[0]+.1                                    # it advances
    assert RADIUS_M+ROBOT_REACH_M-.01 <= 2*PITCH_M-robot[-1, 0] < RADIUS_M+ROBOT_REACH_M+.02
    _, robot = mp.branch([], start, tape, lambda xy: start[0]+1.-xy[0])   # a wall 1 m ahead
    assert mp.WALL_STAND_OFF_M-.01 <= start[0]+1.-robot[-1, 0] < mp.WALL_STAND_OFF_M+.02


def test_heads_and_the_response_gain():
    import torch
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(2000, 6))).float()
    w = torch.from_numpy(rng.normal(size=(6, 2))).float()
    predict = mp.fit_ridge(x, x@w+1., np.arange(2000) % 40)
    assert torch.allclose(predict(x[:5]), x[:5]@w+1., atol=1e-2)
    true = rng.normal(size=(100, 2))
    assert abs(mp.gain(.5*true, true)-.5) < 1e-9 and mp.gain(np.zeros_like(true), true) == 0.
    noisy = .5*true+.01*rng.normal(size=true.shape)
    low, high = mp.bootstrap(np.arange(100) % 10, lambda i: mp.gain(noisy[i], true[i]))
    assert low < .5 < high


def test_auc_and_the_give_way_test():
    rng = np.random.default_rng(1)
    label = rng.random(400) < .3
    assert mp.auc(label+rng.normal(0, .01, 400), label) == 1. and mp.auc(np.ones(400), label) == .5
    assert abs(mp.auc(rng.random(400), label)-.5) < .1
    n, H = 300, len(mp.HORIZONS)
    xy = np.stack([rng.uniform(.3, 3., n), rng.uniform(-1.5, 1.5, n)], 1)
    hold = np.repeat(xy[:, None], H, 1)
    shift = np.where(rng.random(n) < .4, .3, 0.)[:, None, None]*np.array([1., 0.])  # some movers respond to retreat
    data = dict(xy_t=xy, cf_hold=hold, cf_retreat=hold+shift, cf_advance=hold.copy(), unit=np.arange(n) % 30)
    rows = np.arange(n)
    near = mp.give_way_windows(data, rows)
    assert near.any() and (np.linalg.norm(xy[near], axis=1) < mp.GIVE_WAY_M).all() and (xy[near, 0] > 0).all()
    perfect = mp.give_way(data, rows, dict(hold=hold[:, 1], retreat=(hold+shift)[:, 1], advance=hold[:, 1]), 1)
    assert perfect['windows'] == near.sum() and perfect['retreat']['r2'] == 1. and perfect['retreat']['auc'] == 1.
    blind = mp.give_way(data, rows, dict(hold=hold[:, 1], retreat=hold[:, 1], advance=hold[:, 1]), 1)
    assert blind['retreat']['r2'] == 0. and blind['retreat']['auc'] == .5    # constant velocity: no contrast


def test_give_way_ceiling_track_is_in_the_body_frame_at_t_and_pads_the_start():
    from go2nav.eval import give_way_ceiling as g
    pose = np.zeros((5, 7))
    pose[:, 0] = np.arange(5)*.1
    pose[:, 6] = 1.                                                    # identity quaternion: yaw 0
    recorded = np.zeros((5, 1, 2))
    recorded[:, 0, 0] = 2.
    out = g.track(pose, recorded, 1, 0, 4)                            # frames -2..1: two before the unit's start
    mover, robot = out[:8].reshape(4, 2), out[8:16].reshape(4, 2)
    np.testing.assert_allclose(mover[:, 0], [1.9, 1.9, 1.9, 1.9])    # 2 m minus the robot's 0.1 m at t
    np.testing.assert_allclose(robot[:, 0], [-.1, -.1, -.1, 0.])
    assert out[-1] == .5
