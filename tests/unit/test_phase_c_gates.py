"""Phase C's gates (eval.phase_c_gates; docs/dynamic/PLAN.md P15) and the 12.8-s give-way label."""
import math

import numpy as np

from go2nav.eval import mover_probe as mp
from go2nav.eval import phase_c_gates as g


def pose_of(x, y, yaw):
    return np.array([x, y, .3, 0., 0., math.sin(yaw/2), math.cos(yaw/2)])


def test_the_way_is_clear_unless_a_walker_stands_on_the_way_back():
    start = pose_of(0., 0., 0.)                                   # facing +x; the robot backed off to x = -2.5
    back = np.array([-2.5, 0.])
    assert g.mp.way_clear(np.array([[0.5, 1.2]]), back, start)    # stepped aside, 1.2 m off the way
    assert not mp.way_clear(np.array([[0.6, 0.1]]), back, start)  # still in the way, ahead of where it stood
    assert not mp.way_clear(np.array([[-1.5, 0.3]]), back, start)  # followed it back
    assert mp.way_clear(np.array([[1.7, 0.]]), back, start)        # beyond the metre ahead


def results(auc_val, auc_transfer, horizons=(.4, .8, 1.6, 3.2)):
    rows = lambda auc: [dict(horizon_s=h, give_way=dict(retreat=dict(auc=auc))) for h in horizons]
    return dict(heads=dict(ridge_rollout=dict(val=rows(auc_val), transfer=rows(auc_transfer)),
                           mlp_rollout=dict(val=rows(.5), transfer=rows(.5))))


def test_give_way_passes_only_with_one_readout_on_both_splits_at_both_horizons():
    assert g.give_way_gate({1: results(.85, .81)})['pass_']
    assert not g.give_way_gate({1: results(.85, .79)})['pass_']
    assert g.give_way_gate({1: results(.85, .70), 2: results(.82, .80)})['pass_']


def test_auc_by_split():
    values, label = np.array([.1, .9, .2, .8]), np.array([False, True, True, False])
    split = np.array(['val', 'val', 'transfer', 'transfer'])
    assert g.by_split(values, label, split) == dict(val=1., transfer=0.)


def test_the_side_step_backs_to_the_opening_and_goes_in():
    from go2nav.data import lewm_corpus_v2 as v2
    from tests.unit.test_standoff_driver import t_junction
    geometry = v2.FreeDriverV2(t_junction(), np.random.default_rng(0))
    trail = np.array([[x, 0.] for x in np.arange(1., -1.6, -.05)])          # it came along the passage from x = -1.5
    tape, found = mp.side_step_tape(geometry, pose_of(1., 0., 0.), trail, np.zeros(3, np.float32))
    assert found and tape.shape == (mp.SIDE_FRAMES, 5, 3)
    x, y, h = 1., 0., 0.
    for vx, vy, wz in tape[:, 0]:
        x, y = x+.1*(vx*math.cos(h)-vy*math.sin(h)), y+.1*(vx*math.sin(h)+vy*math.cos(h))
        h += .1*wz
    assert y > .6 and abs(x) < .4                                            # into the branch at x = 0
    assert not tape[-20:].any()                                              # then holds
