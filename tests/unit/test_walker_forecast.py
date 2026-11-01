"""The offline walker forecast: following a labelled walker between frames, in the world and the body frame."""
import math

import numpy as np

from go2nav.eval import walker_forecast as wf


def labels(*frames):
    """(frames, columns, 3) labels, one optional (x, y) per frame in column 1."""
    out = np.zeros((len(frames), 4, 3))
    for k, xy in enumerate(frames):
        if xy is not None:
            out[k, 1] = (1., *xy)
    return out


def test_world_and_body_are_inverse():
    pose = np.array([1., -2., .7])
    point = np.array([.3, .4])
    assert np.allclose(wf.body(pose, wf.world(pose, point)), point)


def test_a_turned_robot_sees_the_world_point_where_it_should():
    assert np.allclose(wf.body(np.array([0., 0., math.pi/2]), np.array([0., 1.])), [1., 0.])


def test_a_walker_is_followed_from_t_to_the_horizon():
    n = wf.HISTORY+wf.STEPS
    poses = np.zeros((n, 3))
    walked = [(1.+.03*k, 0.) for k in range(n)]
    rows = wf.tracked(labels(*walked), poses, wf.STEPS)
    assert len(rows) == 1
    now, earlier, there = rows[0]
    assert np.allclose(now, walked[wf.HISTORY-1]) and np.allclose(earlier, walked[0]) and np.allclose(there, walked[-1])


def test_a_walker_gone_at_the_horizon_is_not_followed():
    n = wf.HISTORY+wf.STEPS
    walked = [(1., 0.)]*(n-1)+[None]
    assert not wf.tracked(labels(*walked), np.zeros((n, 3)), wf.STEPS)


def test_a_detection_needs_presence_and_the_summary_counts_only_found_errors():
    out = np.array([[1., .1, .2], [-1., .9, .9]])
    assert np.allclose(wf.detections(out), [[.1*3, .2*3]])
    report = wf.summary([.1, .2, 9.])
    assert abs(report['found']-2/3) < 1e-9 and abs(report['error_m']-.15) < 1e-9
