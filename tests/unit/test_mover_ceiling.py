"""The mover probe's supervised ceiling (go2nav.eval.mover_ceiling)."""
import numpy as np


def test_targets_hold_the_position_then_the_displacements():
    from go2nav.eval.mover_ceiling import targets
    data = dict(xy_t=np.array([[1., 2.]]), target=np.array([[[1.5, 2.], [2., 2.], [3., 1.], [5., 0.]]]))
    assert np.allclose(targets(data, np.array([0])), [[1., 2., .5, 0., 1., 0., 2., -1., 4., -2.]])


def test_the_network_reads_uint8_frame_stacks():
    import torch
    from go2nav.eval.mover_ceiling import HISTORY, network
    net = network(56, 72, 10)
    assert net(torch.zeros(2, HISTORY, 56, 72, 3, dtype=torch.uint8)).shape == (2, 10)
