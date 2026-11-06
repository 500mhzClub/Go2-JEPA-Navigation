"""The walker memory (P9): the occupancy labels' view test, the baselines' sightings, and the memory's ego-motion warp."""
import math

import numpy as np
import torch

from go2nav.data import memory_labels as ml
from go2nav.eval import memory_baselines as mb
from go2nav.models.memory import BEV_CELLS, WalkerMemory
from go2nav.training.memory import motions

IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])


def pose(x, y, heading):
    return np.array([x, y, 0., 0., 0., math.sin(heading/2), math.cos(heading/2)])


def hot(x, y):
    m = torch.zeros(1, 1, BEV_CELLS, BEV_CELLS)
    half = BEV_CELLS*ml.CELL_M/2
    m[0, 0, int((x+half)//ml.CELL_M), int((y+half)//ml.CELL_M)] = 1.
    return m


def where(m):
    half = BEV_CELLS*ml.CELL_M/2
    i, j = divmod(int(m[0, 0].flatten().argmax()), BEV_CELLS)
    return round((i+.5)*ml.CELL_M-half, 2), round((j+.5)*ml.CELL_M-half, 2)


def test_a_wall_across_the_view_hides_the_cells_beyond_it():
    centres, halves = np.array([[.9, 0.]]), np.array([[.05, 1.]])     # a wall 0.9 m ahead, across the view
    visible = ml.cell_visible(IDENTITY, centres, halves).reshape(ml.CELLS, ml.CELLS)
    x = ml.CENTRES
    middle = ml.CELLS//2
    assert visible[(x > .5) & (x < .8), middle].all()                 # between the camera and the wall
    assert not visible[x > 1.1, middle].any()                         # behind it
    assert not visible[x < 0, :].any()                                # behind the robot


def test_the_view_test_follows_the_robot_s_heading():
    left = ml.cell_visible(pose(0., 0., math.pi/2), np.zeros((0, 2)), np.zeros((0, 2))).reshape(ml.CELLS, ml.CELLS)
    straight = ml.cell_visible(IDENTITY, np.zeros((0, 2)), np.zeros((0, 2))).reshape(ml.CELLS, ml.CELLS)
    assert (left == straight).all()                                   # the grid is in the body frame


def test_the_warp_moves_remembered_cells_against_the_robot_s_motion():
    assert where(WalkerMemory.warp(hot(1.1, .3), torch.tensor([[.2, 0., 0.]]))) == (.9, .3)
    assert where(WalkerMemory.warp(hot(1.1, .3), torch.tensor([[0., .4, 0.]]))) == (1.1, -.1)
    # a quarter turn left puts what was ahead on the right
    assert where(WalkerMemory.warp(hot(1.1, -.1), torch.tensor([[0., 0., math.pi/2]]))) == (-.1, -1.1)


def test_step_motions_are_each_pose_in_the_body_frame_of_the_one_before():
    poses = np.stack([pose(0., 0., 0.), pose(0., 0., 0.), pose(1., 0., math.pi/2), pose(1., 0., math.pi/2),
                      pose(1., 1., math.pi/2)])
    m = motions(poses)                                                # every second frame: poses 0, 2, 4
    assert np.allclose(m[0], 0.)
    assert np.allclose(m[1], [1., 0., math.pi/2], atol=1e-6)
    assert np.allclose(m[2], [1., 0., 0.], atol=1e-6)                 # forward in the turned body frame


def test_average_precision_ranks_and_shares_ties():
    label = np.array([True, False, True, False])
    assert mb.average_precision(np.array([.9, .8, .7, .1]), label) == (1+2/3)/2
    assert mb.average_precision(np.array([1., 1., 1., 1.]), label) == .5   # one tie: the base rate
    assert math.isnan(mb.average_precision(np.ones(3), np.zeros(3, bool)))
