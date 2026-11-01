"""Mover labels: a walker's image column from its bearing."""
import math

from go2nav.data import mover_labels as ml
from go2nav.eval import mover_probe as mp


def test_columns_run_left_to_right_and_edge_walkers_take_the_edge_column():
    edge = math.tan(mp.HALF_FOV_RAD)                                  # y/x at the image's edge
    assert ml.column(2., .9*edge*2.) == 0 and ml.column(2., .2*edge*2.) == 1
    assert ml.column(2., -.2*edge*2.) == 2 and ml.column(2., -.9*edge*2.) == 3
    assert ml.column(.2, .5) == 0 and ml.column(.2, -.5) == 3        # centre out of the image, body in view
    assert ml.column(-.05, .3) == 0 and ml.column(-.05, -.3) == 3
