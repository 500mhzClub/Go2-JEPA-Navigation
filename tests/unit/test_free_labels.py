"""The free-space labels: ray distances against wall boxes and walker disks."""
import math

import numpy as np

from go2nav.data import free_labels as fl
from go2nav.sim.obstacles import RADIUS_M

WALL = dict(centre_xyz=[1., 0., .3], size_xyz=[.2, 4., .6], yaw_rad=0.)   # a wall across the x axis, its face at 0.9 m


def rays(origin, angles, discs=None, walls=(WALL,)):
    edges = fl.wall_edges(list(walls)) if walls else (np.zeros((1, 2)), np.zeros((1, 2)))
    return fl.ray_distances(np.array([origin], float), np.array([angles], float), edges,
                            None if discs is None else np.array([discs], float))[0]


def test_a_ray_stops_at_the_wall_face_and_a_ray_away_is_capped():
    d = rays((0., 0.), [0., math.pi])
    assert abs(d[0]-.9) < 1e-9 and d[1] == fl.MAX_M


def test_an_oblique_ray_travels_further_to_the_same_wall():
    assert abs(rays((0., 0.), [math.pi/4])[0]-.9*math.sqrt(2)) < 1e-9


def test_a_turned_wall_is_where_its_yaw_puts_it():
    wall = dict(WALL, centre_xyz=[0., 1., .3], yaw_rad=math.pi/2)
    assert abs(rays((0., 0.), [math.pi/2], walls=[wall])[0]-.9) < 1e-9


def test_a_walker_nearer_than_the_wall_stops_the_ray_and_a_missing_one_does_not():
    assert abs(rays((0., 0.), [0.], [(.5, 0.)])[0]-(.5-RADIUS_M)) < 1e-9
    assert abs(rays((0., 0.), [0.], [(math.nan, math.nan)])[0]-.9) < 1e-9
    assert abs(rays((0., 0.), [0.], [(.5, 1.)])[0]-.9) < 1e-9


def test_a_walker_behind_the_ray_is_ignored():
    assert abs(rays((0., 0.), [0.], [(-.5, 0.)])[0]-.9) < 1e-9
