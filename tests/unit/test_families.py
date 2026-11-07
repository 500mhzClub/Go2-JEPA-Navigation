"""Layout families and the per-layout pitch (mazes.families, docs/dynamic/PLAN.md P13 item 4)."""
import random

import numpy as np
import pytest

from go2nav.mazes import families, grid
from go2nav.mazes.generate import Role, Seeds, generate
from go2nav.sim import obstacles

SEEDS = Seeds(construction=11, physics_base=1000, appearance_base=2000, episode_base=3000)


def connected(links, cells):
    adjacency = {c: set() for c in cells}
    for a, b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, todo = {grid.START}, [grid.START]
    while todo:
        for n in adjacency[todo.pop()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return seen == set(cells)


def test_the_tree_family_draws_exactly_as_the_spanning_tree():
    links, pitch = families.layout(families.TREE, random.Random(5))
    assert links == grid.random_spanning_tree(random.Random(5)) and pitch == grid.PITCH_M


def test_loops_and_rooms():
    shape = grid.SQUARE
    for seed in range(20):
        links, pitch = families.layout(families.LOOPS_ROOMS, random.Random(seed), shape)
        assert pitch in families.LOOPS_ROOMS_PITCHES_M and connected(links, shape.cells)
        assert len(links) >= len(shape.cells)-1+min(families.EXTRA_PASSAGES)       # loops beyond a tree
        rooms = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if families.room(x, y) <= set(links)]
        assert rooms                                                                 # at least one open 2 x 2 room
        assert grid.edge(grid.START, (0, 0)) in links


def test_offices():
    shape = grid.Shape(6, 4)
    for seed in range(20):
        links, pitch = families.layout(families.OFFICES, random.Random(seed), shape)
        assert pitch in families.OFFICES_PITCHES_M and connected(links, shape.cells)
        assert all(grid.edge((x, 0), (x+1, 0)) in links for x in range(-1, 4))     # the corridor
        below = [grid.edge((x, -1), (x, 0)) for x in range(-1, 5)]
        assert 3 <= sum(e in links for e in below) <= 6                              # a door or two per room below
    with pytest.raises(ValueError):
        families.layout(families.OFFICES, random.Random(0), grid.Shape(5, 4))    # an odd number of columns


def test_walls_and_bounds_follow_the_pitch():
    links = grid.random_spanning_tree(random.Random(1))
    narrow, wide = (grid.wall_boxes(links, wall_id_prefix='w', pitch_m=p) for p in (1.1, 1.6))
    assert len(narrow) == len(wide)
    for a, b in zip(narrow, wide):
        assert np.allclose(np.array(a['centre_xyz'][:2])/1.1, np.array(b['centre_xyz'][:2])/1.6)
        assert max(b['size_xyz'][:2]) == pytest.approx(1.6+grid.WALL_THICKNESS_M)
    assert grid.world_bounds() == grid.WORLD_BOUNDS_XY_M
    assert grid.world_bounds(grid.SQUARE, 1.6) == ((-2.4, -2.4), (4.0, 4.0))


@pytest.mark.parametrize('kind, shape', [(families.LOOPS_ROOMS, grid.SQUARE), (families.OFFICES, grid.Shape(6, 4))])
def test_a_family_set_records_its_pitch_and_places_episodes_on_it(kind, shape):
    built = generate([Role(name='validation', count=2, episodes=(0, 1))], seeds=SEEDS, excluded_graphs=[],
                     scene_id_prefix='test', family='TEST', shape=shape, workers=1, layout_family=kind)
    for entry in built['entries']:
        spec = entry['spec']
        pitch = spec['evaluation_layout']['pitch_m']
        assert spec['layout_family'] == kind and spec['evaluation_layout']['metric_code']
        assert max(max(w['size_xyz'][:2]) for w in spec['geometry']['wall_boxes']) == pytest.approx(pitch+grid.WALL_THICKNESS_M)
        for episode in entry['episodes']:
            home = np.array(episode['home_se2_world'][:2])/pitch
            assert np.allclose(home, np.round(home))                                # on a cell centre of this pitch
        placement = obstacles.place_obstacles(spec, entry['episodes'][0], 1, 3)
        field = obstacles.ObstacleField(spec, placement, [None])
        assert np.allclose(field.walkers[0].xy, np.array(placement[0]['start_cell'], float)*pitch)


def test_registered_specifications_keep_their_form():
    built = generate([Role(name='validation', count=1, episodes=(0,))], seeds=SEEDS, excluded_graphs=[],
                     scene_id_prefix='test', family='TEST', workers=1)
    assert 'layout_family' not in built['entries'][0]['spec']
    assert built['entries'][0]['spec']['evaluation_layout']['pitch_m'] == grid.PITCH_M
