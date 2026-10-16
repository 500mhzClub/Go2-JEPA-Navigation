"""Maze generation reproduces the registered sets byte for byte, and the generator's rules hold."""
import random

import pytest

from go2nav.mazes import grid
from go2nav.mazes.cli import generate_from_config
from go2nav.mazes.sets import MazeSet, encode, sha256


def test_spanning_tree_is_a_tree_with_the_start_passage():
    links = grid.random_spanning_tree(random.Random(7))
    assert len(links) == len(grid.CELLS)-1 and grid.edge(grid.START, (0, 0)) in links
    assert grid.farthest_route(links)[0] == grid.START


def test_walls_close_the_boundary():
    walls = grid.wall_boxes(grid.random_spanning_tree(random.Random(1)), wall_id_prefix='w')
    # 4x4 grid: 16 boundary walls, plus one per missing interior passage (24 interior edges - 15 tree edges).
    assert len(walls) == 16+(24-15)


def _shipped_files(set_names):
    files = {}
    for name in set_names:
        maze_set = MazeSet(name)
        for record in maze_set.registry['entries']:
            for r in [record['maze'], *record['episodes']]:
                files[r['file']] = r['sha256']
    return files


@pytest.mark.slow
@pytest.mark.parametrize('config, shipped', [
    ('configs/mazes/c3v2_sets_v1.yaml', ['c3v2_sets_v1']),
    ('configs/mazes/c3v3_sets_v1.yaml', ['c3v3_sets_v1']),
    ('configs/mazes/capability_v1.yaml', ['dev_tune', 'validation', 'prelim_test_v1']),
    ('configs/mazes/stage2_sets_v1.yaml', ['stage2_sets_v1']),
])
def test_regeneration_reproduces_registered_sets(config, shipped):
    _, generated = generate_from_config(config)
    expected = _shipped_files(shipped)
    produced = {}
    for entry in generated['entries']:
        index = entry['spec']['layout_index']
        produced[f'maze_{index:02d}.json'] = sha256(encode(entry['spec']))
        for episode in entry['episodes']:
            produced[f"episode_{index:02d}_{episode['episode_index']}.json"] = sha256(encode(episode))
    assert produced == expected


def test_other_grid_shapes_keep_the_family_rules():
    shape = grid.Shape(5, 4)
    links = grid.random_spanning_tree(random.Random(3), shape)
    assert len(links) == len(shape.cells)-1 == 19 and grid.edge(grid.START, (0, 0)) in links
    assert grid.farthest_route(links, shape)[0] == grid.START
    # 5 x 4 grid: 18 boundary walls, plus one per missing interior passage (31 interior edges - 19 tree edges).
    assert len(grid.wall_boxes(links, wall_id_prefix='w', shape=shape)) == 18+(31-19)
    assert shape.world_bounds_xy_m == ((-2.1, -2.1), (4.7, 3.4))
    assert grid.SQUARE.world_bounds_xy_m == grid.WORLD_BOUNDS_XY_M and grid.SQUARE.cells == grid.CELLS


def test_episode_workers_do_not_change_the_set():
    from go2nav.mazes.generate import Role, Seeds, generate
    kwargs = dict(seeds=Seeds(11, 100, 200, 300), excluded_graphs=[], scene_id_prefix='t', family='T', shape=grid.Shape(5, 4))
    serial = generate([Role('a', 6)], workers=1, **kwargs)
    parallel = generate([Role('a', 6)], workers=3, **kwargs)
    assert encode(serial) == encode(parallel)
    assert serial['entries'][0]['spec']['grid_shape'] == [5, 4]
