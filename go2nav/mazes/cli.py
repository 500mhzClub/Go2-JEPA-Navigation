"""Generate a maze set from a config.

    python -m go2nav.mazes.cli configs/mazes/capability_v1.yaml --out outputs/mazes [--name capability_v1]

A config gives the scene-id prefix, family, seeds, the exclusions and the roles (see configs/mazes/). Each `exclude`
entry is either a graph registry (a JSON list of {'name', 'edges'}) or a set folder, whose mazes are all excluded.
`qualify: {strip_coverage: F}` keeps only layouts with an episode whose stage-2 strips (`go2nav.sim.dynamics`) cover
at least F of the route (the stage-2 selection). `grid_shape: [columns, rows]` builds on another grid (default 4 x 4).
For a new set, exclude data/mazes/registries/built_graphs_2026-10-06.json plus any set built since.
"""
import argparse
import json
from pathlib import Path

import yaml

from go2nav.mazes import grid
from go2nav.mazes.generate import Role, Seeds, generate
from go2nav.mazes.sets import MazeSet, write_set

REPO = Path(__file__).resolve().parents[2]


def excluded_graphs(entries):
    graphs = []
    for entry in entries:
        path = REPO/entry
        if path.is_dir():
            maze_set = MazeSet(path.name, root=path.parent)
            graphs += [dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
                       for s in map(maze_set.maze, maze_set.maze_ids())]
        else:
            graphs += json.loads(path.read_text())
    return graphs


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    roles = [Role(name=r['name'], count=r['count'], episodes=tuple(r.get('episodes', (0, 1))),
                  spec_fields=r.get('spec_fields') or {}) for r in config['roles']]
    return config, roles, Seeds(**config['seeds'])


def strip_coverage(minimum):
    """The stage-2 qualification: the episode's strip placement (seeded by role, maze and episode) covers at least
    `minimum` of the route."""
    from go2nav.sim.dynamics import place_strips, placement_seed

    def qualifies(spec, episode):
        seed = placement_seed(spec['data_role'], spec['layout_index'], episode['episode_index'])
        return place_strips(spec, episode, seed)['coverage'] >= minimum
    return qualifies


def generate_from_config(path, qualify=None, workers=None):
    config, roles, seeds = load_config(path)
    if qualify is None and config.get('qualify'):
        qualify = strip_coverage(config['qualify']['strip_coverage'])
    shape = grid.Shape(*config['grid_shape']) if config.get('grid_shape') else grid.SQUARE
    generated = generate(roles, seeds=seeds, excluded_graphs=excluded_graphs(config['exclude']),
                         scene_id_prefix=config['scene_id_prefix'], family=config['family'], qualify=qualify, shape=shape,
                         workers=workers)
    return config, generated


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config')
    p.add_argument('--out', required=True, help='root folder for the new set')
    p.add_argument('--name', help='set name (default: the config file name)')
    p.add_argument('--workers', type=int, help='processes building episodes (default: every core)')
    a = p.parse_args()
    config, generated = generate_from_config(a.config, workers=a.workers)
    name = a.name or Path(a.config).stem
    write_set(a.out, name, generated, description=config['description'], seeds=config['seeds'],
              family=config['family'], scene_id_prefix=config['scene_id_prefix'], roles=config['roles'],
              excluded_registries=config['exclude'])
    print(f"{name}: {len(generated['entries'])} layouts from {generated['candidates_examined']} candidates")


if __name__ == '__main__':
    main()
