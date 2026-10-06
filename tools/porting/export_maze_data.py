"""Export the registered maze sets and the exclusion registries from LeWMQuad-v3 into data/mazes (porting tool).

- Sets: dev_tune and validation (capability_v1), prelim_test_v1 (the declassified capability layouts 30-89), the c3v2 and
  c3v3 sets, and the stage-2 sets. Each file is copied byte for byte after its SHA-256 is checked against the original
  registry; a new go2nav registry is written per set. The rigorous-phase `sealed_test_v2` is never opened.
- Exclusion registries (data/mazes/registries):
  - capability_v1_prior_graphs.json: the graphs the capability generator excluded (computed by the original code in
    the frozen snapshot), needed to regenerate the capability sets exactly;
  - built_graphs_2026-10-06.json: every graph built through 6 October 2026 (the capability prior graphs plus every
    registered set above), the exclusion list for any new set.

Usage: GO2NAV_SNAPSHOT=/path/to/snapshot python tools/porting/export_maze_data.py --root <capability root>
       (run with the programme environment; imports the snapshot's code)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/mazes'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy_set(name, description, rows, meta):
    """rows: [(maze_id, role, maze_path, [episode paths], {'maze': sha, episode path: sha})]."""
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=False)
    entries, graphs = [], []
    for maze_id, role, maze_path, episode_paths, expected in rows:
        def copy(path):
            data = Path(path).read_bytes()
            if sha(data) != expected[str(path)]:
                raise ValueError(f'{path} does not match its original registry')
            target = folder/Path(path).name
            target.write_bytes(data)
            return dict(file=target.name, sha256=sha(data), bytes=len(data))
        maze = copy(maze_path)
        spec = json.loads(Path(maze_path).read_text())
        graphs.append(dict(name=spec['scene_id'], edges=spec['evaluation_layout']['edges']))
        entries.append(dict(maze_id=maze_id, role=role, maze=maze, episodes=[copy(p) for p in episode_paths]))
    registry = dict(schema='go2nav_maze_set.v1', name=name, description=description, entries=entries, **meta)
    (folder/'registry.json').write_text(json.dumps(registry, indent=2)+'\n')
    return graphs


def legacy_rows(registry, path_fix=lambda p: p, roles=None):
    rows = []
    for e in registry['entries']:
        if roles and e['role'] not in roles:
            continue
        maze = path_fix(e['maze']['path'])
        episodes = [path_fix(x['path']) for x in e['episodes']]
        expected = {maze: e['maze']['sha256'], **{path_fix(x['path']): x['sha256'] for x in e['episodes']}}
        rows.append((e['maze_id'], e['role'], maze, episodes, expected))
    return rows


def main(root):
    root = Path(root)
    sys.path[:0] = [str(Path(os.environ['GO2NAV_SNAPSHOT']))]
    from scripts import generate_go2_navigation_capability_sets_development as capability  # snapshot code
    (OUT/'registries').mkdir(parents=True, exist_ok=True)
    prior = capability.prior_graphs()
    (OUT/'registries/capability_v1_prior_graphs.json').write_text(json.dumps(prior, indent=1)+'\n')
    graphs = list(prior)
    registry = json.loads((root/'registry.json').read_text())
    declassified = lambda p: p.replace('/sets/sealed_test_v1/', '/sets/prelim_test_v1/')
    capability_meta = dict(family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY', scene_id_prefix='navigation-capability-v1',
                           seeds=dict(construction=2026092501, physics_base=2026112500, appearance_base=2026112600,
                                      episode_base=2026093000), generator_config='configs/mazes/capability_v1.yaml')
    for name, role, description in (
            ('dev_tune', 'dev_tune', 'capability_v1 layouts 0-9: development tuning'),
            ('validation', 'validation', 'capability_v1 layouts 10-29: validation (used for tuning)'),
            ('prelim_test_v1', 'sealed_test', 'capability_v1 layouts 30-89: originally sealed, declassified 1 Oct 2026 '
                                              'for the preliminary run; their specs keep data_role "sealed_test"')):
        graphs += copy_set(name, description, legacy_rows(registry, declassified, roles={role}), capability_meta)
    for name, description, seeds in (
            ('c3v2_sets_v1', 'C3-v2 fresh-check (0-9) and rest/turn recording (10-15) layouts',
             dict(construction=2026092901, physics_base=2026113000, appearance_base=2026113100, episode_base=2026113200)),
            ('c3v3_sets_v1', 'C3-v3 on-policy fit (0-15), held-out (16-21) and safety-check (22-31) layouts',
             dict(construction=2026093001, physics_base=2026113400, appearance_base=2026113500, episode_base=2026113600))):
        graphs += copy_set(name, description, legacy_rows(json.loads((root/name/'registry.json').read_text())),
                           dict(family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY', seeds=seeds,
                                generator_config=f'configs/mazes/{name}.yaml'))
    stage2 = json.loads((root/'stage2_sets_v1_registry.json').read_text())
    graphs += copy_set('stage2_sets_v1', 'stage-2 slip-strip layouts, a selected family (routes with long straights): '
                       'fit 0-23, held-out 24-29, evaluation 30-49', legacy_rows(stage2),
                       dict(family='NAVIGATION_STAGE2_LONG_STRAIGHTS_SELECTED', generator_config='configs/mazes/stage2_sets_v1.yaml'))
    (OUT/'registries/built_graphs_2026-10-06.json').write_text(json.dumps(graphs, indent=1)+'\n')
    print(dict(prior=len(prior), built=len(graphs)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    main(p.parse_args().root)
