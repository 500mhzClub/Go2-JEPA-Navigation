"""Maze sets on disk: one folder per set with maze and episode files and a hash-binding registry.

    <root>/<set>/maze_07.json
    <root>/<set>/episode_07_0.json
    <root>/<set>/registry.json      # set metadata and, per maze, file names with SHA-256

Files are written once (never overwritten) as indented JSON with a trailing newline, the format of every registered set.
Loading verifies each file against the registry, so a run always uses exactly the registered geometry.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHIPPED_ROOT = REPO/'data/mazes'


def encode(value):
    return (json.dumps(value, indent=2)+'\n').encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, value):
    data = encode(value)
    with path.open('xb') as stream:
        stream.write(data)
    return dict(file=path.name, sha256=sha256(data), bytes=len(data))


def write_set(root, name, generated, *, description, seeds, family, scene_id_prefix, roles, excluded_registries):
    """Write a generated set (`generate.generate` output) and its registry. The set folder must not exist."""
    folder = Path(root)/name
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for entry in generated['entries']:
        spec = entry['spec']
        index = spec['layout_index']
        row = dict(maze_id=index, role=spec['data_role'], maze=_write(folder/f'maze_{index:02d}.json', spec), episodes=[])
        for episode in entry['episodes']:
            row['episodes'].append(_write(folder/f"episode_{index:02d}_{episode['episode_index']}.json", episode))
        entries.append(row)
    registry = dict(schema='go2nav_maze_set.v1', name=name, description=description, family=family,
                    scene_id_prefix=scene_id_prefix, seeds=seeds, roles=roles, excluded_registries=excluded_registries,
                    candidates_examined=generated['candidates_examined'], rejection_counts=generated['rejection_counts'],
                    entries=entries)
    _write(folder/'registry.json', registry)
    return registry


class MazeSet:
    """A registered set, with every file verified against its registry on load."""

    def __init__(self, name, root=SHIPPED_ROOT):
        self.folder = Path(root)/name
        self.registry = json.loads((self.folder/'registry.json').read_text())
        self.entries = {e['maze_id']: e for e in self.registry['entries']}

    def _load(self, record):
        data = (self.folder/record['file']).read_bytes()
        if sha256(data) != record['sha256']:
            raise ValueError(f"{self.folder/record['file']} does not match its registry")
        return json.loads(data)

    def maze(self, maze_id):
        return self._load(self.entries[maze_id]['maze'])

    def episode(self, maze_id, episode_index):
        record = next((e for e in self.entries[maze_id]['episodes'] if e['file'] == f'episode_{maze_id:02d}_{episode_index}.json'), None)
        if record is None:
            raise KeyError(f'maze {maze_id} has no episode {episode_index} in set {self.registry["name"]}')
        return self._load(record)

    def maze_ids(self, role=None):
        return sorted(i for i, e in self.entries.items() if role is None or e['role'] == role)
