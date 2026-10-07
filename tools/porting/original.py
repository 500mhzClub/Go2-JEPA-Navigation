"""Where the original programme's files are on the development machine (porting tools and equivalence tests only).

tools/porting/original_roots.json (not tracked; copy original_roots.example.json and edit it) maps each root to a
local path:
- `original`: the original LeWMQuad-v3 repository;
- `recovery`, `workspace`, `steam`: the LeWMQuad-v3 folders on the three artefact drives;
- `cache`: the original user cache (the T1 caches and the V-JEPA weights).

Records that name original files (the artefact manifest's `source`, collection-plan dependencies) write them as
`<root>/relative/path`; `resolve` turns such a name back into a local path.
"""
import json
from pathlib import Path

CONFIG = Path(__file__).with_name('original_roots.json')
ROOTS = ('original', 'recovery', 'workspace', 'steam', 'cache')


def roots():
    if not CONFIG.exists():
        raise FileNotFoundError(f'{CONFIG} is missing: copy {CONFIG.with_name("original_roots.example.json")} and set '
                                'the original roots')
    return {name: Path(value) for name, value in json.loads(CONFIG.read_text()).items()}


def available(*names):
    """Whether the roots are configured (and, for each name given, present on this machine)."""
    return CONFIG.exists() and all(roots()[name].exists() for name in names)


def path(root, *parts):
    return roots()[root].joinpath(*parts)


def resolve(name):
    """'<root>/relative/path' -> the local path (an absolute path is returned as it is)."""
    if not name.startswith('<'):
        return Path(name)
    root, _, relative = name.partition('/')
    return path(root[1:-1].strip('<>'), relative)


def name(local):
    """A local path under one of the roots -> '<root>/relative/path' (the most specific root)."""
    local = Path(local)
    matches = [(len(str(base)), root) for root, base in roots().items() if local.is_relative_to(base)]
    if not matches:
        raise ValueError(f'{local} is under no original root')
    root = max(matches)[1]
    return f'<{root}>/{local.relative_to(roots()[root])}'
