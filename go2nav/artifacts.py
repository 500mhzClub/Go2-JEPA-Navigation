"""Shipped artefacts by name (artifacts/manifest.yaml).

`path(name)` returns the artefact's copy in the go2nav cache ($GO2NAV_CACHE, default ~/.cache/go2nav), as
`<folder>/<file name>`; the folder is the artefact's name unless the manifest gives one (companion files share their
checkpoint's folder, as the locomotion policy's training config does). The cache is filled by artifacts/fetch.py (and, on the development machine until the artefacts
are hosted, by tools/porting/link_original_artifacts.py).
"""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('GO2NAV_CACHE', Path.home()/'.cache/go2nav'))


def entries():
    manifest = yaml.safe_load((REPO/'artifacts/manifest.yaml').read_text())
    return {entry['name']: entry for entry in manifest['artifacts']}


def cached(name):
    entry = entries()[name]
    return CACHE/entry.get('folder', name)/Path(entry['source']).name


def path(name):
    if not cached(name).exists():
        raise FileNotFoundError(f'artefact {name} is not in the cache ({cached(name)}); see artifacts/fetch.py')
    return cached(name)


def available(name):
    """Whether the artefact is in the cache."""
    return cached(name).exists()
