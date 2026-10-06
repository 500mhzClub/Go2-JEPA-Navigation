"""Shipped artefacts by name (artifacts/manifest.yaml).

`path(name)` returns the fetched copy in the go2nav cache ($GO2NAV_CACHE, default ~/.cache/go2nav, as
`<name>/<file name>`) or, until the artefacts are hosted, the original file at the manifest's `source` path.
"""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('GO2NAV_CACHE', Path.home()/'.cache/go2nav'))


def entries():
    manifest = yaml.safe_load((REPO/'artifacts/manifest.yaml').read_text())
    return {entry['name']: entry for entry in manifest['artifacts']}


def path(name):
    entry = entries()[name]
    source = Path(entry['source'])
    cached = CACHE/name/source.name
    for candidate in (cached, source):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'artefact {name} not found at {cached} or {source}; see artifacts/fetch.py')
