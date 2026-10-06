"""Equivalence tests: ported components against replay bundles recorded from the original runtime.

They need the frozen snapshot (GO2NAV_SNAPSHOT, to unpickle bundles) and recorded bundles (GO2NAV_BUNDLES, default
outputs/porting/bundles; see tools/porting/replay_bundle.py). Missing inputs skip the tests.
"""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUNDLES = Path(os.environ.get('GO2NAV_BUNDLES', REPO/'outputs/porting/bundles'))


def bundles():
    return sorted(p.name for p in BUNDLES.iterdir() if (p/'bundle.json').exists()) if BUNDLES.exists() else []


@pytest.fixture(params=bundles() or [None])
def bundle(request):
    if request.param is None:
        pytest.skip('no replay bundles recorded')
    snapshot = Path(os.environ.get('GO2NAV_SNAPSHOT', Path.home()/'Workspace/LeWMQuad-v3-snapshot-2026-10-06'))
    if not snapshot.exists():
        pytest.skip('the frozen snapshot is needed to unpickle bundles')
    return BUNDLES/request.param
