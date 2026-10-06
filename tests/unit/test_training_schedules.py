"""The T3 and T4 batch schedules, regenerated from their seeds, equal the original saved schedules."""
import json
from pathlib import Path

import pytest

from go2nav.training import native_predictor as native

ORIGINAL = Path('/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
needs = pytest.mark.skipif(not (ORIGINAL.exists() and native.DATA.exists()), reason='original run folders or exported datasets missing')


@needs
def test_t3_schedule_matches_original():
    original = json.loads((ORIGINAL/'go2_balanced_start_predictor_v1_attempt_001/schedule.json').read_text())
    regenerated = native.t3_schedule(len(native.load_samples('t2_samples')), len(native.load_samples('t3_balanced_start_samples')))
    assert regenerated == [[batch['mixed'] for batch in epoch] for epoch in original]


@needs
def test_t4_schedule_matches_original():
    original = json.loads((ORIGINAL/'go2_horizon_dense_predictor_v1_attempt_001/schedule.json').read_text())
    assert native.t4_schedule(native.load_samples('t4_samples')) == original
