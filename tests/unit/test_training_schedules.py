"""The T3 and T4 batch schedules, regenerated from their seeds, equal the original saved schedules."""
import json
from pathlib import Path

import pytest

from go2nav.training import native_predictor as native
from tools.porting import original

ORIGINAL = original.path('workspace', '.generated/navigation_development_artifacts_v1') if original.available('workspace') else Path('/nonexistent')
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


MAZE_VIEW = (original.path('recovery', '.generated/navigation_development_artifacts_v1/go2_maze_view_readout_v1_attempt_003') if original.available('recovery')
             else Path('/nonexistent'))


@pytest.mark.skipif(not MAZE_VIEW.exists(), reason='original T5b run folder missing')
def test_t5b_schedule_matches_original():
    import numpy as np
    from go2nav.training import readouts
    original = json.loads((MAZE_VIEW/'schedule.json').read_text())
    data = json.loads((readouts.DATA/'t5_maze_view.json').read_text())
    steps, half, seed = readouts.CONTINUATION_STEPS, readouts.HALF, readouts.T5B_SEED
    assert readouts.cycle(len(data['old']['pairs']), steps*half, seed).reshape(steps, half).tolist() == original['shared']
    assert readouts.cycle(len(data['maze']['pairs']), steps*half, seed+2).reshape(steps, half).tolist() == original['maze']
    for k, key in ((10, 'shared_horizons'), (11, 'other_horizons')):
        h = np.tile(np.arange(8), steps*half//8)
        np.random.default_rng(seed+k).shuffle(h)
        assert h.reshape(steps, half).tolist() == original[key]
