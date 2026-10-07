"""Shipped checkpoints load into the ported models, and the ported models compute exactly what the originals compute."""
import os
from pathlib import Path
import sys

import pytest
import torch

from go2nav import artifacts
from go2nav.models.motion import load_decoder_and_c4, load_motion_readout
from go2nav.models.predictor import DensePredictor, load_horizon_predictor
from go2nav.models.vjepa import pool_tokens

CHECKPOINTS = dict(t4=artifacts.cached('t4_horizon_dense_predictor'), t5b=artifacts.cached('t5b_maze_view_readout'),
                   t7=artifacts.cached('t7_c3_decoder_and_c4'))
needs_artifacts = pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS.values()), reason='shipped checkpoints not in the cache')
needs_snapshot = pytest.mark.skipif('GO2NAV_SNAPSHOT' not in os.environ, reason='needs the frozen snapshot')


def snapshot_imports():
    snapshot = Path(os.environ['GO2NAV_SNAPSHOT'])
    for p in (snapshot/'lewm_worlds', snapshot/'lewm_genesis', snapshot):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def test_seeded_predictor_is_reproducible():
    a, b = DensePredictor.seeded(5), DensePredictor.seeded(5)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@needs_snapshot
def test_seeded_predictor_matches_original_initialisation():
    snapshot_imports()
    from scripts.dev_proprio_predictor_v1 import build_paired
    original, ported = build_paired(2026080901, use_proprio=False), DensePredictor.seeded(2026080901)
    assert list(original.state_dict()) == list(ported.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(original.state_dict().values(), ported.state_dict().values()))


@needs_artifacts
@needs_snapshot
def test_shipped_models_match_original_modules():
    snapshot_imports()
    from lewm.horizon_conditioned_dense_predictor_development import HorizonConditionedDensePredictor
    from lewm.dev_readout_variants_development import ReadoutVariant
    from lewm.dense_visual_motion_readout_development import DenseVisualMotionReadout
    from lewm.navigation_capability_supervised_development import DirectMotionPredictor as OriginalC4
    from scripts.dev_proprio_predictor_v1 import ProprioActionPredictor
    torch.manual_seed(0)
    context, control = torch.randn(2, 3, 768, 1024), torch.randn(2, 3, 5, 2)
    commands, horizon, mask = torch.randn(2, 8, 2), torch.tensor([3, 8]), torch.ones(2, 768, dtype=torch.bool)

    ported_predictor = load_horizon_predictor(CHECKPOINTS['t4'])
    original_predictor = HorizonConditionedDensePredictor(ProprioActionPredictor(use_proprio=False))
    original_predictor.load_state_dict(torch.load(CHECKPOINTS['t4'], map_location='cpu', weights_only=False)['model_state_dict'])
    with torch.no_grad():
        assert torch.equal(ported_predictor(context, commands, horizon, mask, control=control),
                           original_predictor.eval()(context, commands, horizon, mask, control=control))

    readout = load_motion_readout(CHECKPOINTS['t5b'])
    state = torch.load(CHECKPOINTS['t5b'], map_location='cpu', weights_only=False)['model_state_dict']
    original_readout = DenseVisualMotionReadout(state['target_mean'], state['target_scale'])
    original_readout.load_state_dict(state)
    pooled = [pool_tokens(torch.randn(2, 768, 1024)) for _ in range(4)]
    with torch.no_grad():
        assert torch.equal(readout(*pooled[:2]), original_readout.eval()(*pooled[:2]))

    decoder, c4 = load_decoder_and_c4(CHECKPOINTS['t7'], readout)
    fit = torch.load(CHECKPOINTS['t7'], map_location='cpu', weights_only=False)
    original_decoder = ReadoutVariant(original_readout, past_frames=True, **fit['readout_config'])
    original_decoder.load_state_dict(fit['readout'])
    original_c4 = OriginalC4(fit['c4']['target_mean'], fit['c4']['target_scale'])
    original_c4.load_state_dict(fit['c4'])
    features = torch.stack(pooled[:3], dim=1)
    with torch.no_grad():
        assert torch.equal(decoder(*pooled), original_decoder.eval()(*pooled))
        assert torch.equal(c4(features, control, commands, horizon), original_c4.eval()(features, control, commands, horizon))


C1_MODEL = artifacts.cached('t6_c1_command_model')


@pytest.mark.skipif(not C1_MODEL.exists(), reason='shipped C1 model not in the cache')
@needs_snapshot
def test_command_model_matches_original():
    import numpy as np
    snapshot_imports()
    from lewm.short_pulse_navigation_runtime_development import command_predictions
    from go2nav.models.command import CommandHistoryModel
    model = CommandHistoryModel.load(C1_MODEL)
    with np.load(C1_MODEL) as z:
        original = {k: z[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')}
    rng = np.random.default_rng(1)
    commands, history = rng.normal(size=(6, 8, 3))*[.2, 0, .4], rng.normal(size=420)
    np.testing.assert_array_equal(model.predict(commands, history), command_predictions(original, history, commands))
