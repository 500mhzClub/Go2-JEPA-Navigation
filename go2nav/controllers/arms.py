"""The controllers compared in the experiments ("arms"), by name: each is the shared navigation stack with its own
motion forecaster.

- **C1**: the command-history model (T6); **C1A**: C1 with online travel adaptation; **C1R**: C1 with the stage-2
  refit model.
- **C2**: model-free reactive control (no forecaster).
- **C3**: the frozen V-JEPA 2.1 encoder, the T4 predictor and the C3 decoder (T7, or a stage-2 refit).
- **C4**: the direct readout matched to the C3 decoder (same fit file).
- **C0**: the simulator oracle (`go2nav.controllers.oracle`).
"""
import json

from go2nav import artifacts
from go2nav.models.motion import load_decoder_and_c4, load_motion_readout

ARMS = ('C0', 'C1', 'C1A', 'C1R', 'C2', 'C3', 'C4')
REPO_STATS = artifacts.REPO/'data/training/control_norm_stats.json'


def control_stats():
    stats = json.loads(REPO_STATS.read_text())
    return stats['control_mean'], stats['control_std']


def forecaster(arm, *, decoder_fit=None, device='cuda:0'):
    """The forecaster of a learned arm (None for C2). `decoder_fit` is the C3-decoder/C4 fit file (default: the
    deployed T7 fit)."""
    from go2nav.controllers.adaptive import AdaptiveTravelForecaster
    from go2nav.controllers.forecasters import CommandHistoryForecaster
    if arm == 'C1':
        return CommandHistoryForecaster(artifacts.path('t6_c1_command_model'))
    if arm == 'C1A':
        return AdaptiveTravelForecaster(CommandHistoryForecaster(artifacts.path('t6_c1_command_model')))
    if arm == 'C1R':
        return CommandHistoryForecaster(artifacts.path('t8_stage2_c1_refit'))
    if arm == 'C2':
        return None
    if arm in ('C3', 'C4'):
        from go2nav.controllers.neural import DirectForecaster, JepaForecaster
        decoder, c4 = load_decoder_and_c4(decoder_fit or artifacts.path('t7_c3_decoder_and_c4'),
                                          load_motion_readout(artifacts.path('t5b_maze_view_readout')))
        mean, std = control_stats()
        if arm == 'C4':
            return DirectForecaster(c4, mean, std, device=device)
        from go2nav.models.predictor import load_horizon_predictor
        return JepaForecaster(load_horizon_predictor(artifacts.path('t4_horizon_dense_predictor')), decoder, mean, std,
                              device=device)
    raise ValueError(f'unknown or unported arm {arm!r}')
