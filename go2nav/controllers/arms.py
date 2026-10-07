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
from pathlib import Path

from go2nav import artifacts
from go2nav.models.motion import load_c3_decoder, load_decoder_and_c4, load_motion_readout

ARMS = ('C0', 'C1', 'C1A', 'C1R', 'C2', 'C3', 'C4')
REPO_STATS = artifacts.REPO/'data/training/control_norm_stats.json'


def control_stats():
    stats = json.loads(REPO_STATS.read_text())
    return stats['control_mean'], stats['control_std']


def forecaster(arm, *, decoder_fit=None, models=None, device='cuda:0'):
    """The forecaster of a learned arm (None for C2). The models are the shipped artefacts unless `models` maps an
    artefact name to another checkpoint (a retrained one); `decoder_fit` is the C3-decoder/C4 fit file (default: the
    T7 fit), or an artefact name; a plain readout checkpoint (e.g. `c3v2_readout`) gives the v4-era C3 decoder."""
    from go2nav.controllers.adaptive import AdaptiveTravelForecaster
    from go2nav.controllers.forecasters import CommandHistoryForecaster

    def model(name):
        return Path(models[name]) if models and name in models else artifacts.path(name)
    if arm == 'C1':
        return CommandHistoryForecaster(model('t6_c1_command_model'))
    if arm == 'C1A':
        return AdaptiveTravelForecaster(CommandHistoryForecaster(model('t6_c1_command_model')))
    if arm == 'C1R':
        return CommandHistoryForecaster(model('t8_stage2_c1_refit'))
    if arm == 'C2':
        return None
    if arm in ('C3', 'C4'):
        from go2nav.controllers.neural import DirectForecaster, JepaForecaster
        if decoder_fit is not None and str(decoder_fit) in artifacts.entries():
            decoder_fit = artifacts.path(str(decoder_fit))
        fit, base = decoder_fit or model('t7_c3_decoder_and_c4'), load_motion_readout(model('t5b_maze_view_readout'))
        mean, std = control_stats()
        if arm == 'C4':
            return DirectForecaster(load_decoder_and_c4(fit, base)[1], mean, std, device=device)
        decoder = load_c3_decoder(fit, base)
        from go2nav.models.predictor import load_horizon_predictor
        return JepaForecaster(load_horizon_predictor(model('t4_horizon_dense_predictor')), decoder, mean, std,
                              device=device)
    raise ValueError(f'unknown or unported arm {arm!r}')
