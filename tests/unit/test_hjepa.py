"""H-JEPA: the level layout, joint gradients into level 1, macro-action composition, checkpoints, and the clip crops."""
import numpy as np
import pytest
import torch

from go2nav.data.hjepa_dataset import clip_batch, crop_layout
from go2nav.models.hjepa import (HJEPA, HJEPAConfig, UpperConfig, export_level1, load_hjepa, losses, period_seconds,
                                 save_hjepa)
from go2nav.models.lewm import LeWMConfig, SIGReg, load_world_model

LEVEL1 = dict(rollout_steps=8, lam=.08, inverse_weight=100.)
UPPER = [dict(sigreg=.16, inverse=50., action_sigreg=.005)]*2


def tiny():
    level1 = LeWMConfig(image_hw=(28, 28), width=32, depth=1, heads=2, mlp=64, latent=16, projector_hidden=32,
                        predictor_depth=1, predictor_heads=2, predictor_head_dim=8, predictor_mlp=32,
                        inverse_dynamics=True, inverse_spans=(1, 8), inverse_span_weights=(1, 3))
    upper = dict(latent=16, encoder_hidden=32, projector_hidden=32, macro=4, macro_hidden=8, predictor_depth=1,
                 predictor_heads=2, predictor_head_dim=8, predictor_mlp=32)
    return HJEPAConfig(level1=level1, upper=(UpperConfig(stride=4, context=3, **upper),
                                             UpperConfig(stride=2, context=2, **upper)))


def batch(config, b=6, mask=None):
    g = torch.Generator().manual_seed(0)
    layout = crop_layout(config, 12)
    frames = [torch.randn(b, count, 3, 28, 28, generator=g) for _, count in layout]
    actions = [torch.randn(b, 11, 10, generator=g)]+[torch.randn(b, (count-1)*period, 10, generator=g)
                                                     for period, count in layout[1:]]
    return dict(frames=frames, actions=actions,
                upper_mask=torch.ones(b, dtype=torch.bool) if mask is None else torch.tensor(mask))


def test_go2_levels_step_100_ms_1_6_s_and_12_8_s():
    config = HJEPAConfig()
    assert config.periods == (1, 16, 128)
    assert period_seconds(config) == (.1, 1.6, 12.8)
    assert config.clip_length == 257  # the top level's three states, 128 steps apart


def test_upper_losses_train_level_one_encoder():
    """No stop-gradient: the upper levels' objective alone moves level 1's ViT."""
    torch.manual_seed(0)
    model = HJEPA(tiny()).train()
    terms = losses(model, batch(model.config), level1=LEVEL1, upper=UPPER, sigreg=SIGReg(num_proj=16))
    assert all(torch.isfinite(v) for v in terms.values())
    assert {'l1/loss', 'l2/pred', 'l2/inverse', 'l3/action_sigreg', 'loss'} <= set(terms)
    (terms['l2/loss']+terms['l3/loss']).backward()
    encoder_grads = [p.grad for p in model.level1.encoder.parameters() if p.grad is not None]
    assert encoder_grads and any(g.abs().sum() > 0 for g in encoder_grads)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.level1.predictor.parameters())


def test_upper_levels_skip_unmasked_samples():
    torch.manual_seed(0)
    model = HJEPA(tiny()).train()
    terms = losses(model, batch(model.config, mask=[False]*6), level1=LEVEL1, upper=UPPER, sigreg=SIGReg(num_proj=16))
    assert not any(k.startswith(('l2/', 'l3/')) for k in terms)
    assert torch.equal(terms['loss'], terms['l1/loss'])


def test_macro_actions_compose_through_the_levels():
    model = HJEPA(tiny())
    actions = torch.randn(3, 16, 10)          # two level-3 transitions of 8 level-1 blocks
    assert model.macros(actions, 2).shape == (3, 4, 4)
    assert model.macros(actions, 3).shape == (3, 2, 4)
    # a level-3 macro-action depends only on the blocks of its own transition
    changed = actions.clone()
    changed[:, 8:] += 1.
    assert torch.equal(model.macros(actions, 3)[:, 0], model.macros(changed, 3)[:, 0])
    with pytest.raises(ValueError):
        model.macros(actions[:, :15], 2)


def test_checkpoints_and_level_one_export(tmp_path):
    model = HJEPA.seeded(3, tiny()).eval()
    save_hjepa(tmp_path/'hjepa.pt', model)
    loaded = load_hjepa(tmp_path/'hjepa.pt')
    assert loaded.config == model.config
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
    export_level1(model, tmp_path/'lewm.pt')
    world = load_world_model(tmp_path/'lewm.pt')
    assert all(torch.equal(a, b) for a, b in zip(model.level1.state_dict().values(), world.state_dict().values()))


class FakeCorpus:
    """Two units whose frame i holds the value i and whose action block i holds i, for checking crops."""

    def __init__(self, length):
        self.length = length
        self.units = [dict(kind='mission'), dict(kind='free')]
        self.unit, self.start = np.array([0, 1]), np.array([5, 0])

    def frames(self, unit):
        return np.arange(400, dtype=np.uint8)[:, None, None, None]*np.ones((1, 2, 2, 3), np.uint8)

    def actions(self, unit):
        return np.repeat(np.arange(400, dtype=np.float32)[:, None], 10, axis=1)


def test_clip_crops_align_frames_and_actions():
    config = tiny()
    layout = crop_layout(config, 12)
    corpus = FakeCorpus(config.clip_length)
    out = clip_batch(corpus, np.array([0, 1]), np.random.default_rng(0), layout, config.clip_length, ('mission',))
    assert out['upper_mask'].tolist() == [True, False]
    for b, start in enumerate(corpus.start):
        for level, (period, count) in enumerate(layout):
            index = out['frames'][level][b, :, 0, 0, 0].numpy().astype(int)
            assert len(index) == count and np.all(np.diff(index) == period)
            assert start <= index[0] and index[-1] < start+config.clip_length
            blocks = out['actions'][level][b, :, 0].numpy().astype(int)
            # the blocks run from the crop's first frame to the one before its last: every transition, nothing more
            assert blocks.tolist() == list(range(index[0], index[-1]))
