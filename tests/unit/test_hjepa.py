"""H-JEPA: the level layout, joint gradients into level 1, macro-action composition, checkpoints, the clip crops and
the batches (level 1 exactly the flat LeWM's, the upper levels from the same mazes)."""
import numpy as np
import pytest
import torch

from go2nav.data.hjepa_dataset import HJEPABatches, clip_crops, upper_layout
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


def batch(config, b1=6, bu=4):
    g = torch.Generator().manual_seed(0)
    layout = upper_layout(config)
    frames = [torch.randn(b1, 12, 3, 28, 28, generator=g)]+[torch.randn(bu, count, 3, 28, 28, generator=g)
                                                            for _, count in layout]
    actions = [torch.randn(b1, 11, 10, generator=g)]+[torch.randn(bu, (count-1)*period, 10, generator=g)
                                                      for period, count in layout]
    return dict(frames=frames, actions=actions)


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
    """Windows of `length` frames in units on mazes 0-2; frame i holds the value i, action block i holds i."""

    def __init__(self, length, starts=(5, 0, 30, 2)):
        self.length = length
        self.units = [dict(maze=m) for m in (0, 1, 2, 2)]
        self.unit, self.start = np.arange(len(starts)), np.asarray(starts)
        self.mazes = np.array([0, 1, 2])
        self.by_maze = {0: np.array([0]), 1: np.array([1]), 2: np.array([2, 3])}

    def frames(self, unit):
        return np.arange(250, dtype=np.uint8)[:, None, None, None]*np.ones((1, 2, 2, 3), np.uint8)

    def actions(self, unit):
        return np.repeat(np.arange(250, dtype=np.float32)[:, None], 10, axis=1)

    def sample(self, seed, step, *, mazes_per_batch, windows_per_maze, batch_size=None):
        return np.array([0, 2, 3])  # mazes 0 and 2

    def batch(self, indices):
        return torch.zeros(len(indices), self.length), torch.zeros(len(indices), self.length-1), torch.zeros(len(indices), 2)


def test_clip_crops_align_frames_and_actions():
    config = tiny()
    layout = upper_layout(config)
    corpus = FakeCorpus(config.clip_length)
    frames, actions = clip_crops(corpus, np.arange(4), np.random.default_rng(0), layout)
    for b, start in enumerate(corpus.start):
        for level, (period, count) in enumerate(layout):
            index = frames[level][b, :, 0, 0, 0].numpy().astype(int)
            assert len(index) == count and np.all(np.diff(index) == period)
            assert start <= index[0] and index[-1] < start+config.clip_length
            blocks = actions[level][b, :, 0].numpy().astype(int)
            # the blocks run from the crop's first frame to the one before its last: every transition, nothing more
            assert blocks.tolist() == list(range(index[0], index[-1]))


def test_batches_keep_level_one_on_the_flat_stream_and_upper_clips_on_its_mazes():
    config = tiny()
    windows, clips = FakeCorpus(12), FakeCorpus(config.clip_length)
    data = HJEPABatches(windows, clips, config, seed=1, first=0, last=5, clips_per_maze=3)
    item = data[2]
    assert item['frames'][0].shape[0] == 3                 # windows.sample's draw, as the flat LeWM's
    upper = item['frames'][1]
    assert upper.shape[:2] == (6, 4) and item['frames'][2].shape[:2] == (6, 3)  # 2 mazes x 3 clips, fixed shapes
    first = item['frames'][2][:, 0, 0, 0, 0].numpy().astype(int)  # frame values index the unit's frames
    assert set(first) <= {5, 30, 2}                       # the starts of units 0, 2 and 3: mazes 0 and 2 only
    again = data[2]
    assert all(torch.equal(a, b) for a, b in zip(item['frames'][1:], again['frames'][1:]))  # a pure function of the step
