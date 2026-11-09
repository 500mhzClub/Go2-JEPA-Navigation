"""Phase C's model (docs/dynamic/PLAN.md P15): the ring and proprioception tokens, the upper levels' rollouts, the
stop-gradient heads, and their inputs and labels from the corpus."""
import json
import math

import numpy as np
import pytest
import torch

from go2nav.data import observations as o
from go2nav.data.lewm_dataset import Corpus
from go2nav.models.hjepa import HJEPA, HJEPAConfig, UpperConfig, losses, upper_terms
from go2nav.models.lewm import LeWMConfig, LeWorldModel, SIGReg

LEVEL1 = dict(rollout_steps=8, lam=.08, inverse_weight=100.)
UPPER = [dict(sigreg=.16, inverse=50., action_sigreg=.005, rollout=1.)]*2


def level1_config(**overrides):
    return LeWMConfig(image_hw=(28, 28), width=32, depth=1, heads=2, mlp=64, latent=16, projector_hidden=32,
                      predictor_depth=1, predictor_heads=2, predictor_head_dim=8, predictor_mlp=32, command_dim=3,
                      inverse_dynamics=True, inverse_spans=(1, 8), inverse_span_weights=(1, 3), **overrides)


def config(rollout=4, heads=True, obs=True):
    upper = dict(latent=16, encoder_hidden=32, projector_hidden=32, macro=4, macro_hidden=8, predictor_depth=1,
                 predictor_heads=2, predictor_head_dim=8, predictor_mlp=32)
    sensors = dict(range_sectors=8, proprio_dim=o.PROPRIO_DIM) if obs else {}
    return HJEPAConfig(level1=level1_config(**sensors), heads=heads,
                       upper=(UpperConfig(stride=4, context=3, rollout_steps=rollout, **upper),
                              UpperConfig(stride=2, context=2, **upper)))


def batch(c, b1=6, bu=4):
    from go2nav.data.hjepa_dataset import upper_layout
    g = torch.Generator().manual_seed(0)
    layout = upper_layout(c)
    out = dict(frames=[torch.randn(b1, 12, 3, 28, 28, generator=g)]
               + [torch.randn(bu, count, 3, 28, 28, generator=g) for _, count in layout],
               actions=[torch.randn(b1, 11, 15, generator=g)]
               + [torch.randn(bu, (count-1)*period, 15, generator=g) for period, count in layout])
    if c.level1.obs_dim:
        out['obs'] = [torch.rand(f.shape[0], f.shape[1], o.OBS_DIM, generator=g) for f in out['frames']]
    if c.heads:
        out['poses'] = [torch.randn(b1, 12, 3, generator=g), torch.randn(bu, layout[0][1], 3, generator=g)]
        out['labels'] = dict(clearance=torch.rand(b1, 12, generator=g)*.3,
                             presence=(torch.rand(b1, 12, 2, generator=g) > .5).float())
    return out


def test_the_ring_and_the_body_are_tokens_beside_the_camera():
    torch.manual_seed(0)
    model = LeWorldModel(level1_config(range_sectors=8, proprio_dim=o.PROPRIO_DIM)).eval()
    pixels, obs = torch.randn(2, 3, 3, 28, 28), torch.rand(2, 3, o.OBS_DIM)
    z = model.encode(pixels, obs)
    assert z.shape == (2, 3, 16)
    turned = obs.clone()
    turned[..., o.RANGE] = torch.roll(obs[..., o.RANGE], 90, -1)            # the same ring, a quarter turn
    moved = obs.clone()
    moved[..., o.PROPRIO] += 1.
    assert not torch.allclose(model.encode(pixels, turned), z) and not torch.allclose(model.encode(pixels, moved), z)
    with pytest.raises(ValueError):
        model.encode(pixels)                                                 # it needs its sensors
    with pytest.raises(ValueError):
        LeWorldModel(level1_config()).encode(pixels, obs)                    # a camera-only model takes none


def test_phase_c_losses_have_rollouts_and_heads_and_the_heads_never_reach_the_latent():
    torch.manual_seed(0)
    model = HJEPA(config()).train()
    terms = losses(model, batch(model.config), level1=LEVEL1, upper=UPPER, sigreg=SIGReg(num_proj=16))
    assert all(torch.isfinite(v) for v in terms.values())
    assert {'l2/rollout', 'heads/contact', 'heads/presence', 'heads/motion', 'heads/motion2'} <= set(terms)
    assert 'l3/rollout' not in terms
    heads = sum(v for k, v in terms.items() if k.startswith('heads/'))
    heads.backward()
    assert all(p.grad is None for p in model.level1.parameters())           # stop-gradient
    assert all(p.grad is None for p in model.upper.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads.parameters() if p.dim() > 1)


def test_the_rollout_needs_a_longer_crop_and_without_one_the_terms_are_the_old_ones():
    assert config(rollout=6).upper[0].crop == 9 and config(rollout=0).upper[0].crop == 4
    torch.manual_seed(0)
    model = HJEPA(config(rollout=0, heads=False, obs=False)).eval()
    level = model.upper[0]
    states, macros = torch.randn(3, 4, 16), torch.randn(3, 3, 4)
    terms = upper_terms(level, states, macros, SIGReg(num_proj=16))
    assert 'rollout' not in terms
    assert torch.allclose(terms['pred'], (level.predict(states[:, :-1], macros)-states[:, 1:]).pow(2).mean())


def test_proprioception_is_in_the_body_frame():
    yaw = math.pi/2
    q = np.array([[0., 0., math.sin(yaw/2), math.cos(yaw/2)]])               # x, y, z, w: a quarter turn left
    assert np.allclose(o.rotate_into_body(q, np.array([[1., 0., 0.]])), [[0., -1., 0.]])
    assert np.allclose(o.rotate_into_body(q, np.array([[0., 0., -1.]])), [[0., 0., -1.]])
    assert np.allclose(o.rotate_into_body(q, np.array([[0., 0., .7]])), [[0., 0., .7]])   # a yaw rate stays a yaw rate


def test_the_corpus_hands_over_sensors_and_head_labels(tmp_path):
    root, obs_root, heads_root, n = tmp_path/'c', tmp_path/'o', tmp_path/'h', 20
    unit = root/'units/lewm_train/0000_standoff'
    unit.mkdir(parents=True)
    np.save(unit/'frames.npy', np.zeros((n, 4, 4, 3), np.uint8))
    np.savez(unit/'steps.npz', executed=np.zeros((n, 5, 3), np.float32), pose=np.tile([0, 0, .3, 0, 0, 0, 1.], (n, 1)))
    (obs_root/'units/lewm_train/0000_standoff').mkdir(parents=True)
    np.save(obs_root/"units/lewm_train/0000_standoff"/o.FILE, np.repeat(np.arange(n, dtype=np.float16)[:, None], o.OBS_DIM, 1))
    (heads_root/'units/lewm_train/0000_standoff').mkdir(parents=True)
    np.savez(heads_root/'units/lewm_train/0000_standoff/heads.npz', clearance=np.linspace(0, 1, n).astype(np.float16),
             presence=np.arange(n) % 2 == 0, presence_ring=np.ones(n, bool))
    units = [dict(path='units/lewm_train/0000_standoff', role='lewm_train', kind='standoff', maze=0, frames=n)]
    (root/'index_4.json').write_text(json.dumps(dict(length=4, units=units)))
    np.savez(root/'index_4.npz', unit=np.zeros(3, np.int32), start=np.array([0, 5, 16], np.int32))
    corpus = Corpus(root, length=4, command_dim=3, poses=True, obs_root=obs_root, heads_root=heads_root)
    frames, actions, keys, poses, obs, labels = corpus.batch(np.array([1, 2]))
    assert obs.shape == (2, 4, o.OBS_DIM) and float(obs[0, 0, 0]) == 5 and float(obs[1, 3, -1]) == 19
    assert labels['clearance'].shape == (2, 4) and labels['presence'].shape == (2, 4, 2)
    assert labels['presence'][0, :, 0].tolist() == [0., 1., 0., 1.] and labels['presence'][..., 1].all()
