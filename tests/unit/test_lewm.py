"""The LeWorldModel port: sizes, the official modules' behaviour, causality, rollout alignment and the loss."""
import math

import numpy as np
import pytest
import torch

from go2nav.models.lewm import (ConditionalBlock, LeWMConfig, LeWorldModel, SIGReg, losses, normalise_frames,
                                load_world_model, resize_frame, save_world_model)


def small(**overrides):
    """A small model for fast tests (same structure)."""
    return LeWMConfig(**(dict(width=48, depth=2, heads=2, mlp=96, latent=32, projector_hidden=64, predictor_depth=2,
                               predictor_heads=2, predictor_head_dim=16, predictor_mlp=64) | overrides))


def test_go2_sizes_match_the_paper():
    model = LeWorldModel()
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert 5.4e6 < count(model.encoder) < 5.6e6           # ViT-Tiny
    assert 10.7e6 < count(model.predictor) < 10.9e6       # about 10M in the paper
    assert model.encoder.position.shape[1] == 1+12*16     # 224 x 168 at patch 14
    assert model.config.action_dim == 10                  # 5 policy steps x (forward, yaw)


def test_adaln_zero_blocks_start_as_identity():
    block = ConditionalBlock(32, 2, 16, 64).eval()
    x, c = torch.randn(3, 4, 32), torch.randn(3, 4, 32)
    assert torch.equal(block(x, c), x)


def test_predictor_is_causal():
    model = LeWorldModel.seeded(1, small()).eval()
    for block in model.predictor.layers:  # make the blocks act, so causality is actually exercised
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)
    z, actions = torch.randn(2, 4, 32), torch.randn(2, 4, 10)
    later = z.clone()
    later[:, 3] += torch.randn(2, 32)  # not a constant: the LayerNorms remove a uniform shift
    a, b = model.predict(z, actions), model.predict(later, actions)
    assert torch.allclose(a[:, :3], b[:, :3]) and not torch.allclose(a[:, 3], b[:, 3])


def test_rollout_steps_use_the_aligned_actions():
    model = LeWorldModel.seeded(2, small()).eval()
    for block in model.predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)
    z, actions = torch.randn(3, 4, 32), torch.randn(3, 4-1+5, 10)
    rolled = model.rollout(z, actions, 5)
    assert rolled.shape == (3, 5, 32)
    assert torch.allclose(rolled[:, 0], model.predict(z, actions[:, :4])[:, -1])
    window = torch.cat((z[:, 2:], rolled[:, :2]), 1)  # step 2 attends over frames 2..5
    assert torch.allclose(rolled[:, 2], model.predict(window, actions[:, 2:6])[:, -1])
    changed = actions.clone()
    changed[:, 7] += 1.  # the last action only affects the last step
    assert torch.allclose(model.rollout(z, changed, 5)[:, :4], rolled[:, :4])


def reference_sigreg(proj, A, knots=17):
    """The official SIGReg statistic, written out, for given projections A (D, M)."""
    t = torch.linspace(0, 3, knots)
    dt = 3/(knots-1)
    w = torch.full((knots,), 2*dt)
    w[[0, -1]] = dt
    window = torch.exp(-t.square()/2)
    x = (proj@(A/A.norm(dim=0))).unsqueeze(-1)*t
    err = (x.cos().mean(-3)-window).square()+x.sin().mean(-3).square()
    return ((err@(w*window))*proj.size(-2)).mean()


def test_sigreg_matches_the_official_statistic_and_separates_collapse():
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(3, 256, 32, generator=g)
    sigreg = SIGReg(num_proj=64)
    A = torch.randn(32, 64, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(sigreg(proj, torch.Generator().manual_seed(5)), reference_sigreg(proj, A), rtol=1e-5)
    gaussian = sigreg(proj, torch.Generator().manual_seed(1))
    collapsed = sigreg(proj*.05+1., torch.Generator().manual_seed(1))
    assert collapsed > 20*gaussian  # a Gaussian sample scores O(1) (the statistic is scaled by the sample count)


@pytest.mark.parametrize('rollout_steps', [1, 3])
def test_losses_are_finite_and_train(rollout_steps):
    model = LeWorldModel.seeded(3, small()).train()
    n = model.config.history
    pixels = normalise_frames(torch.randint(0, 255, (4, n+rollout_steps, 168, 224, 3), dtype=torch.uint8))
    actions = torch.randn(4, n+rollout_steps-1, 10)
    terms = losses(model, pixels, actions, rollout_steps=rollout_steps, sigreg=SIGReg(num_proj=32), lam=.09,
                   generator=torch.Generator().manual_seed(0))
    assert ('rollout' in terms) == (rollout_steps > 1) and torch.isfinite(terms['loss'])
    terms['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0  # gradients reach the encoder (no stop-gradient)


def test_inverse_dynamics_variant_trains_the_encoder(tmp_path):
    model = LeWorldModel.seeded(3, small(inverse_dynamics=True)).train()
    n = model.config.history
    pixels = normalise_frames(torch.randint(0, 255, (4, n+2, 168, 224, 3), dtype=torch.uint8))
    actions = torch.randn(4, n+1, 10)
    terms = losses(model, pixels, actions, rollout_steps=2, sigreg=SIGReg(num_proj=32), lam=.09, inverse_weight=.1,
                   generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(terms['inverse'])
    terms['inverse'].backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0  # the inverse term alone reaches the encoder
    save_world_model(tmp_path/'m.pt', model)
    assert load_world_model(tmp_path/'m.pt').inverse is not None
    assert LeWorldModel.seeded(3, small()).inverse is None  # the paper's model has no such head


def test_seeded_model_is_reproducible_and_frames_resize():
    a, b = LeWorldModel.seeded(9, small()), LeWorldModel.seeded(9, small())
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    frame = resize_frame(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    assert frame.shape == (168, 224, 3) and frame.dtype == np.uint8


def test_linear_patch_embedding_equals_the_strided_convolution():
    model = LeWorldModel.seeded(4, small()).encoder
    conv = torch.nn.Conv2d(3, 48, 14, stride=14)
    with torch.no_grad():
        conv.weight.copy_(model.patch_embed.weight.reshape(48, 3, 14, 14))
        conv.bias.copy_(model.patch_embed.bias)
    pixels = torch.randn(2, 3, 168, 224)
    assert torch.allclose(model.patch_embed(model.patches(pixels)), conv(pixels).flatten(2).transpose(1, 2), atol=1e-5)


def test_longer_inverse_spans_decode_the_mean_action():
    model = LeWorldModel.seeded(5, small(inverse_dynamics=True, inverse_spans=(1, 4))).train()
    n = model.config.history
    pixels = normalise_frames(torch.randint(0, 255, (2, n+4, 168, 224, 3), dtype=torch.uint8))
    actions = torch.randn(2, n+3, 10)
    z = model.encode(pixels)
    assert model.infer_actions(z, 4).shape == (2, n+4-4, 10)
    terms = losses(model, pixels, actions, rollout_steps=4, sigreg=SIGReg(num_proj=32), lam=.09, inverse_weight=1.,
                   generator=torch.Generator().manual_seed(0))
    terms['inverse'].backward()
    assert model.inverse_long['4'].net[0].weight.grad.abs().sum() > 0 and model.inverse.net[0].weight.grad.abs().sum() > 0
    assert not hasattr(LeWorldModel.seeded(5, small(inverse_dynamics=True)), 'inverse_long')  # old checkpoints load


def test_body_motion_and_the_motion_term():
    from go2nav.models.lewm import body_motion
    poses = torch.tensor([[[0., 0., math.pi/2], [0., .1, math.pi/2], [-.1, .1, -math.pi+.05]]])
    m = body_motion(poses, 1)
    assert torch.allclose(m[0, 0], torch.tensor([.1, 0., 0.]), atol=1e-6)        # forward along +y when facing +y
    assert torch.allclose(m[0, 1, :2], torch.tensor([0., .1]), atol=1e-6)         # then a step to the left
    assert abs(float(m[0, 1, 2])-(math.pi/2+.05)) < 1e-5                           # yaw change wrapped into (-pi, pi]
    model = LeWorldModel.seeded(6, small(inverse_dynamics=True, inverse_spans=(1, 4), inverse_span_weights=(1, 3),
                                         motion_dynamics=True)).train()
    n = model.config.history
    pixels = normalise_frames(torch.randint(0, 255, (2, n+4, 168, 224, 3), dtype=torch.uint8))
    terms = losses(model, pixels, torch.randn(2, n+3, 10), rollout_steps=4, sigreg=SIGReg(num_proj=32), lam=.09,
                   inverse_weight=1., motion_weight=1., poses=torch.randn(2, n+4, 3),
                   generator=torch.Generator().manual_seed(0))
    terms['motion'].backward()
    assert model.motion['4'].net[0].weight.grad.abs().sum() > 0 and model.encoder.patch_embed.weight.grad.abs().sum() > 0


def test_action_blocks_take_lateral_only_with_three_component_statistics():
    from go2nav.models.lewm import action_blocks
    commands = torch.tensor([[[.2, .1, .3]]*5])
    two = action_blocks(commands, np.zeros(2, np.float32), np.ones(2, np.float32))
    three = action_blocks(commands, np.zeros(3, np.float32), np.ones(3, np.float32))
    assert two.shape[-1] == 10 and torch.allclose(two[0, :2], torch.tensor([.2, .3]))
    assert three.shape[-1] == 15 and torch.allclose(three[0, :3], torch.tensor([.2, .1, .3]))



@pytest.mark.parametrize('command_dim', [2, 3])
def test_zero_command_baseline_matches_the_action_blocks(command_dim):
    from types import SimpleNamespace
    from go2nav.models.lewm import action_blocks
    from go2nav.training.lewm import zero_command_actions
    mean, std = np.arange(1, command_dim+1, dtype=np.float32)/10, np.full(command_dim, .5, np.float32)
    zero = action_blocks(torch.zeros(4, 8, 5, 3), mean, std)
    assert torch.equal(zero_command_actions(torch.ones_like(zero), SimpleNamespace(mean=mean, std=std)), zero)


@pytest.mark.parametrize('command_dim', [2, 3])
def test_c6_takes_the_world_models_command_components_and_round_trips(tmp_path, command_dim):
    from go2nav.models.lewm import LatentDirectPredictor, LatentMotionDecoder, load_heads
    c6 = LatentDirectPredictor(8, torch.zeros(3), torch.ones(3), command_dim=command_dim)
    out = c6(torch.zeros(2, 4, 8), torch.zeros(2, 15, command_dim), torch.zeros(2, 8, command_dim), torch.tensor([1, 8]))
    assert out.shape == (2, 3)
    c5 = LatentMotionDecoder(8, torch.zeros(3), torch.ones(3))
    torch.save(dict(c5=c5.state_dict(), c6=c6.state_dict(), latent=8, target_mean=torch.zeros(3),
                    target_scale=torch.ones(3), **({'command_dim': 3} if command_dim == 3 else {})), tmp_path/'fit.pt')
    assert load_heads(tmp_path/'fit.pt')[1].command_dim == command_dim


def test_the_grid_state_pools_the_patches_and_its_predictor_is_block_causal(tmp_path):
    model = LeWorldModel.seeded(4, small(grid=(3, 4), inverse_dynamics=True)).eval()
    assert model.config.tokens == 13 and not hasattr(LeWorldModel.seeded(4, small()).predictor, 'space_embedding')
    pixels = normalise_frames(torch.randint(0, 255, (2, 4, 168, 224, 3), dtype=torch.uint8))
    z = model.encode(pixels)
    assert z.shape == (2, 4, 13, 32) and model.cls(z).shape == (2, 4, 32)
    tokens = model.encoder.tokens(pixels[:, 0])
    pooled = tokens[:, 1:].reshape(2, 3, 4, 4, 4, 48).mean((2, 4)).reshape(2, 12, 48)  # 4 x 4 blocks of patches
    assert torch.allclose(z[:, 0, 1:], model.grid_projector(pooled.flatten(0, 1)).unflatten(0, (2, 12)), atol=1e-5)
    assert torch.allclose(z[:, 0, 0], model.projector(tokens[:, 0]), atol=1e-5)
    for block in model.predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)
    actions = torch.randn(2, 4, 10)
    later = z.clone()
    later[:, 3, 5] += torch.randn(2, 32)                                  # one grid token of the last frame
    a, b = model.predict(z, actions), model.predict(later, actions)
    assert a.shape == (2, 4, 13, 32)
    assert torch.allclose(a[:, :3], b[:, :3]) and not torch.allclose(a[:, 3], b[:, 3])
    assert not torch.allclose(a[:, 3, 0], b[:, 3, 0])                       # [CLS] sees its own frame's grid
    rolled = model.rollout(z, torch.randn(2, 4-1+3, 10), 3)
    assert rolled.shape == (2, 3, 13, 32)
    assert model.infer_actions(z).shape == (2, 3, 10)                      # inverse dynamics on [CLS]
    save_world_model(tmp_path/'g.pt', model)
    loaded = load_world_model(tmp_path/'g.pt')
    assert loaded.config.grid == (3, 4) and torch.allclose(loaded.encode(pixels), z, atol=1e-5)


def test_grid_losses_train_both_parts_and_keep_the_cls_terms():
    model = LeWorldModel.seeded(5, small(grid=(3, 4))).train()
    pixels = normalise_frames(torch.randint(0, 255, (4, 6, 168, 224, 3), dtype=torch.uint8))
    actions = torch.randn(4, 5, 10)
    terms = losses(model, pixels, actions, rollout_steps=2, sigreg=SIGReg(num_proj=32), lam=.09,
                   generator=torch.Generator().manual_seed(0), grid_weight=.5)
    assert {'pred', 'rollout', 'grid_pred', 'grid_rollout', 'sigreg'} <= set(terms) and torch.isfinite(terms['loss'])
    expected = terms['pred']+terms['rollout']+.09*terms['sigreg']+.5*(terms['grid_pred']+terms['grid_rollout'])
    assert torch.allclose(terms['loss'], expected)
    terms['grid_rollout'].backward()
    assert model.grid_projector.net[0].weight.grad.abs().sum() > 0 and model.predictor.space_embedding.grad is not None
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
