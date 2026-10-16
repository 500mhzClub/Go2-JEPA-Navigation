"""The C5/C6 forecasters: the controller's interface, and C5's rollout equal to the world model's own."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from go2nav.controllers.controller import PlanContext
from go2nav.controllers.lewm import LeWMDirectForecaster, LeWMForecaster
from go2nav.models.lewm import (LatentDirectPredictor, LatentMotionDecoder, LeWMConfig, LeWorldModel, control_stats,
                                normalise_frames, resize_frame)
from go2nav.planning.actions import command_sequences


def packet(ns, rng, history):
    return {'image': {'rgb': rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), 'measured_ns': ns},
            'sensor_state': {'control': {'applied_command': {'values': history}}}}


@pytest.fixture
def setup():
    torch.manual_seed(0)
    config = LeWMConfig(width=48, depth=2, heads=2, mlp=96, latent=32, projector_hidden=64, predictor_depth=2,
                        predictor_heads=2, predictor_head_dim=16, predictor_mlp=64)
    world = LeWorldModel(config).eval()
    for block in world.predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)
    rng = np.random.default_rng(0)
    history = np.zeros((15, 3), np.float32)
    history[-3:] = [[.2, 0., 0.], [.2, 0., .45], [.16, 0., .45]]
    packets = tuple(packet(2_000_000_000+k*100_000_000, rng, history) for k in range(4))
    context = PlanContext(frame=20, measured_ns=packets[-1]['image']['measured_ns'], history=packets, native=packets,
                          poses={}, commitments=())
    return world, context, history


def test_c5_rollout_matches_the_world_model(setup):
    world, context, history = setup
    decoder = LatentMotionDecoder(32, [0., 0., 0.], [.1, .1, .1]).eval()
    forecaster = LeWMForecaster(world, decoder, device='cpu')
    prefix = [(.2, 0., 0.)]*3
    out = forecaster.forecast(context, prefix, pulse=False)
    assert out.shape == (6, 8, 5) and out.dtype == np.float32 and np.all(out[..., 4] == -1000.)
    assert np.allclose(out[..., 2]**2+out[..., 3]**2, 1., atol=1e-5)
    # recompute candidate 1 (forward) by hand
    frames = torch.from_numpy(np.stack([resize_frame(p['image']['rgb']) for p in context.history]))
    z = world.encode(normalise_frames(frames)[None])
    mean, std = (torch.as_tensor(v) for v in control_stats())
    tape = command_sequences(prefix)[1]  # within the limits already
    commands = (torch.tensor(np.concatenate((history[-3:], tape))[:, [0, 2]], dtype=torch.float32)-mean)/std
    rolled = world.rollout(z, commands[None, :, None].expand(-1, -1, 5, -1).flatten(2), 8)
    expected = decoder(z.expand(8, -1, -1), rolled[0])
    assert np.allclose(out[1, :, :2], expected[:, :2].detach().numpy(), atol=2e-4)


def test_c6_interface_and_distinct_candidates(setup):
    world, context, _ = setup
    head = LatentDirectPredictor(32, [0., 0., 0.], [.1, .1, .1]).eval()
    torch.nn.init.normal_(head.net[0].weight, std=.05)
    out = LeWMDirectForecaster(world, head, device='cpu').forecast(context, [(0., 0., 0.)]*3, pulse=False)
    assert out.shape == (6, 8, 5) and not np.allclose(out[1], out[4])


def test_the_history_must_be_four_frames_100_ms_apart(setup):
    world, context, _ = setup
    late = context.history[:3]+({**context.history[3], 'image': {**context.history[3]['image'],
                                                                 'measured_ns': context.history[3]['image']['measured_ns']+1}},)
    forecaster = LeWMForecaster(world, LatentMotionDecoder(32, [0.]*3, [1.]*3), device='cpu')
    with pytest.raises(ValueError):
        forecaster.forecast(replace(context, history=late), [(0., 0., 0.)]*3, pulse=False)
