"""The mover probe's supervised ceiling (go2nav.eval.mover_ceiling)."""
import numpy as np


def test_targets_hold_the_position_then_the_displacements():
    from go2nav.eval.mover_ceiling import targets
    data = dict(xy_t=np.array([[1., 2.]]), target=np.array([[[1.5, 2.], [2., 2.], [3., 1.], [5., 0.]]]))
    assert np.allclose(targets(data, np.array([0])), [[1., 2., .5, 0., 1., 0., 2., -1., 4., -2.]])


def test_the_network_reads_uint8_frame_stacks():
    import torch
    from go2nav.eval.mover_ceiling import HISTORY, network
    net = network(56, 72, 10)
    assert net(torch.zeros(2, HISTORY, 56, 72, 3, dtype=torch.uint8)).shape == (2, 10)


def test_mirroring_flips_the_image_and_the_lateral_targets():
    import torch
    from go2nav.eval.mover_ceiling import mirrored
    y = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]])
    assert torch.equal(mirrored(y, torch.tensor([True, False])), torch.tensor([[1., -2., 3., -4.], [1., 2., 3., 4.]]))


def test_the_patch_head_reads_token_stacks_and_the_encoder_exposes_its_tokens():
    import torch
    from go2nav.eval.mover_ceiling import HISTORY, patch_network
    from go2nav.models.lewm import LeWMConfig, LeWorldModel
    assert patch_network(12, 16, 8, 10)(torch.zeros(2, HISTORY, 12, 16, 8, dtype=torch.float16)).shape == (2, 10)
    model = LeWorldModel.seeded(0, LeWMConfig.named('tiny', depth=2)).eval()
    x = torch.randn(2, 3, *model.config.image_hw)
    tokens = model.encoder.tokens(x)
    assert tokens.shape[1] == 1+(model.config.image_hw[0]//14)*(model.config.image_hw[1]//14)
    assert torch.allclose(model.encoder(x), tokens[:, 0])
