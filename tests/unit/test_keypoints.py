"""P30 (docs/dynamic/PLAN.md P30): learned keypoints, their scoring, and the object tokens over them."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def test_a_peaked_heatmap_gives_its_pixel_centre_and_the_masks_sit_there():
    from go2nav.models.keypoints import KeypointConfig, KeypointNet, Transporter
    net = KeypointNet(KeypointConfig(keypoints=2))
    logits = torch.full((1, 2, 64, 64), -1e4)
    logits[0, 0, 10, 40], logits[0, 1, 50, 5] = 0., 0.                     # rows are y, columns x
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return logits
    net.net = Fixed()
    xy = net(torch.zeros(1, 3, 64, 64))
    assert torch.allclose(xy, torch.tensor([[[40.5, 10.5], [5.5, 50.5]]]), atol=1e-4)
    model = Transporter(KeypointConfig(keypoints=2))
    masks = model.masks(xy)
    assert masks.shape == (1, 2, 64, 64) and float(masks[0, 0, 10, 40]) > .9 and float(masks[0, 0, 30, 30]) < 1e-6


def test_the_transporter_rebuilds_a_frame_and_its_keypoints_learn_from_it():
    from go2nav.models.keypoints import KeypointConfig, Transporter
    model = Transporter(KeypointConfig(keypoints=3))
    s, t = torch.rand(4, 3, 64, 64)*2-1, torch.rand(4, 3, 64, 64)*2-1
    out, kp = model(s, t)
    assert out.shape == (4, 3, 64, 64) and kp.shape == (4, 3, 2) and ((kp >= 0) & (kp <= 64)).all()
    torch.nn.functional.mse_loss(out, t).backward()
    assert model.keypoints.net[0].weight.grad.abs().sum() > 0                # the loss reaches the keypoint network


def test_the_scoring_assigns_keypoints_and_removes_a_constant_offset():
    from go2nav.models.keypoints import assign, measures
    rng = np.random.default_rng(0)
    truth = rng.uniform(3, 61, (500, 4, 2))
    keypoints = np.concatenate((truth[:, [2, 0, 3, 1]]+np.array([1.2, -.9]), rng.uniform(0, 64, (500, 2, 2))), 1)
    order = assign(keypoints, truth)
    assert order.tolist() == [1, 3, 0, 2]                                   # object j's keypoint
    offsets = (truth-keypoints[:, order]).mean(0)
    m = measures(keypoints, truth, order, offsets)
    assert max(m['corrected']['precision']) < 1e-9 and min(m['corrected']['binding']) == 1. and max(m['corrected']['swaps']) == 0.
    assert abs(m['raw']['precision'][0]-1.5) < 1e-9 and m['raw']['binding'][0] == 0.


def test_the_object_tokens_run_over_a_locator_in_its_assigned_order():
    from go2nav.training import recipe_bench as training
    model = training.build('q3k').eval()
    assert model.locator is not None and not any(p.requires_grad for p in model.locator.parameters())
    pixels = torch.rand(2, 3, 3, 64, 64)*2-1
    z = model.encode(pixels)
    assert z.shape == (2, 3, 4, 128)
    model.order.copy_(torch.tensor([3, 2, 1, 0]))
    with torch.no_grad():
        xy = model.locator(pixels.flatten(0, 1))
    assert torch.allclose(xy[:, model.order][:, 0], xy[:, 3])
    with torch.autocast('cpu', dtype=torch.bfloat16):                       # the locator stays in float32 under autocast
        assert model.locator(pixels.flatten(0, 1)).dtype == torch.float32


def test_separation_is_one_when_keypoints_coincide_and_near_zero_when_apart():
    from go2nav.models.keypoints import separation
    same = torch.tensor([[[10., 10.]]*4])
    apart = torch.tensor([[[5., 5.], [5., 55.], [55., 5.], [55., 55.]]])
    assert abs(float(separation(same))-1.) < 1e-6 and float(separation(apart)) < 1e-6
