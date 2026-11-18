"""P37 (docs/dynamic/PLAN.md P37): one shared detector for identical obstacles: its rebuilding and its detections."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def discs(centres, size=64, radius=3.):
    """A mask (1, 1, H, W) of anti-aliased discs at centres (n, 2) px (x, y), as the toy draws them."""
    c = np.arange(size)+.5
    out = np.zeros((size, size))
    for x, y in centres:
        out = np.maximum(out, np.clip(radius+.5-np.hypot(c[None, :]-x, c[:, None]-y), 0, 1))
    return torch.tensor(out, dtype=torch.float32)[None, None]


def test_detections_are_the_discs_centres_with_touching_discs_split_and_the_agent_dropped():
    from go2nav.models.shared_detector import detect
    centres = np.array([[12.3, 40.7], [30.0, 20.25], [36.0, 20.25], [50.6, 52.1]])   # the middle two touch (6 px)
    agent = np.array([50.6, 52.1])
    xy = detect(discs(centres), torch.tensor(agent[None], dtype=torch.float32))[0].numpy()
    found = xy[np.isfinite(xy[:, 0])]
    assert len(found) == 3 and np.isnan(xy[3:]).all()
    d = np.linalg.norm(found[:, None]-centres[None, :3], axis=-1).min(0)
    assert d[0] < .05 and d[1] < .3 and d[2] < .3                         # isolated: exact; touching: split
    assert len(detect(discs(centres))[0].isfinite().all(-1).nonzero()) == 4  # without the agent, all four


def test_an_empty_mask_has_no_detections_and_a_full_frame_keeps_to_the_slots():
    from go2nav.models.shared_detector import detect
    assert detect(torch.zeros(2, 1, 64, 64)).isnan().all()
    grid = np.stack(np.meshgrid(np.arange(6, 64, 10)+.5, np.arange(6, 64, 10)+.5), -1).reshape(-1, 2)
    assert detect(discs(grid), slots=6).isfinite().all()


def test_the_mask_rebuilds_a_frame_and_every_pixel_of_it_gets_a_gradient():
    from go2nav.models.shared_detector import MaskTransporter
    model = MaskTransporter()
    s, t = torch.rand(2, 3, 64, 64)*2-1, torch.rand(2, 3, 64, 64)*2-1
    out, ms, mt = model(s, t)
    assert out.shape == (2, 3, 64, 64) and ms.shape == mt.shape == (2, 1, 64, 64)
    assert ((ms >= 0) & (ms <= 1)).all()
    ms.retain_grad(), mt.retain_grad()
    torch.nn.functional.mse_loss(out, t).backward()
    assert (mt.grad != 0).float().mean() > .99                              # no pixel of the mask sits dead
    assert model.mask.net[0].weight.grad.abs().sum() > 0
    with torch.autocast('cpu', dtype=torch.bfloat16):                       # the mask stays float32 under autocast
        assert model.mask(s).dtype == torch.float32
