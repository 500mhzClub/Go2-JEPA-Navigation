"""P31 (docs/dynamic/PLAN.md P31): the scans as images, and the locator's stage-1 measures."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def test_a_return_is_drawn_where_it_is_in_the_frame():
    from go2nav.data import ring_rasters as rr
    from go2nav.perception.range import AZIMUTHS, MOUNT_BODY_M
    ranges = torch.zeros(1, 720)
    beam = int(np.argmin(np.abs(AZIMUTHS-0.)))                              # straight ahead
    ranges[0, beam] = 1.
    pose = torch.tensor([[2., 3., np.pi/2]], dtype=torch.float32)
    points, valid = rr.returns(ranges, pose)
    assert torch.allclose(points[0, beam], torch.tensor([2., 3.+float(MOUNT_BODY_M[0])+1.]), atol=1e-5) and valid.sum() == 1
    image = rr.rasterise(ranges, pose, pose)                                 # drawn in the robot's own pose
    ys, xs = np.mgrid[0:64, 0:64]+.5
    w = image[0, 0].numpy()
    centre = np.array([(w*xs).sum()/w.sum(), (w*ys).sum()/w.sum()])
    expected = rr.to_pixels(torch.tensor([float(MOUNT_BODY_M[0])+1., 0.])).numpy()   # ahead is the frame's +x
    assert np.abs(centre-expected).max() < 1e-4 and abs(w.sum()-1.) < 1e-5
    local = rr.to_frame(points[:, beam:beam+1], pose)
    assert torch.allclose(rr.to_world(local, pose), points[:, beam:beam+1], atol=1e-5)
    assert torch.allclose(rr.to_local(rr.to_pixels(local)), local, atol=1e-5)


def test_the_nearest_return_stands_in_for_the_walls():
    from go2nav.data import ring_rasters as rr
    scan = torch.tensor([[[0., 0.], [1., 0.], [5., 5.]]])
    valid = torch.tensor([[True, True, False]])
    d = rr.nearest_return(torch.tensor([[[0.5, 0.], [4., 5.]]]), scan, valid, clip=1.)
    assert torch.allclose(d, torch.tensor([[.5, 1.]]))


def test_joining_keypoints_follows_them_and_a_one_channel_locator_runs():
    from go2nav.models.keypoints import KeypointConfig, Transporter
    from go2nav.training.ring_keypoints import join
    a = np.array([[0., 0.], [5., 5.], [9., 1.]])
    assert join(a, a[[2, 0, 1]]+.1).tolist() == [1, 2, 0]
    model = Transporter(KeypointConfig(keypoints=6, channels=1))
    out, kp = model(torch.rand(2, 1, 64, 64)*2-1, torch.rand(2, 1, 64, 64)*2-1)
    assert out.shape == (2, 1, 64, 64) and kp.shape == (2, 6, 2)


def test_the_exact_join_finds_the_ordering_of_least_total_distance():
    from go2nav.data.ring_windows import best_join
    rng = np.random.default_rng(3)
    a = torch.tensor(rng.uniform(-3, 3, (50, 6, 2)), dtype=torch.float32)
    perm = torch.tensor(np.stack([rng.permutation(6) for _ in range(50)]))
    b = torch.gather(a, 1, perm[..., None].expand(-1, -1, 2))+.01
    join = best_join(a, b)
    assert torch.equal(torch.gather(perm, 1, join), torch.arange(6).expand(50, -1))   # b[join[k]] is a[k]
