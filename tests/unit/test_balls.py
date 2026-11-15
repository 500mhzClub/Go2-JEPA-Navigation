"""P25's second world (docs/dynamic/PLAN.md P25): the bouncing balls, their data, and what the models and planner need."""
import numpy as np
import pytest

from go2nav.sim import balls


def test_actions_are_clipped_to_three_pixels_and_smaller_ones_kept():
    a = balls.clip_actions(np.array([[3., 4.], [.3, -.4]]))
    assert np.allclose(np.linalg.norm(a, axis=-1), [3., .5]) and np.allclose(a[0], [1.8, 2.4])


def test_the_red_ball_stops_at_the_walls_and_white_balls_bounce_back_at_their_speed():
    state = balls.State(red=np.array([[4., 60.]]), white=np.array([[[3.5, 30.], [59., 40.]]]),
                        velocity=np.array([[[-2., 1.], [2.5, 0.]]]))
    nxt = balls.step(state, np.array([[-3., 0.]]))
    assert np.allclose(nxt.red, [[balls.LOW, 60.]])
    assert np.allclose(nxt.white[0, 0], [2*balls.LOW-1.5, 31.]) and np.allclose(nxt.velocity[0, 0], [2., 1.])
    assert np.allclose(nxt.white[0, 1], [2*balls.HIGH-61.5, 40.]) and np.allclose(nxt.velocity[0, 1], [-2.5, 0.])
    assert np.allclose(np.linalg.norm(nxt.velocity, axis=-1), np.linalg.norm(state.velocity, axis=-1))


def test_a_ball_is_drawn_where_it_is_to_a_twentieth_of_a_pixel_and_the_red_one_over_the_white():
    red, white = np.array([[20.3, 40.7]]), np.array([[[45.6, 12.2]]])
    frame = balls.render(red, white)[0].astype(float)
    ys, xs = np.mgrid[0:balls.SIZE, 0:balls.SIZE]+.5
    for xy, mask in ((red[0], (frame[..., 0] > 0) & (frame[..., 1] == 0)), (white[0, 0], frame[..., 1] > 0)):
        w = frame[..., 0]*mask
        assert np.hypot((w*xs).sum()/w.sum()-xy[0], (w*ys).sum()/w.sum()-xy[1]) < .05
    over = balls.render(np.array([[30., 30.]]), np.array([[[30., 30.]]]))[0, 30, 30]
    assert tuple(over) == (255, 0, 0)


def test_an_episode_follows_its_actions_and_the_policy_holds_them():
    rng = np.random.default_rng(0)
    ep = balls.episodes(rng, 3, 2, 30)
    assert ep['frames'].shape == (3, 30, 64, 64, 3) and ep['actions'].shape == (3, 29, 2)
    assert ep['white'].shape == (3, 30, 2, 2) and np.linalg.norm(ep['actions'], axis=-1).max() <= 3.+1e-6
    expected = np.clip(ep['red'][:, :-1]+ep['actions'], balls.LOW, balls.HIGH)
    assert np.allclose(ep['red'][:, 1:], expected, atol=1e-5)
    assert ((ep['white'] >= balls.LOW-1e-6) & (ep['white'] <= balls.HIGH+1e-6)).all()
    held = (np.abs(np.diff(ep['actions'], axis=1)).sum(-1) < 1e-9).mean()
    assert .4 < held < .85                                                       # holds of 1-5 steps: about 2/3 repeat


def test_the_red_ball_s_rollout_in_torch_is_the_simulator_s():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(1)
    red, actions = rng.uniform(3, 61, (5, 2)), rng.normal(0, 3, (5, 8, 2))
    state, path = balls.State(red, np.zeros((5, 0, 2)), np.zeros((5, 0, 2))), []
    for k in range(8):
        state = balls.step(state, actions[:, k])
        path.append(state.red)
    out = balls.red_rollout(torch.tensor(red), torch.tensor(actions)).numpy()
    assert np.allclose(out, np.stack(path, 1))


def test_a_data_chunk_is_a_pure_function_of_its_seed():
    from go2nav.data import balls as data
    a, b = data.chunk((3, 'val', 1, 2)), data.chunk((3, 'val', 1, 2))
    assert a[0] == 1 and np.array_equal(a[1]['frames'], b[1]['frames']) and np.array_equal(a[1]['white'], b[1]['white'])
    assert not np.array_equal(a[1]['red'], data.chunk((3, 'train', 1, 2))[1]['red'])


def test_the_fourier_target_does_not_depend_on_the_white_balls_order():
    torch = pytest.importorskip('torch')
    from go2nav.training.balls import fourier_target, frequencies
    w = frequencies()
    red, white = torch.tensor([[10., 20.]]), torch.tensor([[[30., 40.], [50., 12.], [5., 60.]]])
    a, b = fourier_target(red, white, w), fourier_target(red, white[:, [2, 0, 1]], w)
    assert a.shape == (1, 18) and torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, fourier_target(red+1., white, w), atol=1e-3)


def test_a_window_s_actions_move_its_red_ball_and_both_runs_train_on_it(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from go2nav.data import balls as data
    from go2nav.training import balls as training
    monkeypatch.setattr(data, 'SPLITS', dict(train=4, val=4))
    monkeypatch.setattr(data, 'FRAMES', 12)
    monkeypatch.setattr(data, 'CHUNK', 2)
    data.build(tmp_path, ns=(3,), workers=1)
    windows = training.Windows(tmp_path, 3, 'val', 'cpu')
    assert windows.starts == 9
    pixels, actions, red, white = windows.batch(torch.tensor([1, 3]), torch.tensor([0, 8]))
    assert pixels.shape == (2, 4, 3, 64, 64) and actions.shape == (2, 3, 2) and white.shape == (2, 4, 3, 2)
    assert float(pixels.min()) >= -1. and float(pixels.max()) <= 1.
    moved = (red[:, :-1]+actions*3.).clamp(3., 61.)
    assert torch.allclose(red[:, 1:], moved, atol=1e-4)
    sigreg, w = training.SIGReg(num_proj=64), training.frequencies()
    for kind in ('l0', 'l1'):
        model, head = training.build(kind, 0, 'cpu')
        out = training.terms(model, head, sigreg, w, pixels, actions, red, white, training.DEFAULTS)
        assert torch.isfinite(out['loss']) and (('aux_encoded' in out) == (kind == 'l1'))
        out['loss'].backward()
