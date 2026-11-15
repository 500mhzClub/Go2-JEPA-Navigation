import numpy as np

from go2nav.eval.balls_binding import discs, measure, summarize


def test_binding_requires_object_masks_not_one_foreground_or_background_mask():
    red, white = np.array([[15., 15.]]), np.array([[[35., 35.], [50., 15.]]])
    full, visible = discs(red, white)
    background = ~full.any(1, keepdims=True)
    masks = np.concatenate((full, background), 1).astype(float)
    result = measure(masks, red, white)
    assert result['bound'].all()
    assert np.all(result['iou'] == 1)
    assert np.array_equal(visible, full)
    permuted = measure(masks[:, [3, 2, 0, 1]], red, white)
    np.testing.assert_array_equal(permuted['iou'], result['iou'])
    merged = np.concatenate((full.any(1, keepdims=True), background), 1).astype(float)
    assert not measure(merged, red, white)['bound'].any()
    assert not measure(np.ones((1, 1, 64, 64)), red, white)['bound'].any()


def test_full_disc_targets_retain_completely_occluded_white_ball():
    red, white = np.array([[15., 15.]]), np.array([[[15., 15.]]])
    full, visible = discs(red, white)
    assert full[:, 0].sum() == full[:, 1].sum() > 0
    assert visible[:, 1].sum() == 0
    masks = np.concatenate((full[:, :1], ~full[:, :1]), 1).astype(float)
    result = measure(masks, red, white)
    assert result['visible_fraction'].tolist() == [[1., 0.]]
    assert result['reused'].all()  # reported ambiguity, not discarded from the primary measure


def test_binding_summary_keeps_misses_and_resamples_whole_episodes():
    rows = [dict(episode=e, ball=k, bound=bool(e == 0), iou=float(e == 0),
                 visible_fraction=1. if k == 0 else 0., reused=False) for e in (0, 1) for k in (0, 1)]
    summary = summarize(rows)
    assert summary['all']['count'] == 4
    assert summary['all']['bound_share'] == .5
    assert summary['all']['bound_ci95'] == [0., 1.]
    assert summary['at_least_90_percent_visible']['count'] == 2
