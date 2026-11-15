import numpy as np

from go2nav.data.slot_tracks import TrackSeries
from go2nav.eval.slot_dynamics import ARMS, match, summarize


def test_absence_and_large_jump_break_identity_and_reset_velocity():
    tracks = TrackSeries(1)
    rows = [tracks.update(t, [[t*.02, 0]], [0, 0]) for t in range(4)]
    assert len({int(r['id'][0]) for r in rows}) == 1
    assert np.allclose(rows[-1]['velocity'][0], [.2, 0])
    absent = tracks.update(4, [[np.nan, np.nan]], [0, 0])
    assert absent['id'][0] == -1 and not absent['seen'][0]
    again = tracks.update(5, [[.1, 0]], [0, 0])
    assert again['id'][0] != rows[0]['id'][0] and np.allclose(again['velocity'], 0)
    jumped = tracks.update(6, [[1., 0]], [0, 0])
    assert jumped['id'][0] != again['id'][0] and np.allclose(jumped['velocity'], 0)


def test_matching_is_one_to_one_and_does_not_reward_duplicate_slots():
    xy = np.array([[1, 1], [1.01, 1]])
    truth = np.array([[1., 1], [4, 4]])
    matching = match(xy, np.ones(2, bool), truth, [0, 1])
    assert matching == {0: 0}


def test_missing_predictions_are_counted_and_occlusion_is_not_discarded():
    rows = []
    for h in (16, 32, 64):
        rows.extend([dict(horizon=h, waiting=True, future_visible=False, future_near=True,
                         error={a: .1 for a in ARMS}),
                     dict(horizon=h, waiting=False, future_visible=True, future_near=True,
                          error={a: None for a in ARMS})])
    detection = [dict(source=s, eligible=2, found=1, predictions=2, switches=0, fragments=0, near=3, near_in_view=2)
                 for s in ('slots', 'engineered')]
    result = summarize(rows, detection)
    assert result['prediction']['initially_visible']['3.2s']['eligible'] == 2
    assert result['prediction']['initially_visible']['3.2s']['arms']['m5_slots']['coverage'] == .5
    assert result['prediction']['after_occlusion']['3.2s']['common_count'] == 1
    assert result['prediction']['legacy_visible']['3.2s']['common_count'] == 0
    assert not result['gates']['passed']
