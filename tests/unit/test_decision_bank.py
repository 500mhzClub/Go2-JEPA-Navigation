import numpy as np

FROZEN = dict(GO2NAV_EXACT='1', GO2NAV_RECOVER='1', GO2NAV_WALKER_PRICE='300', GO2NAV_W_STILL='0')
TIMING = ('plan_s', 'wall_s')


def test_the_bank_leaves_its_missions_as_they_were_and_each_source_repeats_its_own_choices(tmp_path, monkeypatch):
    from go2nav.eval import decision_bank as db
    from go2nav.eval import kinematic_missions as km
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    key = ('validation', 12, 0, 4)
    plain = km.run_mission(*key[:3], walkers=4, kind='rude', horizon_s=6.4, terminal='walkers', model='true', frames=150)
    row, arrays = db.build_mission(tmp_path, 'true_rude4', key, per_mission=2, near=50., frames=150)
    assert {k: v for k, v in plain.items() if k not in TIMING} == \
        {k: v for k, v in row.items() if k not in TIMING+('eligible', 'moments')}       # recording changes nothing
    assert len(row['moments']) == 2 and row['eligible'] > 2 and arrays['steps'].shape[1] == 3
    for j, m in enumerate(row['moments']):
        assert len(m['kept']) == len(m['truth']) == len(m['true_cost']) == len(m['kinds']) > 10
        assert arrays[f'tapes_{j}'].shape == (len(m['names']), 64, 3)
        assert np.allclose(np.sum(m['true_terms'], 0), m['true_cost'], atol=.01)         # the terms make up the cost
    true = db.score_mission(tmp_path/'true_rude4'/'12_0_4.json', 'true')
    assert len(true) == 2 and all(r['agrees'] for r in true)                           # the replay is the mission
    assert all(r['true_cost'][r['chosen']]-min(r['true_cost']) < 5. for r in true)      # 100-ms walkers, near the truth
    assert all(c in (0., 1.) for r in true for c in r['chance'])                         # one future: 0 or 1
    row, _ = db.build_mission(tmp_path, 'tracker_rude4', key, per_mission=1, near=50., frames=150)
    tracker = db.score_mission(tmp_path/'tracker_rude4'/'12_0_4.json', 'ring_velocity')
    assert len(tracker) == 1 and tracker[0]['agrees']                                   # the tracker's history, replayed
    report = db.summarise(true+tracker)
    assert report['moments'] == 3 and report['agrees'] == 1. and report['mean_regret_s'] is not None
