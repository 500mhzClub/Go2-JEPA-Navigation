import json

import pytest

from go2nav.eval.balls_comparison import build, gates, load_result, paired_difference
from go2nav.research import atomic_json, digest


def cases(values):
    return [dict(episode=i, n=3, start_frame=10+i, horizon=4, seed=[123, i], success=value)
            for i, value in enumerate(values)]


def test_paired_uncertainty_preserves_matching_instead_of_independent_resampling():
    same = cases([True, False, True, False])
    result = paired_difference(same, same[::-1], 4)
    assert result['difference'] == 0
    assert result['ci95'] == [0., 0.]
    with pytest.raises(ValueError, match='unmatched'):
        paired_difference(same, same[:3], 4)
    changed = [row | dict(start_frame=row['start_frame']+1) for row in same]
    with pytest.raises(ValueError, match='identities'):
        paired_difference(same, changed, 4)
    with pytest.raises(ValueError, match='More than one'):
        paired_difference(same+same, same, 4)


def test_missing_models_do_not_pass_and_seeds_are_separate():
    models = {(3, 0, 'l0'): dict(success={'4': .05}, probes={'current': {'pearson': [.2, .3]}}),
              (3, 0, 'l1'): dict(success={'4': .6}),
              (3, 0, 'o1'): dict(probes={'current': {'pearson': [.95]*8}, 'rollout': {'pearson': [.1, .2]+[.85]*6}})}
    differences = {'n3/s0/o1-l0/h4': {'difference': .4}, 'n3/s0/o1-l1/h4': {'difference': -.10}}
    report = gates(models, differences)
    assert all(item['status'] == 'met' for item in report['reference']['0'].values())
    assert all(item['status'] == 'met' for item in report['probes']['0'].values())
    assert all(item['status'] == 'met' for item in report['planning']['n3/s0'].values())
    assert all(item['status'] == 'incomplete' for item in report['planning']['n3/s1'].values())
    assert all(item['status'] == 'incomplete' for item in report['planning']['n1/s0'].values())


def test_future_white_bar_excludes_red_but_current_bar_keeps_every_coordinate():
    models = {(3, 0, 'o1'): dict(probes={'current': {'pearson': [.89]+[.99]*7},
                                      'rollout': {'pearson': [.1, .2]+[.85]*6}})}
    report = gates(models, {})
    assert report['probes']['0']['current_min_r']['status'] == 'not_met'
    assert report['probes']['0']['future_white_min_r']['status'] == 'met'


def test_negative_review_is_reportable_but_stale_or_unreviewed_data_is_not(tmp_path):
    cases_path, result, review_path = (tmp_path/name for name in ('cases.jsonl', 'result.json', 'review.json'))
    cases_path.write_text('{}\n')
    atomic_json(result, dict(n=3, seed=0, kind='slots', measures={}, cases_sha256=digest(cases_path)))
    review = dict(research_id='C002', result=result.name, result_sha256=digest(result), reviewed_at='2026-11-15',
                  sources=[dict(url='https://example.org/fixture', version='test', sections='test', consequence='test')],
                  measurements='synthetic fixture', uncertainty='unit test only', explanations=['fixture'],
                  next_step='stop', status='measured_negative')
    atomic_json(review_path, review)
    entry = dict(n=3, seed=0, kind='o1', binding=result.name)
    data, status = load_result(tmp_path, entry, 'binding', [])
    assert data['n'] == 3 and status == 'measured_negative'
    cases_path.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='cases have changed'):
        load_result(tmp_path, entry, 'binding', [])
    review_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_result(tmp_path, entry, 'binding', [])


def test_empty_evidence_index_lists_all_missing_models_without_claiming_success(tmp_path):
    index = tmp_path/'index.json'
    index.write_text('[]')
    report = build(index, tmp_path/'report', 'C002')
    assert not report['evidence_complete']
    assert len(report['models']) == 18
    assert all(row['evidence']['planning'] == 'missing' for row in report['models'])
    saved = json.loads((tmp_path/'report/result.json').read_text())
    assert saved['status'] == 'awaiting_literature_review'
