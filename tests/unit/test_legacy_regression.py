from go2nav.eval.legacy_regression import compare


def test_legacy_comparison_checks_every_cohort_and_exact_counts():
    reference = {'all': {'count': 13090, 'model': .0279}, 'by': {'waiting': {'count': 3, 'model': .0232}}}
    assert compare(reference, reference) == []
    near = {'all': {'count': 13090, 'model': .0280}, 'by': reference['by']}
    assert compare(reference, near) == []
    wrong_count = {'all': {'count': 13089, 'model': .0279}, 'by': reference['by']}
    assert compare(reference, wrong_count)[0]['path'] == '/all/count'
    assert compare(reference, {'all': reference['all']})[0]['reason'] == 'keys differ'
    wrong_subgroup = {'all': reference['all'], 'by': {'waiting': {'count': 3, 'model': .04}}}
    assert compare(reference, wrong_subgroup)[0]['path'] == '/by/waiting/model'
    assert compare({'metric': 1.}, {'metric': float('nan')})
