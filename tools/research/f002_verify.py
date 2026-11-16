"""Independent scalar point checks and explicit row-resampling checks for F002."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/gated/F002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    return ordered[low] + (position - low) * (ordered[math.ceil(position)] - ordered[low])


def main():
    output = OUT / 'verification.json'
    if output.exists():
        raise FileExistsError('Preserve prior verification')
    path = OUT / 'result.json'
    result = json.loads(path.read_text())
    record = json.loads((ROOT / 'docs/research/records/F002.json').read_text())
    for p, expected in record['protocol']['inputs'].items():
        assert sha(ROOT / p) == expected, p
    assert sha(ROOT / 'docs/research/records/F002.json') == result['record_sha256']
    assert sha(ROOT / 'tools/research/f002_reaction_audit.py') == result['source_sha256']
    documents = {m: json.loads((ROOT / f'outputs/object_dynamics/{m}_reaction.json').read_text())
                 for m in ('m5', 'm6', 'n4')}
    rows = {m: sorted(d['rows'], key=lambda r: (r['episode'], r['t0'])) for m, d in documents.items()}
    reference = rows['m5']
    keys = [(r['episode'], r['t0']) for r in reference]
    assert len(keys) == len(set(keys)) == result['population']['rows'] == 511
    units = sorted(set(k[0] for k in keys))
    assert len(units) == result['population']['episodes'] == 338
    for m, rs in rows.items():
        assert [(r['episode'], r['t0']) for r in rs] == keys
        assert [tuple((r['true'], r['kind'], r['waiting'])) for r in rs] == [
            tuple((r['true'], r['kind'], r['waiting'])) for r in reference]
        for h, summary in result['historical_reconstruction'][m].items():
            assert summary == documents[m][f'{float(h):.1f}s']
    checked, groups_checked, intervals_checked = 0, 0, 0
    def check(actual, expected):
        nonlocal checked
        if expected is None:
            assert actual is None
        else:
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)
        checked += 1
    def avg(values):
        return statistics.fmean(values) if values else None
    members = {u: [i for i, key in enumerate(keys) if key[0] == u] for u in units}
    # Explicit repeated rows, independent of the analysis's frequency-weight matrix.
    rng = np.random.default_rng(2026111602)
    draws = [np.array([i for u in rng.choice(units, len(units)) for i in members[u]]) for _ in range(2000)]
    for j, h in enumerate(('1.6', '3.2', '6.4')):
        truth = [r['true'][j] for r in reference]
        tn = [math.hypot(*t) for t in truth]
        strata = dict(all=list(range(511)), responsive=[i for i, n in enumerate(tn) if n > .05],
                      nonresponsive=[i for i, n in enumerate(tn) if n <= .05],
                      exact_zero=[i for i, n in enumerate(tn) if n <= 1e-6],
                      waiting=[i for i, r in enumerate(reference) if r['waiting']],
                      not_waiting=[i for i, r in enumerate(reference) if not r['waiting']])
        strata.update({f'kind={k}': [i for i, r in enumerate(reference) if r['kind'] == k]
                       for k in sorted(set(r['kind'] for r in reference))})
        errors = {m: [math.dist(r['predicted'][j], truth[i]) for i, r in enumerate(rs)] for m, rs in rows.items()}
        for m, rs in rows.items():
            predictions = [r['predicted'][j] for r in rs]
            pn = [math.hypot(*p) for p in predictions]
            for name, ids in strata.items():
                actual = result['horizons'][h][m][name]
                err = [errors[m][i] for i in ids]
                responsive = [i for i in ids if tn[i] > .05]
                positive = [i for i in ids if pn[i] > .05]
                nonresponsive = [i for i in ids if tn[i] <= .05]
                direction = [i for i in responsive if pn[i] > 1e-8]
                cos = [max(-1, min(1, sum(a*b for a, b in zip(truth[i], predictions[i])) / (tn[i]*pn[i])))
                       for i in direction]
                expected = dict(count=len(ids), episodes=len(set(keys[i][0] for i in ids)),
                                mean_error_m=avg(err), median_error_m=statistics.median(err), p90_error_m=quantile(err, .9),
                                zero_response_error_m=avg([tn[i] for i in ids]),
                                mean_error_minus_zero_m=avg([errors[m][i]-tn[i] for i in ids]),
                                magnitude_mae_m=avg([abs(pn[i]-tn[i]) for i in ids]),
                                true_mean_m=avg([tn[i] for i in ids]), predicted_mean_m=avg([pn[i] for i in ids]),
                                direction_count=len(direction), mean_direction_cosine=avg(cos),
                                mean_direction_angle_deg=avg([math.degrees(math.acos(c)) for c in cos]),
                                responsive_count=len(responsive), predicted_responsive_count=len(positive),
                                response_precision=avg([tn[i] > .05 for i in positive]),
                                response_recall=avg([pn[i] > .05 for i in responsive]),
                                nonresponse_false_positive_rate=avg([pn[i] > .05 for i in nonresponsive]))
                for key, value in expected.items():
                    check(actual[key], value)
                groups_checked += 1
                if name not in ('all', 'responsive', 'nonresponsive'):
                    continue
                mask = np.array([i in set(ids) for i in range(511)])
                for baseline, values in [('zero', tn), ('m5', errors['m5'])]:
                    diff = np.array([e - b for e, b in zip(errors[m], values)])
                    sampled = [float(diff[indices[mask[indices]]].mean()) for indices in draws if mask[indices].any()]
                    entry = actual[f'error_minus_{baseline}_bootstrap']
                    check(entry['nonempty_draws'], len(sampled))
                    check(entry['empty_draws'], 2000-len(sampled))
                    for a, e in zip(entry['ci95'], [quantile(sampled, .025), quantile(sampled, .975)]):
                        check(a, e)
                    if baseline == 'm5':
                        check(actual['mean_error_minus_m5_m'], avg([float(diff[i]) for i in ids]))
                    intervals_checked += 1
    evidence = dict(research_id='F002', created_at=datetime.now(timezone.utc).isoformat(), status='passed',
                    result_sha256=sha(path), verifier_sha256=sha(Path(__file__)),
                    record_sha256=result['record_sha256'], source_sha256=result['source_sha256'],
                    scalar_checks=checked, stratum_model_horizon_groups=groups_checked, interval_checks=intervals_checked,
                    original_summary_groups=9, exact_input_hashes=len(record['protocol']['inputs']),
                    method='math/statistics scalar points; explicit whole-episode repeated-row bootstrap, no analysis helper imports',
                    limitations='Verifies saved-vector calculations, not historical model execution or omitted windows.')
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
