"""F002: audit saved paired reaction vectors; never run a model or simulator.

Run with the research Python and GPU visibility disabled, from this checkout:
  python tools/research/f002_reaction_audit.py software
  # Review outputs/gated/F002/software/result.json before the actual analysis.
  python tools/research/f002_reaction_audit.py analyse
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from go2nav.research import atomic_json, digest, require_record, require_review

OUT = ROOT / 'outputs/gated/F002'
MODELS = ('m5', 'm6', 'n4')
HORIZONS = (1.6, 3.2, 6.4)


def validate_rows(rows):
    indexed = {}
    for row in rows:
        key = (row['episode'], row['t0'])
        if any(type(v) is not int or v < 0 for v in key) or key in indexed:
            raise ValueError('Invalid or duplicate episode/t0')
        if type(row['waiting']) is not bool or not isinstance(row['kind'], str):
            raise ValueError('Malformed row metadata')
        for field in ('true', 'predicted'):
            a = np.asarray(row[field], dtype=np.float64)
            if a.shape != (3, 2) or not np.isfinite(a).all():
                raise ValueError('Malformed or nonfinite reaction vector')
        indexed[key] = row
    if not indexed:
        raise ValueError('Empty population')
    return indexed


def aligned_rows(documents):
    indexed = {m: validate_rows(d['rows']) for m, d in documents.items()}
    first = indexed[next(iter(indexed))]
    for other in indexed.values():
        if other.keys() != first.keys():
            raise ValueError('Different paired populations')
        for k, row in first.items():
            if any(other[k][f] != row[f] for f in ('true', 'kind', 'waiting')):
                raise ValueError('Changed truth or row stratum')
    keys = sorted(first)
    return {m: [rows[k] for k in keys] for m, rows in indexed.items()}


def frozen_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Changed input: {path}')
    return data


def point_stats(true, predicted):
    tn, pn = np.linalg.norm(true, axis=1), np.linalg.norm(predicted, axis=1)
    error = np.linalg.norm(predicted - true, axis=1)
    responsive, positive = tn > .05, pn > .05
    direction = responsive & (pn > 1e-8)
    cosine = np.clip(np.sum(true[direction] * predicted[direction], axis=1)
                     / (tn[direction] * pn[direction]), -1, 1)
    def mean(a):
        return float(np.mean(a)) if len(a) else None
    return dict(count=len(true), mean_error_m=mean(error),
                median_error_m=float(np.median(error)) if len(error) else None,
                p90_error_m=float(np.quantile(error, .9)) if len(error) else None,
                zero_response_error_m=mean(tn), mean_error_minus_zero_m=mean(error - tn),
                magnitude_mae_m=mean(np.abs(pn - tn)), true_mean_m=mean(tn),
                predicted_mean_m=mean(pn), direction_count=int(direction.sum()),
                mean_direction_cosine=mean(cosine), mean_direction_angle_deg=mean(np.degrees(np.arccos(cosine))),
                responsive_count=int(responsive.sum()), predicted_responsive_count=int(positive.sum()),
                response_precision=mean(responsive[positive]), response_recall=mean(positive[responsive]),
                nonresponse_false_positive_rate=mean(positive[~responsive]))


def bootstrap_weights(groups, draws=2000, seed=2026111602):
    units, inverse = np.unique(groups, return_inverse=True)
    picks = np.random.default_rng(seed).choice(len(units), (draws, len(units)))
    frequencies = np.stack([np.bincount(p, minlength=len(units)) for p in picks])
    return frequencies[:, inverse]


def interval(values, mask, weights):
    w = weights[:, mask]
    denominator = w.sum(axis=1)
    ok = denominator > 0
    sampled = (w[ok] @ values[mask]) / denominator[ok]
    return dict(nonempty_draws=int(ok.sum()), empty_draws=int((~ok).sum()),
                ci95=[float(x) for x in np.quantile(sampled, [.025, .975])] if ok.sum() >= 1900 else None)


def historical_summary(true, predicted, groups):
    # Same centered slope, group sampler and rounding as the frozen legacy source.
    def fit(t, p):
        tc, pc = t - t.mean(axis=0), p - p.mean(axis=0)
        b = float(np.sum(pc * tc) / max(float(np.sum(tc * tc)), 1e-12))
        return b, p.mean(axis=0) - b * t.mean(axis=0)
    b, offset = fit(true, predicted)
    rng, units = np.random.default_rng(2026102701), np.unique(groups)
    members = {u: np.flatnonzero(groups == u) for u in units}
    slopes = []
    for _ in range(1000):
        i = np.concatenate([members[u] for u in rng.choice(units, len(units))])
        slopes.append(fit(true[i], predicted[i])[0])
    return dict(slope=round(b, 3), slope_95=[round(float(x), 3) for x in np.quantile(slopes, [.025, .975])],
                offset_m=round(float(np.linalg.norm(offset)), 3),
                true_moved_share=round(float((np.linalg.norm(true, axis=1) > .05).mean()), 3),
                true_mean_m=round(float(np.linalg.norm(true, axis=1).mean()), 3),
                predicted_mean_m=round(float(np.linalg.norm(predicted, axis=1).mean()), 3))


def software():
    checks = []
    true = np.array([[1., 0.], [0., 2.], [0., 0.]])
    assert point_stats(true, true)['mean_error_m'] == 0
    assert point_stats(true, true)['response_precision'] == 1
    assert point_stats(true, true)['mean_direction_cosine'] == 1
    checks.append('perfect predictions have zero error and correct directions')
    assert point_stats(true, np.zeros_like(true))['mean_error_minus_zero_m'] == 0
    assert point_stats(true, np.zeros_like(true))['direction_count'] == 0
    assert point_stats(true, np.zeros_like(true))['response_precision'] is None
    assert point_stats(true, -true)['mean_error_m'] == 2
    assert point_stats(true, -true)['mean_direction_angle_deg'] == 180
    checks.append('zero and opposite predictions, including empty direction denominators')
    row = dict(episode=0, t0=8, kind='polite', waiting=True, true=true.tolist(), predicted=true.tolist())
    base = dict(rows=[row])
    bad = copy.deepcopy(base); bad['rows'].append(copy.deepcopy(row))
    faults = [bad]
    for change in ('nonfinite', 'shape'):
        bad = copy.deepcopy(base)
        bad['rows'][0]['predicted'] = [[float('nan'), 0.]] * 3 if change == 'nonfinite' else [[0., 0.]]
        faults.append(bad)
    for bad in faults:
        try:
            aligned_rows(dict(a=base, b=bad))
        except ValueError:
            continue
        raise AssertionError('Accepted malformed data')
    for field, value in [('true', [[0., 0.]] * 3), ('episode', 1), ('waiting', False)]:
        bad = copy.deepcopy(base); bad['rows'][0][field] = value
        try:
            aligned_rows(dict(a=base, b=bad))
        except ValueError:
            continue
        raise AssertionError('Accepted mismatched pair')
    checks.append('duplicate, shape, nonfinite, population, truth and stratum mismatch refusal')
    try:
        frozen_bytes(Path(__file__), '0' * 64)
    except ValueError:
        checks.append('stale input digest refusal')
    else:
        raise AssertionError('Accepted stale hash')
    w = bootstrap_weights(np.array([10, 10, 20]))
    assert np.array_equal(w[:, 0], w[:, 1])
    assert np.all(w[:, 0] + w[:, 2] == 2)
    assert interval(np.zeros(3), np.ones(3, bool), w)['ci95'] == [0., 0.]
    assert interval(np.zeros(3), np.zeros(3, bool), w)['ci95'] is None
    checks.append('whole-episode paired bootstrap, exact-null interval and empty-stratum reporting')
    return dict(checks=checks, status='passed')


def analyse(record):
    cached = {p: frozen_bytes(ROOT / p, h) for p, h in record['protocol']['inputs'].items()}
    documents = {m: json.loads(cached[f'outputs/object_dynamics/{m}_reaction.json']) for m in MODELS}
    rows = aligned_rows(documents)
    if any(len(r) != 511 for r in rows.values()):
        raise ValueError('Population no longer equals the preregistered511 rows')
    reference = rows['m5']
    groups = np.array([r['episode'] for r in reference])
    truth = np.array([r['true'] for r in reference], dtype=np.float64)
    predictions = {m: np.array([r['predicted'] for r in rlist], dtype=np.float64) for m, rlist in rows.items()}
    waiting = np.array([r['waiting'] for r in reference])
    kinds = np.array([r['kind'] for r in reference])
    weights = bootstrap_weights(groups)
    result = dict(population=dict(rows=len(reference), episodes=len(np.unique(groups)),
                                  asked={m: documents[m]['asked'] for m in MODELS},
                                  keys_sha256=hashlib.sha256(json.dumps([(r['episode'], r['t0']) for r in reference]).encode()).hexdigest()),
                  historical_reconstruction={}, horizons={}, input_sha256=record['protocol']['inputs'])
    for m in MODELS:
        # Preserve original file order when reconstructing its historical bootstrap.
        raw = documents[m]['rows']
        rt, rp = np.array([r['true'] for r in raw]), np.array([r['predicted'] for r in raw])
        rg = np.array([r['episode'] for r in raw])
        result['historical_reconstruction'][m] = {}
        for j, horizon in enumerate(HORIZONS):
            actual = historical_summary(rt[:, j], rp[:, j], rg)
            if actual != documents[m][f'{horizon:.1f}s']:
                raise ValueError(f'Legacy summary mismatch: {m}/{horizon}: {actual}')
            result['historical_reconstruction'][m][str(horizon)] = actual
    for j, horizon in enumerate(HORIZONS):
        t = truth[:, j]
        norm = np.linalg.norm(t, axis=1)
        strata = dict(all=np.ones(len(t), bool), responsive=norm > .05, nonresponsive=norm <= .05,
                      exact_zero=norm <= 1e-6, waiting=waiting, not_waiting=~waiting)
        strata.update({f'kind={k}': kinds == k for k in np.unique(kinds)})
        results = result['horizons'][str(horizon)] = {}
        base_error = np.linalg.norm(predictions['m5'][:, j] - t, axis=1)
        for m in MODELS:
            p = predictions[m][:, j]
            error = np.linalg.norm(p - t, axis=1)
            results[m] = {}
            for name, mask in strata.items():
                entry = point_stats(t[mask], p[mask])
                entry['episodes'] = len(np.unique(groups[mask]))
                if name in ('all', 'responsive', 'nonresponsive'):
                    entry['error_minus_zero_bootstrap'] = interval(error - norm, mask, weights)
                    entry['mean_error_minus_m5_m'] = float((error - base_error)[mask].mean())
                    entry['error_minus_m5_bootstrap'] = interval(error - base_error, mask, weights)
                results[m][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('software', 'analyse'))
    args = parser.parse_args()
    record, rhash = require_record('F002')
    source_hash = digest(__file__)
    target = OUT / ('software/result.json' if args.mode == 'software' else 'result.json')
    if target.exists():
        raise FileExistsError(f'Preserve existing outcome: {target}')
    if args.mode == 'analyse':
        review = require_review(OUT / 'software/review.json')
        checked = json.loads((OUT / 'software/result.json').read_text())
        if review['status'] != 'passed' or checked['source_sha256'] != source_hash or checked['record_sha256'] != rhash:
            raise ValueError('No exact reviewed software prerequisite')
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = software() if args.mode == 'software' else analyse(record)
        result.update(research_id='F002', record_sha256=rhash, source_sha256=source_hash,
                      created_at=started, numpy=np.__version__, mode=args.mode)
        atomic_json(target, result)
    except Exception as e:
        atomic_json(target.with_name('failure.json'), dict(research_id='F002', record_sha256=rhash,
                    source_sha256=source_hash, created_at=started, mode=args.mode, error=repr(e)))
        raise
    print(json.dumps(dict(result=str(target), sha256=digest(target), mode=args.mode)))


if __name__ == '__main__':
    main()
