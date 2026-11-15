"""C002: paired P25 tables and declared bars from literature-reviewed results.

The input JSON is a list of {n, seed, kind, planning, probes, binding} entries.
Paths are result.json files relative to the index (or absolute). Binding is the
shared encoder's result and is required for O1. Missing files remain incomplete;
existing unreviewed or stale artifacts cannot be silently included.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from go2nav.research import atomic_json, digest, provenance, require_record, require_review

KINDS = ('l0', 'l1', 'o0', 'o1')
EXPECTED = [(n, seed, kind) for n in (1, 3, 6) for seed in ((0, 1, 2) if n == 3 else (0,))
            for kind in (KINDS if n == 3 else ('l0', 'l1', 'o1'))]


def paired_difference(left, right, horizon):
    def indexed(rows):
        out = {}
        for row in rows:
            if row['horizon'] != horizon:
                continue
            episode = row['episode']
            if episode in out:
                raise ValueError('More than one task per episode and horizon')
            out[episode] = row
        return out
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError('Planning comparison has unmatched episodes')
    for episode in a:
        if any(a[episode][k] != b[episode][k] for k in ('n', 'start_frame', 'horizon', 'seed')):
            raise ValueError('Planning comparison has different task/CEM identities')
        for key in ('goal_frame', 'start_position', 'goal_position'):
            if key in a[episode] or key in b[episode]:
                if a[episode].get(key) != b[episode].get(key):
                    raise ValueError('Planning comparison has different start/goal observations')
    delta = np.array([float(a[e]['success'])-float(b[e]['success']) for e in sorted(a)])
    rng = np.random.default_rng(2026111513)
    means = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
    return dict(episodes=len(delta), difference=float(delta.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist())


def threshold(value, target, *, less=False):
    return dict(value=value, threshold=target,
                status='incomplete' if value is None else
                ('met' if (value < target if less else value >= target) else 'not_met'))


def gates(models, differences):
    """Point-estimate bars fixed before runs; uncertainty is reported alongside."""
    def success(n, seed, kind):
        return models.get((n, seed, kind), {}).get('success', {}).get('4')
    def probe(n, seed, kind, key, sl):
        values = models.get((n, seed, kind), {}).get('probes', {}).get(key, {}).get('pearson')
        return None if values is None else np.asarray(values)[sl]
    report = dict(reference={}, planning={}, probes={})
    for seed in (0, 1, 2):
        red = probe(3, seed, 'l0', 'current', slice(0, 2))
        report['reference'][str(seed)] = dict(
            l0_h4=threshold(success(3, seed, 'l0'), .10, less=True),
            l0_red_r_mean=threshold(None if red is None else float(red.mean()), .5, less=True),
            l1_h4=threshold(success(3, seed, 'l1'), .50))
        current = probe(3, seed, 'o1', 'current', slice(None))
        future = probe(3, seed, 'o1', 'rollout', slice(2, None))
        report['probes'][str(seed)] = dict(
            current_min_r=threshold(None if current is None else float(current.min()), .9),
            future_white_min_r=threshold(None if future is None else float(future.min()), .8))
    for n in (1, 3, 6):
        for seed in ((0, 1, 2) if n == 3 else (0,)):
            entry = {}
            for reference, bar in (('l0', .30), ('l1', -.10)):
                key = f'n{n}/s{seed}/o1-{reference}/h4'
                diff = differences.get(key)
                entry[reference] = threshold(None if diff is None else diff['difference'], bar)
            report['planning'][f'n{n}/s{seed}'] = entry
    return report


def load_result(base, entry, field, inputs):
    name = entry.get(field)
    if name is None or not (base/name).exists():
        return None, 'missing'
    path = base/name
    review_path = path.with_name('review.json')
    review = require_review(review_path, path)
    inputs.extend((path, review_path))
    if review['status'] not in ('passed', 'measured_negative'):
        return None, review['status']
    data = json.loads(path.read_text())
    seed = data.get('seed', review.get('run_identity', {}).get('seed'))
    if seed is None:
        raise ValueError(f'{path}: legacy result review must attest its verified training seed in run_identity')
    if data['n'] != entry['n'] or seed != entry['seed']:
        raise ValueError(f'{path}: ball count/training seed disagrees with comparison index')
    kind = data.get('kind')
    if field != 'binding' and kind is not None and kind != entry['kind']:
        raise ValueError(f'{path}: model arm disagrees with comparison index')
    cases = path.with_name('cases.npz' if field == 'probes' else 'cases.jsonl')
    if digest(cases) != data['cases_sha256']:
        raise ValueError(f'{cases}: reviewed cases have changed')
    inputs.append(cases)
    if field == 'planning':
        data['cases'] = [json.loads(line) for line in cases.read_text().splitlines()]
        if len(data['cases']) != 3000 or any(row['arm'] != entry['kind'] for row in data['cases']):
            raise ValueError('A complete planning result must contain all 3000 cases of its declared arm')
        for h in (4, 6, 8):
            selected = [r for r in data['cases'] if r['horizon'] == h]
            if len(selected) != 1000 or {r['episode'] for r in selected} != set(range(1000)):
                raise ValueError('Each planning horizon must contain exactly the fixed 1000 episodes')
            if not np.isclose(np.mean([r['success'] for r in selected]), data['arms'][entry['kind']][str(h)]['success']):
                raise ValueError('Planning summary disagrees with its cases')
    if field == 'probes':
        for key in ('current', 'rollout', 'no_rollout', 'shuffled'):
            values = np.asarray(data['measures'][key]['pearson'])
            if values.shape != (2*(entry['n']+1),) or not np.isfinite(values).all():
                raise ValueError('Probe report lacks finite red/white coordinates')
    return data, review['status']


def build(index, out, research_id):
    require_record(research_id)
    index, out = Path(index), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    entries, inputs = {}, [index]
    for entry in json.loads(index.read_text()):
        key = (entry['n'], entry['seed'], entry['kind'])
        if key in entries or key not in EXPECTED:
            raise ValueError('Duplicate or unplanned n/seed/model entry')
        entries[key] = entry
    models, cases = {}, {}
    for key in EXPECTED:
        n, seed, kind = key
        entry = entries.get(key, dict(n=n, seed=seed, kind=kind))
        row = dict(n=n, seed=seed, kind=kind, evidence={})
        for field in ('planning', 'probes', 'binding'):
            data, status = load_result(index.parent, entry, field, inputs)
            row['evidence'][field] = status
            if data is None:
                continue
            if field == 'planning':
                row['success'] = {h: data['arms'][kind][h]['success'] for h in ('4', '6', '8')}
                row['planning'] = data['arms'][kind]
                cases[key] = data['cases']
            else:
                row[field] = data['measures']
        models[key] = row
    differences = {}
    for n, seed, kind in EXPECTED:
        if kind != 'o1' or (n, seed, kind) not in cases:
            continue
        for other in ('l0', 'l1', 'o0'):
            if (n, seed, other) in cases:
                for h in (4, 6, 8):
                    differences[f'n{n}/s{seed}/o1-{other}/h{h}'] = paired_difference(
                        cases[n, seed, kind], cases[n, seed, other], h)
    complete = all('success' in row and 'probes' in row and (row['kind'] != 'o1' or 'binding' in row)
                   for row in models.values())
    report = dict(research_id=research_id, status='awaiting_literature_review', evidence_complete=complete,
                  models=list(models.values()), paired_differences=differences, bars=gates(models, differences),
                  interpretation='Numeric bars only. Literature review is required before scientific conclusions. '
                                 'Missing cells are incomplete; training seeds are never pooled.')
    atomic_json(out/'manifest.json', provenance(research_id, dict(stage='comparison'), inputs))
    atomic_json(out/'result.json', report)
    lines = ['# P25 reviewed measurements', '', 'Training seeds are separate. Missing entries remain incomplete.', '',
             '| n | Seed | Model | H4 success | H6 success | H8 success |', '|---|---|---|---|---|---|']
    for row in models.values():
        values = [f"{row['success'][h]:.3f}" if 'success' in row else 'pending' for h in ('4', '6', '8')]
        lines.append('| '+' | '.join(map(str, [row['n'], row['seed'], row['kind'], *values]))+' |')
    lines += ['', 'See result.json for per-axis probes, binding, paired confidence intervals and declared bars.',
              'The reference pattern is a reproduction check; unexpected L0 success must not be tuned away.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'out', 'research'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    report = build(a.index, a.out, a.research)
    print(json.dumps(dict(evidence_complete=report['evidence_complete'], status=report['status'])))


if __name__ == '__main__':
    main()
