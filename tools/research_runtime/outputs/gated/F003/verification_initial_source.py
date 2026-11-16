"""Independent F003 inventory, totals, and scalar real-file target verification."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/gated/F003'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def scalar(d):
    out = {str(h): Counter() for h in (16, 32, 64)}
    matched_count = 0
    for first in range(0, len(d['t0']), 6):
        pairs = []
        seen = [k for k in range(6) if d['seen'][first, 8, k] and d['id'][first, 8, k] >= 0]
        visible = [j for j in range(4) if d['in_sight'][first, 8, j]
                   and math.dist(d['walkers_xy'][first, 8, j], d['pose'][first, 8, :2]) <= 3]
        for k in seen:
            if not visible: continue
            distances = [(math.dist(d['xy'][first, 8, k], d['walkers_xy'][first, 8, j]), j) for j in visible]
            distance, j = min(distances)
            if distance > .3 or sum(abs(x-distance) <= 1e-8 for x, _ in distances) != 1: continue
            reverse = [(math.dist(d['xy'][first, 8, a], d['walkers_xy'][first, 8, j]), a) for a in seen]
            best, selected = min(reverse)
            if selected == k and sum(abs(x-best) <= 1e-8 for x, _ in reverse) == 1: pairs.append((k, j))
        matched_count += len(pairs)
        for a, b in itertools.combinations(range(first, first+6), 2):
            for h in (16, 32, 64):
                c = out[str(h)]
                for k, j in pairs:
                    c['initial_mapped_pairs'] += 1
                    if any(0 <= d['contact'][q] <= h for q in (a, b)):
                        c['contact_censored_pairs'] += 1; continue
                    ta, tb = d['walkers_xy'][a, 8+h, j], d['walkers_xy'][b, 8+h, j]
                    if not all(math.isfinite(float(x)) for xy in (ta, tb) for x in xy):
                        c['missing_truth_after_contact_mask'] += 1; continue
                    c['truth_target_pairs'] += 1
                    td = [float(y)-float(x) for x, y in zip(ta, tb)]
                    norm = math.hypot(*td)
                    c['truth_responsive_pairs'] += int(norm > .05)
                    c['truth_zero_pairs'] += int(norm <= 1e-6)
                    stable = all(d['id'][q, 8+h, k] == d['id'][first, 8, k] for q in (a, b))
                    shown = all(d['seen'][q, 8+h, k] for q in (a, b))
                    c['truth_pairs_changed_id'] += int(not stable)
                    c['truth_pairs_unseen_endpoint'] += int(not shown)
                    pa, pb = d['xy'][a, 8+h, k], d['xy'][b, 8+h, k]
                    if not (stable and shown and all(math.isfinite(float(x)) for xy in (pa, pb) for x in xy)): continue
                    pd = [float(y)-float(x) for x, y in zip(pa, pb)]
                    error = math.dist(pd, td)
                    c['track_target_pairs'] += 1
                    c['error_sum_m'] += error
                    c['error_sum_squared_m2'] += error*error
                    c['error_max_m'] = max(c['error_max_m'], error)
                    c['track_true_zero_pairs'] += int(norm <= 1e-6)
                    c['track_false_response_on_true_zero'] += int(norm <= 1e-6 and math.hypot(*pd) > .05)
                    c['track_responsive_pairs'] += int(norm > .05)
    return matched_count, out


def main():
    output = OUT / 'verification.json'
    assert not output.exists(), 'Preserve previous verification'
    record = json.loads((ROOT / 'docs/research/records/F003.json').read_text())
    result = json.loads((OUT / 'result.json').read_text())
    assert digest(ROOT / 'docs/research/records/F003.json') == result['record_sha256']
    assert digest(ROOT / 'tools/research/f003_branch_audit.py') == result['source_sha256']
    assert digest(OUT / 'inputs.json') == record['protocol']['manifest_sha256']
    assert digest(OUT / 'per_file.jsonl') == result['per_file_sha256']
    manifest = json.loads((OUT / 'inputs.json').read_text())
    corpus = Path(manifest['corpus'])
    for f in manifest['files']:
        assert digest(corpus / f['path']) == f['sha256'], f['path']
    rows = [json.loads(l) for l in (OUT / 'per_file.jsonl').read_text().splitlines()]
    keys = [(r['split'], r['episode']) for r in rows]
    expected = {('train', i) for i in range(4500)} | {('val', i) for i in range(500)}
    assert len(rows) == len(set(keys)) == 5000 and set(keys) == expected
    lookup = {(r['split'], r['episode'], r['role']): r for r in manifest['files']}
    assert len(lookup) == len(manifest['files']) == 15000
    assert all(r['status'] == 'passed' for r in rows) and result['failed_files'] == 0
    scalar_checks, scalar_files, total_checks = 0, [], 0
    for split in ('train', 'val'):
        subset = [r for r in rows if r['split'] == split]
        for r in subset:
            for role, h in r['inputs'].items(): assert lookup[(split, r['episode'], role)]['sha256'] == h
            metadata = json.loads((corpus / split / f'{r["episode"]:05d}.json').read_text())
            assert r['maze'] == metadata['maze'] and r['kind'] == metadata['kind']
        total = result['splits'][split]
        for k in ('branches', 'holds', 'contacts', 'points', 'identical_action_pairs', 'different_action_pairs',
                  'visible_initial_walkers', 'matched_initial_walkers', 'seen_initial_slots'):
            assert total[k] == sum(r[k] for r in subset); total_checks += 1
        assert total['layouts'] == sorted({r['maze'] for r in subset})
        for key, aggregate in total['metrics'].items():
            kind, h = key.split('/')
            selected = [r['metrics'][h] for r in subset if h in r['metrics'] and (kind == 'all' or r['kind'] == kind)]
            for field in set().union(*(m.keys() for m in selected)) if selected else ():
                values = [m.get(field, 0) for m in selected]
                expected_value = max(values) if field == 'error_max_m' else math.fsum(values)
                assert math.isclose(aggregate[field], expected_value, rel_tol=1e-10, abs_tol=1e-10)
                total_checks += 1
            n, nz = aggregate['track_target_pairs'], aggregate['track_true_zero_pairs']
            for field, value in [('mean_error_m', aggregate.get('error_sum_m', 0)/n if n else None),
                                 ('rmse_m', math.sqrt(aggregate.get('error_sum_squared_m2', 0)/n) if n else None),
                                 ('false_response_on_true_zero', aggregate.get('track_false_response_on_true_zero', 0)/nz if nz else None)]:
                assert aggregate[field] is None if value is None else math.isclose(aggregate[field], value, abs_tol=1e-12)
                total_checks += 1
        chosen = next(r for r in subset if r['matched_initial_walkers'])
        f = corpus / 'branches' / split / f'{chosen["episode"]:05d}.npz'
        with np.load(f, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
        matched, manual = scalar(d)
        assert matched == chosen['matched_initial_walkers']
        for h, values in chosen['metrics'].items():
            for key, value in values.items():
                assert math.isclose(manual[h][key], value, rel_tol=1e-10, abs_tol=1e-10), (split, h, key)
                scalar_checks += 1
        scalar_files.append(dict(split=split, episode=chosen['episode'], matched=matched, sha256=digest(f)))
    assert not set(result['splits']['train']['layouts']) & set(result['splits']['val']['layouts'])
    seed_sets = [{s for r in rows if r['split'] == split for s in r['placement_seeds']} for split in ('train', 'val')]
    assert len(seed_sets[0] & seed_sets[1]) == result['placement_seed_overlap_count'] == 0
    original = json.loads((corpus / 'branches/summary.json').read_text())
    assert all(result['splits'][s][k] == original[s][k] for s in original for k in original[s])
    verification = dict(research_id='F003', status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                        result_sha256=digest(OUT / 'result.json'), record_sha256=result['record_sha256'],
                        verifier_sha256=digest(Path(__file__)), source_sha256=result['source_sha256'],
                        exact_input_hashes=15000, per_file_rows=5000, total_checks=total_checks,
                        scalar_checks=scalar_checks, scalar_files=scalar_files,
                        limits='All-file structural checks are in the main audit; independent scalar target reconstruction covers two entire episode files. No simulator replay or model inference.')
    output.write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification))


if __name__ == '__main__': main()
