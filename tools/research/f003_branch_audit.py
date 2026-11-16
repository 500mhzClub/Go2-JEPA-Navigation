"""F003 read-only audit of every existing ring branch and its parent episode."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from go2nav.research import atomic_json, digest, require_record, require_review

OUT = ROOT / 'outputs/gated/F003'
SHAPES = dict(t0=(), contact=(), hold=(), requested=(72, 3), pose=(73, 3), xy=(73, 6, 2),
              seen=(73, 6), id=(73, 6), velocity=(73, 6, 2), memory=(73, 6, 4),
              walkers_xy=(73, 4, 2), waiting=(73, 4), in_sight=(73, 4))
DTYPES = {k: np.dtype('bool' if k in ('hold', 'seen', 'waiting', 'in_sight') else
                     'int32' if k in ('t0', 'id') else 'int16' if k == 'contact' else 'float32') for k in SHAPES}
HISTORY = ('pose', 'xy', 'seen', 'id', 'velocity', 'memory', 'walkers_xy', 'waiting', 'in_sight')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


def frozen(path, expected):
    b = path.read_bytes()
    need(hashlib.sha256(b).hexdigest() == expected, f'Changed input: {path}')
    return b


def inspect(d, parent, meta, split, index):
    need(meta['split'] == split and meta['index'] == index, 'Metadata identity mismatch')
    need(meta['maze'] in (range(1200) if split == 'train' else range(1200, 1300)), 'Layout split mismatch')
    need(set(d) == set(SHAPES), 'Unexpected array schema')
    b = len(d['t0'])
    for k, shape in SHAPES.items():
        need(d[k].shape == (b, *shape) and d[k].dtype == DTYPES[k], f'Shape/dtype: {k}')
    need(b % 6 == 0, 'Incomplete six-way sibling group')
    for k in ('pose', 'requested', 'velocity', 'memory'):
        need(np.isfinite(d[k]).all(), f'Nonfinite required values: {k}')
    for k in ('xy', 'walkers_xy'):
        need(not np.isinf(d[k]).any(), f'Infinite coordinates: {k}')
        need(np.array_equal(np.isfinite(d[k][..., 0]), np.isfinite(d[k][..., 1])), f'Partial NaN coordinate: {k}')
    need(np.all(~d['seen'] | ((d['id'] >= 0) & np.isfinite(d['xy']).all(-1))), 'Invalid seen track')
    need(np.all(~d['in_sight'] | np.isfinite(d['walkers_xy']).all(-1)), 'Invalid sight label')
    need(np.all((d['contact'] == -1) | ((d['contact'] >= 1) & (d['contact'] <= 64))), 'Invalid contact index')
    out = dict(episode=index, split=split, maze=meta['maze'], kind=meta['kind'], branches=b,
               holds=int(d['hold'].sum()), contacts=int((d['contact'] >= 0).sum()), points=b//6,
               placement_seeds=[int(w['walk_seed']) for w in meta['placement']],
               metrics={}, identical_action_pairs=0, different_action_pairs=0,
               visible_initial_walkers=0, matched_initial_walkers=0, seen_initial_slots=0)
    if not b:
        return out
    n = b//6
    g = {k: a.reshape(n, 6, *a.shape[1:]) for k, a in d.items()}
    t0 = g['t0'][:, 0]
    need(np.all(g['t0'] == t0[:, None]) and np.all(np.diff(t0) >= 10), 'Invalid branch grouping/spacing')
    need(np.all(t0 >= 8) and np.all(t0 < len(parent['pose'])), 'Branch point outside parent')
    for k in HISTORY:
        need(equal(g[k][:, :, :9], np.broadcast_to(g[k][:, :1, :9], g[k][:, :, :9].shape)), f'Changed shared history: {k}')
    need(equal(g['requested'][:, :, :8], np.broadcast_to(g['requested'][:, :1, :8], (n, 6, 8, 3))), 'Changed past commands')
    frames = t0[:, None] - 8 + np.arange(9)
    for k in ('pose', 'walkers_xy'):
        need(np.allclose(g[k][:, 0, :9], parent[k][frames], rtol=0, atol=1e-4, equal_nan=True), f'Parent history mismatch: {k}')
    for k in ('waiting', 'in_sight'):
        need(equal(g[k][:, 0, :9], parent[k][frames]), f'Parent history mismatch: {k}')
    need(equal(g['requested'][:, 0, :8], parent['requested'][frames[:, :8]]), 'Parent command mismatch')
    future = np.arange(73)[None] >= (8 + d['contact'][:, None])
    dead = (d['contact'][:, None] >= 0) & future
    for k in ('seen', 'waiting', 'in_sight', 'velocity', 'memory'):
        need(not np.any(d[k][dead]), f'Post-contact leakage: {k}')
    need(np.all(d['id'][dead] == -1), 'Post-contact IDs')
    for k in ('xy', 'walkers_xy'):
        need(np.isnan(d[k][dead]).all(), f'Post-contact coordinates: {k}')
    for i in np.flatnonzero(d['contact'] >= 0):
        cut = 8 + int(d['contact'][i])
        need(not d['requested'][i, cut:].any(), 'Post-contact commands')
        need(equal(d['pose'][i, cut:], np.broadcast_to(d['pose'][i, cut], d['pose'][i, cut:].shape)), 'Post-contact pose')
    # Ground-truth identity is fixed once, at the common initial scene.
    at, truth = g['xy'][:, 0, 8].astype(float), g['walkers_xy'][:, 0, 8].astype(float)
    seen = g['seen'][:, 0, 8] & (g['id'][:, 0, 8] >= 0)
    visible = (g['in_sight'][:, 0, 8] & np.isfinite(truth).all(-1)
               & (np.linalg.norm(truth - g['pose'][:, 0, 8, None, :2], axis=-1) <= 3))
    distance = np.linalg.norm(at[:, :, None] - truth[:, None], axis=-1)
    distance = np.where(seen[:, :, None] & visible[:, None], np.nan_to_num(distance, nan=np.inf), np.inf)
    walker = distance.argmin(2)
    slot = distance.argmin(1)
    assigned = np.take_along_axis(distance, walker[..., None], axis=2)[..., 0]
    mutual = np.take_along_axis(slot, walker, axis=1) == np.arange(6)[None]
    # Count a match only when neither direction has a near tie.
    finite = np.isfinite(distance)
    difference = np.full_like(distance, np.inf)
    np.subtract(distance, assigned[..., None], out=difference, where=finite)
    ties_slot = (np.abs(difference) <= 1e-8).sum(2) > 1
    best_for_walker = distance.min(1)
    difference = np.full_like(distance, np.inf)
    np.subtract(distance, best_for_walker[:, None], out=difference, where=finite)
    ties_walker = (np.abs(difference) <= 1e-8).sum(1) > 1
    matched = seen & (assigned <= .3) & mutual & ~ties_slot & ~np.take_along_axis(ties_walker, walker, axis=1)
    gi, si = np.nonzero(matched)
    wi = walker[gi, si]
    initial_ids = g['id'][gi, 0, 8, si]
    out.update(visible_initial_walkers=int(visible.sum()), matched_initial_walkers=len(gi), seen_initial_slots=int(seen.sum()))
    totals = {str(h): Counter() for h in (16, 32, 64)}
    for a, c in itertools.combinations(range(6), 2):
        identical = np.all(g['requested'][:, a, 8:] == g['requested'][:, c, 8:], axis=(1, 2))
        out['identical_action_pairs'] += int(identical.sum())
        out['different_action_pairs'] += int((~identical).sum())
        if identical.any():
            for k in ('contact', *HISTORY):
                need(equal(g[k][identical, a], g[k][identical, c]), f'Identical actions, different outcome: {k}')
        for h in (16, 32, 64):
            total = totals[str(h)]
            ca, cc = g['contact'][gi, a], g['contact'][gi, c]
            safe = ((ca == -1) | (ca > h)) & ((cc == -1) | (cc > h))
            ta, tc = g['walkers_xy'][gi, a, 8+h, wi].astype(float), g['walkers_xy'][gi, c, 8+h, wi].astype(float)
            truth_valid = safe & np.isfinite(ta).all(1) & np.isfinite(tc).all(1)
            total['initial_mapped_pairs'] += len(gi)
            total['contact_censored_pairs'] += int((~safe).sum())
            total['missing_truth_after_contact_mask'] += int((safe & ~truth_valid).sum())
            total['truth_target_pairs'] += int(truth_valid.sum())
            truth_delta = tc - ta
            truth_norm = np.linalg.norm(truth_delta, axis=1)
            total['truth_responsive_pairs'] += int((truth_valid & (truth_norm > .05)).sum())
            total['truth_zero_pairs'] += int((truth_valid & (truth_norm <= 1e-6)).sum())
            stable = (g['id'][gi, a, 8+h, si] == initial_ids) & (g['id'][gi, c, 8+h, si] == initial_ids)
            shown = g['seen'][gi, a, 8+h, si] & g['seen'][gi, c, 8+h, si]
            pa, pc = g['xy'][gi, a, 8+h, si].astype(float), g['xy'][gi, c, 8+h, si].astype(float)
            total['truth_pairs_changed_id'] += int((truth_valid & ~stable).sum())
            total['truth_pairs_unseen_endpoint'] += int((truth_valid & ~shown).sum())
            valid = truth_valid & stable & shown & np.isfinite(pa).all(1) & np.isfinite(pc).all(1)
            delta = pc[valid] - pa[valid]
            err = np.linalg.norm(delta - truth_delta[valid], axis=1)
            total['track_target_pairs'] += len(err)
            total['error_sum_m'] += float(err.sum())
            total['error_sum_squared_m2'] += float((err**2).sum())
            total['error_max_m'] = max(total['error_max_m'], float(err.max(initial=0)))
            zero = truth_norm[valid] <= 1e-6
            total['track_true_zero_pairs'] += int(zero.sum())
            total['track_false_response_on_true_zero'] += int((zero & (np.linalg.norm(delta, axis=1) > .05)).sum())
            total['track_responsive_pairs'] += int((truth_norm[valid] > .05).sum())
    out['metrics'] = {k: dict(v) for k, v in totals.items()}
    return out


def worker(job):
    corpus, split, index, entries = job
    evidence = {r['role']: r['sha256'] for r in entries}
    try:
        raw = {r['role']: frozen(Path(corpus) / r['path'], r['sha256']) for r in entries}
        meta = json.loads(raw['parent_metadata'])
        with np.load(io.BytesIO(raw['branch']), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        with np.load(io.BytesIO(raw['parent_episode']), allow_pickle=False) as z:
            parent = {k: z[k] for k in ('pose', 'walkers_xy', 'requested', 'waiting', 'in_sight')}
        return dict(inspect(d, parent, meta, split, index), status='passed', inputs=evidence)
    except Exception as e:
        return dict(split=split, episode=index, status='technical_blocked', error=repr(e), inputs=evidence)


def fixture():
    d = {k: np.zeros((6, *s), dtype=DTYPES[k]) for k, s in SHAPES.items()}
    d['t0'][:] = 8; d['contact'][:] = -1; d['hold'][:] = True
    d['id'][:] = -1; d['id'][:, :, 0] = 1
    d['xy'][:] = np.nan; d['xy'][:, :, 0] = 0
    d['walkers_xy'][:] = np.nan; d['walkers_xy'][:, :, 0] = 0
    d['seen'][:, :, 0] = True; d['in_sight'][:, :, 0] = True
    parent = {k: d[k][0, :9].copy() for k in ('pose', 'walkers_xy', 'waiting', 'in_sight')}
    parent['requested'] = np.zeros((9, 3), np.float32)
    meta = dict(index=0, split='train', maze=0, kind='polite', placement=[dict(walk_seed=1)])
    return d, parent, meta


def software():
    d, parent, meta = fixture()
    r = inspect(d, parent, meta, 'train', 0)
    need(r['identical_action_pairs'] == 15 and r['metrics']['32']['error_sum_m'] == 0, 'Zero response fixture')
    d['hold'][1] = False; d['requested'][1, 8:, 0] = .1
    d['walkers_xy'][1, 9:, 0, 0] = .2; d['xy'][1, 9:, 0, 0] = .3
    r = inspect(d, parent, meta, 'train', 0)
    need(r['metrics']['32']['truth_responsive_pairs'] == 5, 'Responsive pair count')
    need(abs(r['metrics']['32']['error_sum_m'] - .5) < 1e-6, 'Known track-target error')
    checks = ['exact identical-action zero responses', 'known five responsive pairs and target errors']
    a, p, m = fixture()
    a['hold'][0] = False; a['requested'][0, 8:24, 0] = .1; a['contact'][0] = 16
    for k in ('seen', 'waiting', 'in_sight', 'velocity', 'memory'): a[k][0, 24:] = 0
    a['id'][0, 24:] = -1
    for k in ('xy', 'walkers_xy'): a[k][0, 24:] = np.nan
    r = inspect(a, p, m, 'train', 0)
    need(r['metrics']['16']['contact_censored_pairs'] == 5 and r['metrics']['16']['truth_target_pairs'] == 10,
         'Contact at horizon must censor all five affected sibling pairs')
    checks.append('valid contact tail and inclusive horizon censoring')
    for kind in ('history', 'shape', 'nonfinite', 'split', 'contact', 'identical_outcome'):
        a, p, m = fixture()
        if kind == 'history': a['xy'][1, 0, 0, 0] = 1
        if kind == 'shape': a['pose'] = a['pose'][:, :-1]
        if kind == 'nonfinite': a['velocity'][0, 5, 0, 0] = np.nan
        if kind == 'split': m['maze'] = 1200
        if kind == 'contact': a['contact'][0] = 1
        if kind == 'identical_outcome': a['xy'][1, 50, 0, 0] = 1
        try: inspect(a, p, m, 'train', 0)
        except ValueError: continue
        raise AssertionError(f'Accepted {kind}')
    checks.append('history/schema/nonfinite/layout/contact/identical-outcome fault refusal')
    try: frozen(Path(__file__), '0'*64)
    except ValueError: checks.append('changed input digest refusal')
    else: raise AssertionError('Accepted stale input')
    return dict(status='passed', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('software', 'audit'))
    args = parser.parse_args()
    rec, rhash = require_record('F003')
    shash = digest(__file__)
    target = OUT / ('software/result.json' if args.mode == 'software' else 'result.json')
    need(not target.exists(), 'Preserve existing result')
    meta = dict(research_id='F003', record_sha256=rhash, source_sha256=shash,
                started_at=datetime.now(timezone.utc).isoformat(), numpy=np.__version__)
    if args.mode == 'software':
        atomic_json(target, dict(software(), **meta)); print(json.dumps(dict(result=str(target), sha256=digest(target)))); return
    review = require_review(OUT / 'software/review.json')
    checked = json.loads((OUT / 'software/result.json').read_text())
    need(review['status'] == 'passed' and checked['source_sha256'] == shash and checked['record_sha256'] == rhash, 'Exact software review required')
    for p, h in rec['protocol']['source_sha256'].items(): frozen(ROOT / p, h)
    manifest = json.loads(frozen(ROOT / rec['protocol']['manifest'], rec['protocol']['manifest_sha256']))
    by_episode = defaultdict(list)
    for r in manifest['files']: by_episode[(r['split'], r['episode'])].append(r)
    expected = {('train', i) for i in range(4500)} | {('val', i) for i in range(500)}
    need(set(by_episode) == expected and len(manifest['files']) == 15000, 'Incomplete input inventory')
    jobs = [(manifest['corpus'], s, i, entries) for (s, i), entries in sorted(by_episode.items())]
    log = OUT / 'per_file.jsonl'; need(not log.exists(), 'Preserve prior per-file results')
    with log.open('x') as f, ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, r in enumerate(pool.map(worker, jobs, chunksize=4), 1):
            f.write(json.dumps(r, allow_nan=False)+'\n')
            if i % 250 == 0:
                f.flush(); print(json.dumps(dict(files=i, total=5000)), flush=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    failed = [r for r in rows if r['status'] != 'passed']
    result = dict(**meta, input_manifest_sha256=rec['protocol']['manifest_sha256'], per_file_sha256=digest(log),
                  files=len(rows), failed_files=len(failed), failure_examples=failed[:20], splits={},
                  finished_at=datetime.now(timezone.utc).isoformat())
    layout_sets, seed_sets = {}, {}
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split and r['status'] == 'passed']
        layout_sets[split] = sorted({r['maze'] for r in selected})
        seed_sets[split] = {s for r in selected for s in r['placement_seeds']}
        total = {k: sum(r[k] for r in selected) for k in ('branches', 'holds', 'contacts', 'points', 'identical_action_pairs',
                 'different_action_pairs', 'visible_initial_walkers', 'matched_initial_walkers', 'seen_initial_slots')}
        total.update(files=len(selected), layouts=layout_sets[split], metrics={})
        for kind in ('all', 'polite', 'rude', 'reactive', 'none'):
            for h in ('16', '32', '64'):
                c = Counter()
                for r in selected:
                    if kind != 'all' and r['kind'] != kind: continue
                    m = r['metrics'].get(h, {})
                    for k, v in m.items(): c[k] = max(c[k], v) if k == 'error_max_m' else c[k] + v
                count, zeros = c['track_target_pairs'], c['track_true_zero_pairs']
                total['metrics'][f'{kind}/{h}'] = dict(c, mean_error_m=c['error_sum_m']/count if count else None,
                     rmse_m=(c['error_sum_squared_m2']/count)**.5 if count else None,
                     false_response_on_true_zero=c['track_false_response_on_true_zero']/zeros if zeros else None)
        result['splits'][split] = total
    result['layout_overlap'] = sorted(set(layout_sets['train']) & set(layout_sets['val']))
    result['placement_seed_overlap_count'] = len(seed_sets['train'] & seed_sets['val'])
    original = json.loads(frozen(Path(manifest['corpus']) / 'branches/summary.json', rec['protocol']['summary_sha256']))
    result['summary_matches'] = all(result['splits'][s][k] == original[s][k] for s in original for k in original[s])
    result['status'] = 'passed' if not failed and not result['layout_overlap'] and result['summary_matches'] else 'technical_blocked'
    atomic_json(target, result)
    print(json.dumps(dict(result=str(target), sha256=digest(target), status=result['status'])))


if __name__ == '__main__':
    main()
