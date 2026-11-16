"""F004: shared observed histories and seven physical futures from archived files.

Indices 0..5 are sibling interventions; index 6 is the parent's factual future.
Physical identities and masks are labels, never model features. No simulator runs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

HISTORY = np.array([0, 4, 8])
FUTURE = np.arange(4, 65, 4)
KINDS = {'none': 0, 'rude': 1, 'polite': 2, 'reactive': 3}
INPUTS = ('history_xy', 'history_seen', 'history_id', 'history_velocity', 'history_pose',
          'past_commands', 'future_commands', 'future_pose', 'maze')
SCHEMA = {
    'history_xy': ((3, 6, 2), 'float32'), 'history_seen': ((3, 6), 'bool'),
    'history_id': ((3, 6), 'int32'), 'history_velocity': ((3, 6, 2), 'float32'),
    'history_pose': ((3, 3), 'float32'), 'past_commands': ((8, 3), 'float32'),
    'future_commands': ((7, 64, 3), 'float32'), 'future_pose': ((7, 16, 3), 'float32'),
    'target': ((7, 16, 6, 2), 'float32'), 'valid': ((7, 16, 6), 'bool'),
    'walker': ((6,), 'int16'), 'waiting': ((6,), 'bool'), 'contact': ((7,), 'int16'),
    'episode': ((), 'int32'), 't0': ((), 'int32'), 'maze': ((), 'int32'), 'kind': ((), 'int8'),
    'visible_count': ((), 'int16'), 'seen_count': ((), 'int16'),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def initial_match(xy, seen, ids, truth, visible, pose):
    """Mutual nearest initial matching; -1 marks unmatched slots, including ties."""
    visible = visible & np.isfinite(truth).all(-1) & (np.linalg.norm(truth-pose[..., None, :2], axis=-1) <= 3)
    shown = seen & (ids >= 0)
    distance = np.linalg.norm(xy.astype(float)[..., :, None, :] - truth.astype(float)[..., None, :, :], axis=-1)
    distance = np.where(shown[..., :, None] & visible[..., None, :], np.nan_to_num(distance, nan=np.inf), np.inf)
    walker = distance.argmin(-1)
    slot = distance.argmin(-2)
    best = np.take_along_axis(distance, walker[..., None], -1)[..., 0]
    mutual = np.take_along_axis(slot, walker, -1) == np.arange(6)
    delta = np.full_like(distance, np.inf)
    np.subtract(distance, best[..., None], out=delta, where=np.isfinite(distance))
    slot_tie = (np.abs(delta) <= 1e-8).sum(-1) > 1
    delta = np.full_like(distance, np.inf)
    np.subtract(distance, distance.min(-2)[..., None, :], out=delta, where=np.isfinite(distance))
    walker_tie = (np.abs(delta) <= 1e-8).sum(-2) > 1
    matched = shown & (best <= .3) & mutual & ~slot_tie & ~np.take_along_axis(walker_tie, walker, -1)
    return np.where(matched, walker, -1).astype(np.int16), visible.sum(-1).astype(np.int16), shown.sum(-1).astype(np.int16)


def extract(branch, parent, meta):
    """Pure extraction, independently testable without files, Torch, or a GPU."""
    n = len(branch['t0']) // 6
    need(len(branch['t0']) == n*6, 'Incomplete siblings')
    g = {k: v.reshape(n, 6, *v.shape[1:]) for k, v in branch.items()}
    t0 = g['t0'][:, 0]
    need(np.all(g['t0'] == t0[:, None]), 'Sibling t0 mismatch')
    walker, visible, seen = initial_match(g['xy'][:, 0, 8], g['seen'][:, 0, 8], g['id'][:, 0, 8],
                                         g['walkers_xy'][:, 0, 8], g['in_sight'][:, 0, 8], g['pose'][:, 0, 8])
    full = t0+64 < len(parent['pose'])
    has_match = (walker >= 0).any(-1)
    keep = full & has_match
    coverage = dict(points=n, parent_future_missing=int((~full).sum()),
                    no_match_with_full_future=int((full & ~has_match).sum()), retained=int(keep.sum()),
                    visible_initial_walkers=int(visible.sum()), matched_initial_walkers=int((walker >= 0).sum()),
                    seen_initial_slots=int(seen.sum()))
    g = {k: v[keep] for k, v in g.items()}
    t0, walker, visible, seen = t0[keep], walker[keep], visible[keep], seen[keep]
    p = len(t0)
    out = {k: np.zeros((p, *shape), dtype=dtype) for k, (shape, dtype) in SCHEMA.items()}
    for name, key in [('history_xy', 'xy'), ('history_seen', 'seen'), ('history_id', 'id'),
                      ('history_velocity', 'velocity'), ('history_pose', 'pose')]:
        out[name][:] = g[key][:, 0][:, HISTORY]
    out['past_commands'][:] = g['requested'][:, 0, :8]
    out['future_commands'][:, :6] = g['requested'][:, :, 8:]
    out['future_pose'][:, :6] = g['pose'][:, :, 8+FUTURE]
    out['future_commands'][:, 6] = parent['requested'][t0[:, None]+np.arange(64)]
    out['future_pose'][:, 6] = parent['pose'][t0[:, None]+FUTURE]
    out['contact'][:, :6] = g['contact']
    out['contact'][:, 6] = -1
    # Identity indexes are resolved at t0 only. Future tracking is never a target filter.
    truth = np.concatenate((g['walkers_xy'][:, :, 8+FUTURE],
                            parent['walkers_xy'][t0[:, None]+FUTURE][:, None]), axis=1)
    target = np.take_along_axis(truth, np.maximum(walker, 0)[:, None, None, :, None], axis=3)
    safe = (out['contact'][:, :, None] == -1) | (out['contact'][:, :, None] > FUTURE)
    valid = (walker[:, None, None] >= 0) & safe[..., None] & np.isfinite(target).all(-1)
    out['target'][:] = np.where(valid[..., None], target, 0)
    out['valid'][:] = valid
    out['walker'][:] = walker
    out['waiting'][:] = np.take_along_axis(g['waiting'][:, 0, 8], np.maximum(walker, 0), -1) & (walker >= 0)
    for name, value in [('episode', meta['index']), ('t0', t0), ('maze', meta['maze']), ('kind', KINDS[meta['kind']]),
                        ('visible_count', visible), ('seen_count', seen)]:
        out[name][:] = value
    return out, coverage


def source_job(job):
    corpus, entries = job
    raw = {}
    for e in entries:
        path = Path(corpus)/e['path']
        content = path.read_bytes()
        import hashlib
        need(hashlib.sha256(content).hexdigest() == e['sha256'], f'Changed raw input: {path}')
        raw[e['role']] = content
    meta = json.loads(raw['parent_metadata'])
    with np.load(io.BytesIO(raw['branch']), allow_pickle=False) as z:
        b = {k: z[k] for k in ('t0', 'contact', 'requested', 'pose', 'xy', 'seen', 'id', 'velocity',
                              'walkers_xy', 'waiting', 'in_sight')}
    with np.load(io.BytesIO(raw['parent_episode']), allow_pickle=False) as z:
        parent = {k: z[k] for k in ('pose', 'requested', 'walkers_xy')}
    arrays, coverage = extract(b, parent, meta)
    return arrays, dict(coverage, episode=meta['index'], split=meta['split'])


def manifest_jobs(manifest, split):
    grouped = {}
    for e in manifest['files']:
        if e['split'] == split:
            grouped.setdefault(e['episode'], []).append(e)
    need(len(grouped) == (4500 if split == 'train' else 500), 'Wrong episode population')
    need(all(len(v) == 3 for v in grouped.values()), 'Incomplete raw input inventory')
    return [(manifest['corpus'], grouped[k]) for k in sorted(grouped)]


def export(out):
    record, rh = require_record('F004')
    p = record['protocol']['inputs']
    for name in ('branch_manifest', 'additional_manifest'):
        need(digest(p[name]) == p[name+'_sha256'], f'Changed {name}')
    need(digest('outputs/gated/F003/review.json') == p['branch_review_sha256'], 'Changed branch review')
    review = require_review('outputs/gated/F004/software/review.json')
    need(review['status'] == 'passed', 'Software not reviewed')
    software = json.loads(Path('outputs/gated/F004/software/result.json').read_text())
    need(software['record_sha256'] == rh, 'Stale software protocol')
    verify_files(Path.cwd(), software['source_sha256'])
    manifest = json.loads(Path(p['branch_manifest']).read_text())
    additional = json.loads(Path(p['additional_manifest']).read_text())
    verify_files(manifest['corpus'], additional['rasters'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary, hashes = {}, {}
    for split in ('train', 'val'):
        parts = {k: [] for k in SCHEMA}
        coverage = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for arrays, row in pool.map(source_job, manifest_jobs(manifest, split), chunksize=8):
                coverage.append(row)
                for k, v in arrays.items():
                    parts[k].append(v)
        folder = out/split
        folder.mkdir()
        for key in SCHEMA:
            path = folder/(key+'.npy')
            np.save(path, np.concatenate(parts.pop(key)), allow_pickle=False)
            hashes[str(path.relative_to(out))] = digest(path)
        atomic_json(folder/'coverage.json', coverage)
        hashes[str((folder/'coverage.json').relative_to(out))] = digest(folder/'coverage.json')
        summary[split] = {k: sum(r[k] for r in coverage) for k in coverage[0] if k not in ('episode', 'split')}
        print(json.dumps(dict(split=split, **summary[split])), flush=True)
    result = dict(research_id='F004', status='awaiting_verification_and_review',
                  created_at=datetime.now(timezone.utc).isoformat(), record_sha256=rh,
                  source_sha256=software['source_sha256'], input_manifest_sha256=p,
                  files=hashes, summary=summary, software_review_sha256=digest('outputs/gated/F004/software/review.json'))
    atomic_json(out/'result.json', result)
    return result


class Cache:
    def __init__(self, folder, split):
        self.folder = Path(folder)
        self.arrays = {k: np.load(self.folder/split/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in SCHEMA}
        self.n = len(self.arrays['t0'])
        for key, (shape, dtype) in SCHEMA.items():
            need(self.arrays[key].shape == (self.n, *shape) and self.arrays[key].dtype == np.dtype(dtype), f'Bad cache {key}')

    def take(self, points, futures):
        """Flat trajectory rows; two adjacent rows must form each training pair."""
        points, futures = np.asarray(points), np.asarray(futures)
        need(points.ndim == futures.ndim == 1 and len(points) == len(futures), 'Unaligned selection')
        need(np.all((points >= 0) & (points < self.n)) and np.all((futures >= 0) & (futures < 7)), 'Selection out of range')
        selected = {}
        for key, array in self.arrays.items():
            selected[key] = array[points, futures] if key in ('future_commands', 'future_pose', 'target', 'valid', 'contact') else array[points]
        return selected


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='outputs/gated/F004/data')
    args = ap.parse_args()
    export(args.out)
