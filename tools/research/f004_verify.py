"""Independent scalar identity/target reconstruction for the exported F004 cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files


def need(value, message):
    if not value:
        raise ValueError(message)


def equal(a, b, message):
    need(np.array_equal(a, b, equal_nan=True), message)


def scalar_match(b, row):
    distance = np.full((6, 4), np.inf)
    visible = []
    for w in range(4):
        point = b['walkers_xy'][row, 8, w].astype(float)
        if (b['in_sight'][row, 8, w] and np.isfinite(point).all()
                and np.linalg.norm(point-b['pose'][row, 8, :2]) <= 3):
            visible.append(w)
            for s in range(6):
                if b['seen'][row, 8, s] and b['id'][row, 8, s] >= 0:
                    distance[s, w] = np.linalg.norm(b['xy'][row, 8, s].astype(float)-point)
    matches = np.full(6, -1, np.int16)
    for s in range(6):
        w = int(np.argmin(distance[s]))
        best = distance[s, w]
        if not np.isfinite(best) or best > .3:
            continue
        row_ties = sum(np.isfinite(d) and abs(d-best) <= 1e-8 for d in distance[s])
        column_best = distance[:, w].min()
        col_ties = sum(np.isfinite(d) and abs(d-column_best) <= 1e-8 for d in distance[:, w])
        if row_ties == col_ties == 1 and int(np.argmin(distance[:, w])) == s:
            matches[s] = w
    return matches, len(visible), int((b['seen'][row, 8] & (b['id'][row, 8] >= 0)).sum())


def verify_episode(job):
    corpus, folder, split, episode, entries = job
    raw = {}
    for entry in entries:
        content = (Path(corpus)/entry['path']).read_bytes()
        need(hashlib.sha256(content).hexdigest() == entry['sha256'], 'Changed raw source')
        raw[entry['role']] = content
    meta = json.loads(raw['parent_metadata'])
    with np.load(io.BytesIO(raw['branch'])) as z:
        b = {k: z[k] for k in ('t0', 'contact', 'pose', 'requested', 'xy', 'id', 'seen', 'velocity', 'walkers_xy', 'waiting', 'in_sight')}
    with np.load(io.BytesIO(raw['parent_episode'])) as z:
        parent = {k: z[k] for k in ('pose', 'requested', 'walkers_xy')}
    arrays = {p.stem: np.load(p, mmap_mode='r') for p in (Path(folder)/split).glob('*.npy')}
    episode_rows = np.flatnonzero(arrays['episode'] == episode)
    eligible = []
    coverage = dict(points=len(b['t0'])//6, parent_future_missing=0, no_match_with_full_future=0,
                    retained=0, visible_initial_walkers=0, matched_initial_walkers=0, seen_initial_slots=0)
    for start in range(0, len(b['t0']), 6):
        mapping, visible, seen = scalar_match(b, start)
        coverage['visible_initial_walkers'] += visible
        coverage['seen_initial_slots'] += seen
        coverage['matched_initial_walkers'] += int((mapping >= 0).sum())
        if b['t0'][start]+64 >= len(parent['pose']):
            coverage['parent_future_missing'] += 1
        elif (mapping < 0).all():
            coverage['no_match_with_full_future'] += 1
        else:
            eligible.append((start, mapping, visible, seen))
    coverage['retained'] = len(eligible)
    equal(arrays['t0'][episode_rows], [b['t0'][v[0]] for v in eligible], 'Retained point population differs')
    checked = 0
    for ix in sorted(set([0, len(eligible)-1])) if eligible else []:
        start, mapping, visible, seen = eligible[ix]
        row, t0 = episode_rows[ix], int(b['t0'][start])
        need(arrays['maze'][row] == meta['maze'] and arrays['kind'][row] == {'none':0, 'rude':1, 'polite':2, 'reactive':3}[meta['kind']], 'Wrong metadata')
        equal(arrays['walker'][row], mapping, 'Wrong identity mapping')
        need(arrays['visible_count'][row] == visible and arrays['seen_count'][row] == seen, 'Wrong point coverage')
        expected_waiting = np.zeros(6, bool)
        for s, w in enumerate(mapping):
            if w >= 0:
                expected_waiting[s] = b['waiting'][start, 8, w]
        equal(arrays['waiting'][row], expected_waiting, 'Wrong waiting label')
        for key, source in [('history_xy','xy'), ('history_seen','seen'), ('history_id','id'),
                            ('history_velocity','velocity'), ('history_pose','pose')]:
            equal(arrays[key][row], b[source][start, [0,4,8]], f'Wrong {key}')
        equal(arrays['past_commands'][row], b['requested'][start, :8], 'Wrong history actions')
        for arm in range(7):
            target = np.zeros((16,6,2), np.float32)
            mask = np.zeros((16,6), bool)
            contact = int(b['contact'][start+arm]) if arm < 6 else -1
            need(arrays['contact'][row,arm] == contact, 'Wrong contact')
            times = [4*(i+1) for i in range(16)]
            for step, h in enumerate(times):
                for slot, walker in enumerate(mapping):
                    if walker < 0 or (contact >= 0 and contact <= h):
                        continue
                    value = b['walkers_xy'][start+arm,8+h,walker] if arm < 6 else parent['walkers_xy'][t0+h,walker]
                    if np.isfinite(value).all():
                        target[step,slot], mask[step,slot] = value, True
            equal(arrays['target'][row,arm], target, 'Wrong physical target')
            equal(arrays['valid'][row,arm], mask, 'Wrong target validity')
            expected_pose = (b['pose'][start+arm,8+np.array(times)] if arm < 6 else parent['pose'][t0+np.array(times)])
            equal(arrays['future_pose'][row,arm], expected_pose, 'Wrong future ego pose')
            commands = b['requested'][start+arm,8:] if arm < 6 else parent['requested'][t0:t0+64]
            equal(arrays['future_commands'][row,arm], commands, 'Wrong future requests')
        checked += 1
    return dict(split=split, episode=episode, scalar_points_checked=checked, coverage=coverage)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='outputs/gated/F004/data')
    args = ap.parse_args()
    folder = Path(args.data)
    need(not (folder/'verification.json').exists(), 'Refusing verification overwrite')
    record, rh = require_record('F004')
    result = json.loads((folder/'result.json').read_text())
    need(result['record_sha256'] == rh, 'Wrong record')
    review = require_review('outputs/gated/F004/software/review.json')
    need(review['status'] == 'passed', 'Software not reviewed')
    verify_files(ROOT, result['source_sha256'])
    verify_files(folder, result['files'])
    manifest_path = record['protocol']['inputs']['branch_manifest']
    need(digest(manifest_path) == record['protocol']['inputs']['branch_manifest_sha256'], 'Changed raw manifest')
    manifest = json.loads(Path(manifest_path).read_text())
    grouped = {}
    for entry in manifest['files']:
        grouped.setdefault((entry['split'],entry['episode']), []).append(entry)
    jobs = [(manifest['corpus'],str(folder),split,episode,entries) for (split,episode), entries in sorted(grouped.items())]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(verify_episode, jobs, chunksize=8):
            rows.append(row)
    summaries, layouts = {}, {}
    for split in ('train','val'):
        selected = [row for row in rows if row['split'] == split]
        summary = {k: sum(row['coverage'][k] for row in selected) for k in selected[0]['coverage']}
        need(summary == result['summary'][split], 'Coverage reconstruction differs')
        stored = json.loads((folder/split/'coverage.json').read_text())
        need(len(stored) == len(selected), 'Wrong coverage rows')
        for a,b in zip(stored,selected):
            need(a['episode'] == b['episode'] and all(a[k] == v for k,v in b['coverage'].items()), 'Wrong episode coverage')
        target = np.load(folder/split/'target.npy', mmap_mode='r')
        valid = np.load(folder/split/'valid.npy', mmap_mode='r')
        need(np.isfinite(target).all(), 'Nonfinite exported target')
        for start in range(0,len(target),1024):
            t,v = target[start:start+1024], valid[start:start+1024]
            need(not t[~v].any(), 'Invalid targets not zeroed')
        layouts[split] = set(np.load(folder/split/'maze.npy').tolist())
        summaries[split] = summary
    need(not layouts['train'] & layouts['val'], 'Layout leakage')
    need(layouts['train'] <= set(range(1200)) and layouts['val'] <= set(range(1200,1300)), 'Layout outside split')
    atomic_json(folder/'verification.json', dict(research_id='F004', status='passed',
                created_at=datetime.now(timezone.utc).isoformat(), result_sha256=digest(folder/'result.json'),
                record_sha256=rh, verifier_sha256=digest(__file__), source_sha256=result['source_sha256'],
                raw_hashes=len(manifest['files']), exported_hashes=len(result['files']), episodes=len(rows),
                scalar_points_checked=sum(r['scalar_points_checked'] for r in rows),
                summaries=summaries, layout_counts={s:len(v) for s,v in layouts.items()},
                scope='Independent initial matching/eligibility for every branch point; scalar physical targets and input alignment for first and last retained point per episode. All export and raw hashes checked. No simulator replay or prediction skill claim.'))
    print(json.dumps(dict(status='passed',episodes=len(rows),scalar_points_checked=sum(r['scalar_points_checked'] for r in rows))))


if __name__ == '__main__':
    main()
