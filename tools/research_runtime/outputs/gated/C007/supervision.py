"""C004-only resource supervision; preserve the reviewed A016 process handling."""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import time
import uuid

from go2nav.research import atomic_json, digest, require_review, verify_files

_base = Path('outputs/gated/A016')
assert require_review(_base/'software/review.json')['status'] == 'passed'
_result = json.loads((_base/'software/result.json').read_text())
assert digest(_base/'supervision.py') == _result['source_sha256']['supervision.py']
_spec = importlib.util.spec_from_file_location('c007_reviewed_owned_process', _base/'supervision.py')
_owned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_owned)
union_seconds, wait_owned, stop_owned = _owned.union_seconds, _owned.wait_owned, _owned.stop_owned
RESERVE_IDS = {'C003', 'C004', 'C007'}


def remaining_seconds(rows, now):
    return dict(global_seconds=48*3600-union_seconds(rows, now),
                reserve_seconds=8*3600-union_seconds([r for r in rows if r['research_id'] in RESERVE_IDS], now))


def admit(ledger, out, local_seconds=1800, cleanup=45):
    if not 0 < local_seconds <= 1800 or cleanup < 0:
        raise ValueError('Invalid C004 local allowance/cleanup')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):
            raise ValueError('Another GPU interval remains open')
        now = time.time()
        remaining = remaining_seconds(rows, now)
        work = min(local_seconds, *remaining.values())-cleanup
        if work <= 0:
            raise ValueError('Insufficient C004/reserve/global time including cleanup')
        outer = dict(id=uuid.uuid4().hex, research_id='C007', out=str(out), start=now, end=None)
        rows.append(outer)
        atomic_json(ledger, rows)
    return outer, remaining, work


def close_intervals(ledger, outer_id, child, child_out, cwd):
    if child is not None and child.poll() is None:
        raise RuntimeError('Owned child is still live; leave accounting open')
    child_out = Path(child_out).resolve()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        outer = [r for r in rows if r['id'] == outer_id]
        if len(outer) != 1 or outer[0]['research_id'] != 'C007' or outer[0]['end'] is not None:
            raise ValueError('C007 outer identity/state mismatch')
        outer = outer[0]
        inner = [r for r in rows if r['research_id'] == 'C004'
                 and (Path(cwd)/r['out']).resolve() == child_out]
        if len(inner) > 1 or any(r['start'] < outer['start'] for r in inner):
            raise ValueError('Ambiguous or historical C004 child interval')
        manifest = child_out/'manifest.json'
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if len(inner) != 1 or m['run_id'] != inner[0]['id']:
                raise ValueError('C004 manifest/interval mismatch')
        now = time.time()
        if any(now < row['start'] or (row['end'] is not None and row['end'] < row['start'])
               for row in [outer]+inner):
            raise ValueError('Ledger clock reversal')
        changed = []
        for row in [outer]+inner:
            if row['end'] is None:
                row.update(end=now, closure='C007: owned C004 child terminal confirmed')
                changed.append(row['id'])
        atomic_json(ledger, rows)
    return changed


def case_paths(root, name):
    match = re.fullmatch(r'(slots|o0|o1)_n([136])_s([012])', name)
    if match is None:
        raise ValueError('Unplanned selected-checkpoint identity')
    kind, n, seed = match[1], int(match[2]), int(match[3])
    if n != 3 and (seed != 0 or kind == 'o0'):
        raise ValueError('Unplanned count/seed/model combination')
    root = Path(root)
    training = root/'outputs/gated/C001'/name
    return dict(name=name, kind=kind, n=n, seed=seed, checkpoint=training/'model.pt',
                parent=training/'review.json', science=root/'outputs/gated/C004'/name,
                operational=root/'outputs/gated/C007/runs'/name,
                encoder_review=root/f'outputs/gated/C004/slots_n{n}_s{seed}/review.json')


def check_parent(result, paths):
    expected = ('C001', paths['kind'], paths['n'], paths['seed'])
    if tuple(result.get(k) for k in ['research_id', 'kind', 'n', 'seed']) != expected:
        raise ValueError('Selected parent result identity mismatch')


def arguments(python, paths, root):
    argv = [str(python), '-m', 'go2nav.eval.slots_checkpoint', '--data', str(Path(root)/'outputs/data/balls'),
            '--research', 'C004', '--checkpoint', str(paths['checkpoint']),
            '--parent-review', str(paths['parent']), '--out', str(paths['science'])]
    if paths['kind'] != 'slots':
        argv += ['--encoder-review', str(paths['encoder_review'])]
    return argv


def consultation(path, paths, root, c004_hash, c007_hash):
    value = json.loads(Path(path).read_text())
    expected = dict(name=paths['name'], c004_record_sha256=c004_hash, c007_record_sha256=c007_hash,
                    checkpoint_sha256=digest(paths['checkpoint']), parent_review_sha256=digest(paths['parent']))
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('C007 consultation identity/protocol/parent mismatch')
    if not value.get('consulted_at') or not value.get('consequences') or not value.get('sources'):
        raise ValueError('Missing substantive preconsultation')
    for source in value['sources']:
        if not all(source.get(k) for k in ['url', 'version', 'sections', 'consequence']):
            raise ValueError('Incomplete consultation source')
    files = {f"outputs/data/balls/n{paths['n']}/{s}/{f}" for s in ['train', 'val']
             for f in ['frames.npy', 'labels.npz', 'meta.json']}
    if set(value.get('data_sha256', {})) != files:
        raise ValueError('C004 consultation must bind all six train/val files')
    verify_files(root, value['data_sha256'])
    return value
