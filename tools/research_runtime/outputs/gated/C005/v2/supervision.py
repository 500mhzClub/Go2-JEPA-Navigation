"""C005 allocation accounting; reuse the unchanged reviewed A016 process handling."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid

from go2nav.research import atomic_json, digest, require_review

BASE = Path.cwd()/'outputs/gated/A016'
assert require_review(BASE/'software/review.json')['status'] == 'passed'
base_result = json.loads((BASE/'software/result.json').read_text())
assert digest(BASE/'supervision.py') == base_result['source_sha256']['supervision.py']
spec = importlib.util.spec_from_file_location('reviewed_a016_supervision', BASE/'supervision.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
union_seconds, wait_owned, stop_owned = base.union_seconds, base.wait_owned, base.stop_owned
identity_matches = base.identity_matches
BC_IDS = {'B001', 'C001', 'C002', 'C005'}


def remaining_seconds(rows, now):
    return dict(global_seconds=48*3600-union_seconds(rows, now),
                bc_seconds=28*3600-union_seconds([r for r in rows if r['research_id'] in BC_IDS], now))


def admit(ledger, out, reserve=45):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):
            raise ValueError('Another GPU interval remains open')
        now = time.time()
        remaining = remaining_seconds(rows, now)
        work = min(remaining.values())-reserve
        if work <= 0:
            raise ValueError('Insufficient B/C or global budget including cleanup reserve')
        outer = dict(id=uuid.uuid4().hex, research_id='C005', out=str(out), start=now, end=None)
        rows.append(outer)
        atomic_json(ledger, rows)
    return outer, remaining, work


def close_intervals(ledger, outer_id, child, child_out, cwd):
    if child is not None and child.poll() is None:
        raise RuntimeError('Refusing ledger closure: child remains live')
    child_out = Path(child_out).resolve()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        outer = [r for r in rows if r['id'] == outer_id]
        if len(outer) != 1 or outer[0]['research_id'] != 'C005' or outer[0]['end'] is not None:
            raise ValueError('Outer interval identity/state mismatch')
        outer = outer[0]
        inner = [r for r in rows if r['research_id'] == 'C001'
                 and (Path(cwd)/r['out']).resolve() == child_out]
        if len(inner) > 1 or any(r['start'] < outer['start'] for r in inner):
            raise ValueError('Ambiguous or historical child interval')
        manifest = child_out/'manifest.json'
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if len(inner) != 1 or m['run_id'] != inner[0]['id']:
                raise ValueError('Child manifest/interval mismatch')
        now = time.time()
        changed = []
        for row in [outer]+inner:
            if row['end'] is None:
                if now < row['start']:
                    raise ValueError('Ledger clock reversal')
                row.update(end=now, closure='C005: owned child terminal confirmed')
                changed.append(row['id'])
        atomic_json(ledger, rows)
    return changed


def consultation(path, name, record_hash):
    value = json.loads(Path(path).read_text())
    if value.get('name') != name or value.get('c001_record_sha256') != record_hash:
        raise ValueError('Consultation belongs to another experiment/protocol')
    if not value.get('consulted_at') or not value.get('consequences') or not value.get('sources'):
        raise ValueError('Missing actual pre-experiment consultation')
    for source in value['sources']:
        if not all(source.get(k) for k in ['url', 'version', 'sections', 'consequence']):
            raise ValueError('Incomplete consultation source')
    return value
