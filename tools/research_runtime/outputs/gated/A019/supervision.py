"""A019 nested intervals consume existing A4/global48, with a30-minute case cap."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid
from go2nav.research import atomic_json, digest, require_review, verify_files

BASE = Path('outputs/gated/A016')
assert require_review(BASE/'software/review.json')['status'] == 'passed'
verify_files(BASE, json.loads((BASE/'software/result.json').read_text())['source_sha256'])
assert digest(BASE/'supervision.py') == '7cfa40983615bcf263b847c900822c48bc420f2c1d975f4d80ce8034086a7939'
spec = importlib.util.spec_from_file_location('a019_owned_parent', BASE/'supervision.py')
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
stop_owned, wait_owned, union_seconds = parent.stop_owned, parent.wait_owned, parent.union_seconds


def remaining(rows, now, science):
    if science != 'A019':
        raise ValueError('Unplanned numerical allocation')
    return parent.remaining_seconds(rows, now)


def admit(ledger, out, science):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):
            raise ValueError('Open GPU interval')
        now = time.time(); left = remaining(rows, now, science)
        work = min(1800, *left.values())-45
        if work <= 0:
            raise ValueError('Insufficient existing A/global allowance including cleanup')
        outer = dict(id=uuid.uuid4().hex, research_id='A019', supervisor_research_id='A019',
                     out=str(out), start=now, end=None)
        rows.append(outer); atomic_json(ledger, rows)
    return outer, left, work


def close_intervals(ledger, outer_id, child, child_out, cwd, science):
    if child is not None and child.poll() is None:
        raise RuntimeError('Child live; leave intervals open')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); rows = json.loads(ledger.read_text())
        matches = [r for r in rows if r['id'] == outer_id]
        if len(matches) != 1:
            raise ValueError('Outer identity missing/ambiguous')
        outer = matches[0]
        if science != 'A019' or outer['research_id'] != science or outer.get('supervisor_research_id') != 'A019' or outer['end'] is not None:
            raise ValueError('Wrong outer identity/state')
        inner = [r for r in rows if r['id'] != outer_id and r['research_id'] == science
                 and (Path(cwd)/r['out']).resolve() == Path(child_out).resolve()]
        if len(inner) > 1 or any(r['start'] < outer['start'] or r.get('supervisor_research_id') for r in inner):
            raise ValueError('Ambiguous/historical child')
        manifest = Path(child_out)/'manifest.json'
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if len(inner) != 1 or m['run_id'] != inner[0]['id']:
                raise ValueError('Child manifest/interval mismatch')
        now = time.time(); changed = []
        if any(now < r['start'] or (r['end'] is not None and r['end'] < r['start']) for r in [outer]+inner):
            raise ValueError('Invalid ledger clock')
        for row in [outer]+inner:
            if row['end'] is None:
                row.update(end=now, closure='A019: owned child terminal confirmed'); changed.append(row['id'])
        atomic_json(ledger, rows)
    return changed
