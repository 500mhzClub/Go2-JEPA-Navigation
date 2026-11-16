"""D003 cumulative eight-hour allocation; exact reviewed A016 process handling."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid
from go2nav.research import atomic_json, digest, require_review

BASE=Path('outputs/gated/A016')
assert require_review(BASE/'software/review.json')['status']=='passed'
assert digest(BASE/'supervision.py')==json.loads((BASE/'software/result.json').read_text())['source_sha256']['supervision.py']
spec=importlib.util.spec_from_file_location('d003_reviewed_process',BASE/'supervision.py')
owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
union_seconds,wait_owned,stop_owned=owned.union_seconds,owned.wait_owned,owned.stop_owned
D_IDS={'D001','D002','D003'}


def remaining_seconds(rows,now):
    return dict(global_seconds=48*3600-union_seconds(rows,now),
                d_seconds=8*3600-union_seconds([r for r in rows if r['research_id'] in D_IDS],now))


def admit(ledger,out):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):raise ValueError('Another GPU interval remains open')
        now=time.time();remaining=remaining_seconds(rows,now);work=min(remaining.values())-45
        if work<=0:raise ValueError('Insufficient D/global time including cleanup')
        outer=dict(id=uuid.uuid4().hex,research_id='D003',out=str(out),start=now,end=None)
        rows.append(outer);atomic_json(ledger,rows)
    return outer,remaining,work


def close_intervals(ledger,outer_id,child,child_out,cwd):
    if child is not None and child.poll() is None:raise RuntimeError('Child live; accounting stays open')
    child_out=Path(child_out).resolve()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        outer=[r for r in rows if r['id']==outer_id]
        if len(outer)!=1 or outer[0]['research_id']!='D003' or outer[0]['end'] is not None:raise ValueError('Outer identity/state mismatch')
        outer=outer[0]
        inner=[r for r in rows if r['research_id']=='D001' and (Path(cwd)/r['out']).resolve()==child_out]
        if len(inner)>1 or any(r['start']<outer['start'] for r in inner):raise ValueError('Ambiguous/historical child interval')
        if (child_out/'manifest.json').exists():
            m=json.loads((child_out/'manifest.json').read_text())
            if len(inner)!=1 or m['run_id']!=inner[0]['id']:raise ValueError('Child manifest/interval mismatch')
        now=time.time();changed=[]
        if any(now<r['start'] or (r['end'] is not None and r['end']<r['start']) for r in [outer]+inner):raise ValueError('Ledger clock reversal')
        for r in [outer]+inner:
            if r['end'] is None:r.update(end=now,closure='D003: owned child terminal confirmed');changed.append(r['id'])
        atomic_json(ledger,rows)
    return changed
