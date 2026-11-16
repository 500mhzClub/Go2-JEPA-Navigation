"""C010 operations; GPU time is charged to the underlying scientific record."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid
from go2nav.research import atomic_json,digest,require_review,verify_files

INHERITED=Path('outputs/gated/C005/v2')
assert require_review(INHERITED/'software/review.json')['status']=='passed'
verify_files(INHERITED,json.loads((INHERITED/'software/result.json').read_text())['source_sha256'])
spec=importlib.util.spec_from_file_location('c009_reviewed_supervision',INHERITED/'supervision.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
wait_owned,stop_owned,union_seconds=parent.wait_owned,parent.stop_owned,parent.union_seconds
CAPS={'C001':28*3600}


def remaining(rows,now,science):
    if science not in CAPS:raise ValueError('Unplanned scientific allocation')
    values=parent.remaining_seconds(rows,now)
    values['science_seconds']=CAPS[science]-union_seconds([r for r in rows if r['research_id']==science],now)
    return values


def admit(ledger,out,science):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):raise ValueError('Open GPU interval')
        now=time.time();left=remaining(rows,now,science);work=min(left.values())-45
        if work<=0:raise ValueError('Insufficient existing allowance including cleanup')
        outer=dict(id=uuid.uuid4().hex,research_id=science,supervisor_research_id='C010',out=str(out),start=now,end=None)
        rows.append(outer);atomic_json(ledger,rows)
    return outer,left,work


def close_intervals(ledger,outer_id,child,child_out,cwd,science):
    if child is not None and child.poll() is None:raise RuntimeError('Child live; ledger remains open')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        matches=[r for r in rows if r['id']==outer_id]
        if len(matches)!=1:raise ValueError('Outer identity missing/ambiguous')
        outer=matches[0]
        if outer['research_id']!=science or outer.get('supervisor_research_id')!='C010' or outer['end'] is not None:raise ValueError('Wrong outer identity/state')
        inner=[r for r in rows if r['id']!=outer_id and r['research_id']==science and (Path(cwd)/r['out']).resolve()==Path(child_out).resolve()]
        if len(inner)>1 or any(r['start']<outer['start'] or r.get('supervisor_research_id') for r in inner):raise ValueError('Ambiguous/historical child')
        manifest=Path(child_out)/'manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text())
            if len(inner)!=1 or m['run_id']!=inner[0]['id']:raise ValueError('Child manifest identity mismatch')
        now=time.time();changed=[]
        if any(now<r['start'] or (r['end'] is not None and r['end']<r['start']) for r in [outer]+inner):raise ValueError('Invalid ledger clock')
        for row in [outer]+inner:
            if row['end'] is None:row.update(end=now,closure='C010: owned child terminal confirmed');changed.append(row['id'])
        atomic_json(ledger,rows)
    return changed
