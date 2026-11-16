"""C013 exact C012 lineage and one existing-reserve interval."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

BASE=Path('outputs/gated/C013')
NAME='o0_n3_s0'


def prior():
    base=Path('outputs/gated/C012')
    assert require_review(base/'software/review.json')['status']=='passed'
    software=json.loads((base/'software/result.json').read_text())
    verify_files(base,software['source_sha256'])
    spec=importlib.util.spec_from_file_location('c013_prior_c012',base/'common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def inherited():
    return prior().inherited()


def inputs():
    record,_=require_record('C013');bound,source=prior().inputs()
    base=Path('outputs/gated/C012')
    for directory in [base/'o0_n3_s0',base/'operations/o0_n3_s0']:
        assert require_review(directory/'review.json')['status']=='passed'
        for p in directory.iterdir():
            if p.is_file():bound[str(p.resolve())]=digest(p)
    assert require_review(base/'o0_n3_s0/inspection_review.json')['status']=='passed'
    op=json.loads((base/'operations/o0_n3_s0/result.json').read_text())
    verify_files(base/'o0_n3_s0',op['child_artifact_sha256'])
    assert op['returncode']==0 and op['child_terminal']
    for p in [*base.glob('*.py'),base/'software/result.json',base/'software/review.json',
              Path('docs/research/records/C013.json')]:bound[str(p.resolve())]=digest(p)
    assert digest('outputs/gated/C001/o0_n3_s0/model.pt')==record['protocol']['checkpoint_sha256']
    return bound,source


def admit(ledger,out,limit=1800,cleanup=45):
    assert limit==1800 and cleanup==45
    base=inherited()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):raise ValueError('Open GPU interval')
        now=time.time();remaining=base.remaining_seconds(rows,now)
        work=min(limit,*remaining.values())-cleanup
        if work<=0:raise ValueError('Insufficient reserve/global time')
        row=dict(id=uuid.uuid4().hex,research_id='C004',supervisor='C013',out=str(out.resolve()),start=now,end=None)
        rows.append(row);atomic_json(ledger,rows)
    return row,remaining,work


def close(ledger,row_id,child):
    if child is not None and child.poll() is None:raise RuntimeError('Owned child still live')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        found=[r for r in rows if r['id']==row_id]
        if len(found)!=1 or found[0]['research_id']!='C004' or found[0].get('supervisor')!='C013' or found[0]['end'] is not None:
            raise ValueError('Wrong C013 interval identity/state')
        assert time.time()>=found[0]['start']
        found[0].update(end=time.time(),closure='C013: exact owned child terminal confirmed');atomic_json(ledger,rows)
