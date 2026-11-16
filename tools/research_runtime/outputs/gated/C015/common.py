"""C015 exact inputs and inherited C014 bounded reserve accounting."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
BASE=Path('outputs/gated/C015')
HELPERS=('costs.py','common.py','diagnostic.py','run.py','check.py')


def need(v,msg):
    if not v:raise ValueError(msg)


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def inherited():
    base=Path('outputs/gated/C007');r=require_review(base/'software/review.json');need(r['status']=='passed','Unreviewed supervision');s=json.loads((base/'software/result.json').read_text());verify_files(base,s['source_sha256'])
    return module('c015_c007_supervision',base/'supervision.py')


def code_hashes():
    return {str(p):digest(p) for folder in ['go2nav','configs'] for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.yaml']}


def inputs():
    route=module('c015_c009_routes',Path('outputs/gated/C009/v2/stages.py'));bound={}
    for name in ['o0_n3_s0','o1_n3_s0']:
        for p in route.inputs(route.route(Path.cwd(),'planning',name)):bound[str(p.resolve())]=digest(p)
        for p in (Path('outputs/gated/C014')/name).glob('*.npz'):bound[str(p.resolve())]=digest(p)
        for split in ['train','val']:
            for f in ['frames.npy','labels.npz','meta.json']:
                p=Path(f'outputs/data/balls/n3/{split}/{f}');bound[str(p.resolve())]=digest(p)
    for p in [BASE/'retirement_review.json',BASE/'retirement_result.json',BASE/'throughput_review.json',BASE/'throughput_observation.json',Path('outputs/gated/C009/v2/runs/planning/o0_n3_s0/review.json')]:bound[str(p.resolve())]=digest(p)
    need(require_review(BASE/'retirement_review.json')['status']=='passed','Retirement not reviewed')
    return bound


def admit(ledger, out):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        need(all(r['end'] is not None for r in rows), 'Open GPU interval')
        remaining = inherited().remaining_seconds(rows, time.time())
        work = min(900, *remaining.values())-45
        need(work > 0, 'Insufficient reserve/global time')
        row = dict(id=uuid.uuid4().hex, research_id='C004', supervisor='C015', out=str(out.resolve()), start=time.time(), end=None)
        rows.append(row)
        atomic_json(ledger, rows)
    return row, remaining, work

def close(ledger, row_id, child):
    need(child is None or child.poll() is not None, 'Owned child remains live')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        found = [r for r in rows if r['id'] == row_id]
        need(len(found) == 1 and found[0]['research_id'] == 'C004' and found[0].get('supervisor') == 'C015'
             and found[0]['end'] is None, 'Wrong interval identity or state')
        need(time.time() >= found[0]['start'], 'Ledger clock reversal')
        found[0].update(end=time.time(), closure='C015: exact owned child confirmed terminal')
        atomic_json(ledger, rows)
