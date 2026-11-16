"""C005 v2 allocation correction; no process, GPU, or shared-ledger mutation."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files
from supervision import admit, remaining_seconds

record, record_hash = require_record('C005')
assert record['revision'] == 2
v1 = Path('outputs/gated/C005');base = v1/'v2';out=base/'software'
out.mkdir(exist_ok=False)
shared=Path('outputs/gated/gpu_intervals.json');before=digest(shared)
files=['supervision.py','run_stage.py','check_revision.py'];checks={};started=time.time()
old_result=json.loads((v1/'software/result.json').read_text())
require_review(v1/'software/review.json');verify_files(v1,old_result['source_sha256'])

try:
    old=(v1/'supervision.py').read_text()
    assert (base/'supervision.py').read_text()==old.replace(
        "BC_IDS = {'B001', 'C001', 'C002', 'C004', 'C005'}",
        "BC_IDS = {'B001', 'C001', 'C002', 'C005'}")
    assert (base/'run_stage.py').read_text()==(v1/'run_stage.py').read_text().replace(
        "base = ROOT/'outputs/gated/C005'", "base = ROOT/'outputs/gated/C005/v2'")
    for f in files:ast.parse((base/f).read_text())
    checks['only_allocation_constant_and_version_path_changed']=True
    rows=[dict(start=10,end=20,research_id='C005'),dict(start=12,end=19,research_id='C001'),
          dict(start=30,end=60,research_id='C004'),dict(start=70,end=72,research_id='C003'),
          dict(start=80,end=83,research_id='D001'),dict(start=90,end=94,research_id='A003')]
    assert remaining_seconds(rows,100)==dict(global_seconds=48*3600-49,bc_seconds=28*3600-10)
    checks['overlap_union_and_C004_reserve_separation']=True
    for rid in ['B001','C001','C002','C005']:
        assert remaining_seconds([dict(start=0,end=1,research_id=rid)],2)==dict(global_seconds=48*3600-1,bc_seconds=28*3600-1)
    for rid in ['C003','C004','D001','A003']:
        assert remaining_seconds([dict(start=0,end=1,research_id=rid)],2)==dict(global_seconds=48*3600-1,bc_seconds=28*3600)
    checks['each_declared_allocation_ID']=True
    with tempfile.TemporaryDirectory(prefix='c005_v2_') as tmp:
        root=Path(tmp);ledger=root/'ledger.json'
        # Intentionally oversized fake reserve use tests classification, not permission to spend it.
        atomic_json(ledger,[dict(id='reserve',start=0,end=28*3600,research_id='C004',out='fake')])
        _,remaining,work=admit(ledger,root/'out')
        assert remaining==dict(global_seconds=20*3600,bc_seconds=28*3600) and work==20*3600-45
        checks['reserve_use_reduces_global_only_and_global_caps_admission']=True
        for name,rid,hours in [('BC','C001',28),('global','C004',48)]:
            atomic_json(ledger,[dict(id='used',start=0,end=hours*3600,research_id=rid,out='fake')]);h=digest(ledger)
            try:admit(ledger,root/'out')
            except ValueError:pass
            else:raise AssertionError(name+' exhaustion accepted')
            assert digest(ledger)==h
            checks[name+'_exhaustion_refused_without_mutation']=True
    rows=json.loads(shared.read_text());now=time.time()
    assert not any(r['research_id']=='C004' for r in rows)
    spec=importlib.util.spec_from_file_location('c005_v1',v1/'supervision.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    assert old.remaining_seconds(rows,now)==remaining_seconds(rows,now)
    checks['live_v1_allowance_unchanged_without_C004_intervals']=True
    verify_files(v1,old_result['source_sha256'])
    assert digest(shared)==before and 'torch' not in sys.modules
    checks['v1_source_and_shared_ledger_unchanged_no_torch']=True
    atomic_json(out/'result.json',dict(research_id='C005',revision=2,status='passed',checks=checks,
        elapsed_seconds=time.time()-started,research_sha256=record_hash,source_sha256={f:digest(base/f) for f in files},
        inherited_v1_review_sha256=digest(v1/'software/review.json'),
        reused_a016_supervision_sha256=digest(Path('outputs/gated/A016/supervision.py')),
        shared_ledger_unchanged=True,torch_imported=False,actual_gpu_cases=0,
        limitation='Allocation correction only. Unchanged process semantics inherit reviewed v1/A016 checks; current live run remains v1.'))
    print(json.dumps(dict(status='passed',checks=len(checks))))
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C005',revision=2,status='technical_blocked',checks=checks,error=repr(exc),
        source_sha256={f:digest(base/f) for f in files},shared_ledger_unchanged=digest(shared)==before))
    raise
