"""Qualify only the new operation/path adapter; reuse reviewed physical scoring."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import psutil
import torch
from verify_planning import BASE, PARENT, RUNTIME, absent, expected_inputs, intervals, metadata, parents, paths
from go2nav.research import atomic_json, digest, require_record, require_review

def refuses(fn):
    try: fn()
    except (ValueError, KeyError, FileNotFoundError): return
    raise AssertionError('Expected refusal')

began = time.monotonic(); out = BASE/'software'; out.mkdir(exist_ok=False); checks = []
for name in ['verify_planning.py','check.py']:
    (out/'source').mkdir(exist_ok=True); shutil.copy2(BASE/name,out/'source'/name)
ledger_hash = digest('outputs/gated/gpu_intervals.json')
try:
    torch.set_num_threads(2)
    assert all(os.environ.get(k) == '' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    for name in ['verify_planning.py','check.py']: ast.parse((BASE/name).read_text())
    original, runtime = parents(); _, rh = require_record('U002'); _, ah = require_record('U001'); _, sh = require_record('B001')
    for name in ['audit_rows','summary_from_rows','compare_summary']:
        assert Path(getattr(original,name).__code__.co_filename).resolve() == (PARENT/'inspect_planning.py').resolve()
    checks.append('all_three_physical_summary_functions_imported_from_exact_reviewed_source')
    op=dict(research_id='C009',scientific_research_id='B001',stage='planning',name='o0_n3_s0',status='awaiting_literature_review',returncode=0,child_terminal=True,inputs_unchanged=True,timed_out=False,actions=[],errors=[],budget_policy='U001')
    m=dict(budget_policy='U001',authorization_sha256=ah,work_seconds=None,stage='planning',name='o0_n3_s0',scientific_research_id='B001',scientific_research_sha256=sh,source_commit='fixed',cwd=str(Path.cwd()),outer_interval_id='outer')
    sm=dict(budget_policy='U001',authorization_sha256=ah,budget_hours=None,research_sha256=sh,research_id='B001',code_commit='fixed',run_id='inner')
    metadata(original,op,m,sm,'o0_n3_s0',sh,ah)
    corruptions = [(0,'returncode',1),(0,'child_terminal',False),(0,'inputs_unchanged',False),(0,'timed_out',True),(0,'actions',['TERM']),(0,'errors',['failure']),(0,'status','technical_blocked'),(0,'name','o1_n3_s0'),(0,'budget_policy','old'),(1,'authorization_sha256','wrong'),(1,'work_seconds',7200),(1,'source_commit','other'),(1,'scientific_research_sha256','wrong'),(1,'stage','probes'),(2,'budget_hours',48),(2,'authorization_sha256','wrong'),(2,'research_id','C001')]
    for index,key,value in corruptions:
        values=copy.deepcopy([op,m,sm]);values[index][key]=value
        refuses(lambda:metadata(original,*values,'o0_n3_s0',sh,ah))
    checks.append('new_metadata_valid_and_seventeen_incomplete_wrong_identity_or_old_budget_refusals')
    science,operation=paths('o0_n3_s0');assert 'planning_full' in science.parts and 'U001' in operation.parts
    refuses(lambda:paths('l0_n3_s0'))
    rows=[dict(id='outer',research_id='B001',budget_policy='U001',start=10.,end=20.,out=str(operation),supervisor_research_id='C009'),dict(id='inner',research_id='B001',budget_policy='U001',start=11.,end=19.,out=str(science))]
    assert len(intervals(rows,m,sm,'o0_n3_s0'))==2
    for index,key,value in [(0,'end',None),(1,'end',None),(0,'out','wrong'),(1,'research_id','C001'),(1,'budget_policy','old'),(1,'start',9.),(0,'supervisor_research_id','wrong')]:
        bad=copy.deepcopy(rows);bad[index][key]=value;refuses(lambda:intervals(bad,m,sm,'o0_n3_s0'))
    refuses(lambda:intervals(rows+[rows[0]],m,sm,'o0_n3_s0'))
    checks.append('fresh_output_identity_closed_intervals_and_eight_corruption_refusals')
    from go2nav.eval.balls_planning import score,summarize,task_start
    labels={'red':np.full((2,100,2),32.,np.float32),'white':np.full((2,100,3,2),55.,np.float32)}; controls={}; cases=[]
    for e in range(2):
        for h in [4,6,8]:
            t=task_start(3,e,h);a=np.zeros((h,2),np.float32)
            row=dict(research_id='B001',n=3,episode=e,group=0,horizon=h,start_frame=t,goal_frame=t+h,start_position=[32.,32.],goal_position=[32.,32.],arm='o0',seed=[2026111504,3,e,h],actions=a.tolist(),latency_s=.1,**score(labels['red'][e,t],labels['red'][e,t+h],a,labels['white'][e,t+1:t+h+1]))
            cases.append(row);controls[e,h]=dict(row,arm='recorded')
    assert original.audit_rows(cases,controls,labels,3,'o0',2)['cases']==6
    original.compare_summary(summarize(cases)['arms'],original.summary_from_rows(cases,'o0'))
    refuses(lambda:original.audit_rows(cases[:-1],controls,labels,3,'o0',2))
    checks.append('unchanged_scoring_summary_integration_and_partial_case_refusal')
    live_m=json.loads(Path('outputs/gated/U001/operations/o0_n3_s0/manifest.json').read_text())
    expected=expected_inputs(live_m['cwd'],'o0_n3_s0')
    assert expected==live_m['preconsultation']['inputs_sha256'] and len(expected)==87
    proc=psutil.Process();refuses(lambda:absent(proc.pid,proc.create_time()))
    checks.append('complete_real_runtime_cwd_input_binding_and_live_process_refusal')
    assert not torch.cuda.is_initialized()
    # Another process is legitimately writing the real ledger only on terminal closure.
    assert digest('outputs/gated/gpu_intervals.json')==ledger_hash
    checks.append('no_CUDA_or_ledger_mutation')
    result=dict(research_id='U002',research_sha256=rh,status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-began,
        source_sha256={f:digest(BASE/f) for f in ['verify_planning.py','check.py']},original_inspector_sha256=digest(PARENT/'inspect_planning.py'),
        original_inspector_review_sha256=digest(PARENT/'planning_inspection_software/review.json'),runtime_software_review_sha256=digest(RUNTIME/'software/review.json'),
        canonical_input_files=len(expected),limitation='CPU adapter verification only; original thirteen-group physical-scoring qualification is inherited. Active O0 science is not inspected or interpreted.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='U002',status='technical_blocked',checks=checks,error=repr(exc)));raise
