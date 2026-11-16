"""Focused CPU checks for P26 paths, inherited science and unlimited accounting."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from common import BASE, ORIGINAL, UNLIMITED, HELPERS, BALLS_CHANGES, admit, arguments, core_reviews, parents, progress_paths, reconcile
from go2nav.research import atomic_json, digest, require_record

def refuses(fn):
    try: fn()
    except (ValueError, FileNotFoundError, KeyError): return
    raise AssertionError('Expected refusal')

def fixture_review(path, result, status='passed', identity=None):
    atomic_json(path.parent/'result.json', result)
    value=dict(research_id=result['research_id'],result='result.json',result_sha256=digest(path.parent/'result.json'),reviewed_at='synthetic fixture only',
        sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='not actual approval')],measurements='synthetic',uncertainty='synthetic',explanations=['synthetic'],next_step='fixture only',status=status)
    if identity is not None: value['run_identity']=identity
    atomic_json(path,value)

began=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);checks=[]
for name in HELPERS:
    (out/'source').mkdir(exist_ok=True);shutil.copy2(BASE/name,out/'source'/name)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
live_sources={str(p):digest(p) for p in UNLIMITED.glob('*.py')}
try:
    for name in HELPERS: ast.parse((BASE/name).read_text())
    stages,supervision,unlimited,changed=parents();_,rh=require_record('U003');original=json.loads((ORIGINAL/'software/result.json').read_text())
    for name in ['paths','arguments','inventory','stage_parents','consultation','generated']:
        assert Path(getattr(stages,name).__code__.co_filename).resolve()==(ORIGINAL/'stages.py').resolve()
    for name in ['accounting_run','accounting_remaining','wait_progress']:
        fn=getattr(unlimited,name);fn=getattr(fn,'__wrapped__',fn)
        assert Path(fn.__code__.co_filename).resolve()==(UNLIMITED/'common.py').resolve()
    assert Path(supervision.close_intervals.__code__.co_filename).resolve()==(ORIGINAL/'supervision.py').resolve()
    checks.append('exact_inherited_stage_accounting_progress_and_closure_functions')
    for stage in ['export','s1','compare']:
        p=stages.paths(Path.cwd(),stage);old=stages.arguments('/fixture/python',p);new=arguments('/fixture/python',p,stages)
        assert new[2:]==old[2:] and new[:2]==['/fixture/python',str(BASE/'science.py')]
        assert p['operations']==Path.cwd()/'outputs/gated/D003/runs'/stage
        assert progress_paths(p,p['operations'])[0]==p['operations']/'stage.log'
    prepared=stages.inventory(stages.paths(Path.cwd(),'export'));assert len(prepared)==2735
    refuses(lambda:core_reviews(Path.cwd()))
    checks.append('three_original_commands_2735_prepared_files_and_missing_actual_core_refusal')
    with tempfile.TemporaryDirectory(prefix='u003_fixture_') as tmp:
        root=Path(tmp);source=root/'source';old={};admitted={}
        for name in sorted(BALLS_CHANGES|{'other.py'}):
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('old');old[name]=digest(p)
            if name in BALLS_CHANGES:p.write_text('admitted');admitted[name]=digest(p)
        assert set(reconcile(source,old,admitted))==BALLS_CHANGES
        (source/'other.py').write_text('unapproved');refuses(lambda:reconcile(source,old,admitted));(source/'other.py').write_text('old')
        bad=dict(admitted);bad[next(iter(BALLS_CHANGES))]='wrong';refuses(lambda:reconcile(source,old,bad))
        checks.append('exact_three_C014_source_reconciliations_and_unapproved_change_refusals')
        lp=root/'ledger.json';now=time.time();past=dict(id='past',research_id='D001',out='old',start=now-200000,end=now-1)
        atomic_json(lp,[past]);outer=admit(lp,root/'operations');assert outer['budget_policy']=='U001' and outer['research_id']=='D003'
        previous=digest(lp);refuses(lambda:admit(lp,root/'duplicate'));assert previous==digest(lp)
        assert supervision.close_intervals(lp,outer['id'],None,root/'science',root)==[outer['id']]
        assert json.loads(lp.read_text())[0]==past
        checks.append('admission_beyond_old_eight_and_fortyeight_hours_with_original_owned_closure')
        core=root/'core'
        for kind in ['l0','l1','o0','o1']:
            for phase in ['planning','probes']:
                folder='planning_full' if phase=='planning' and kind in ['o0','o1'] else phase
                dest=core/'outputs/gated'/folder/f'{kind}_n3_s0';dest.mkdir(parents=True)
                raw=dest/('cases.jsonl' if phase=='planning' else 'cases.npz');raw.write_bytes(b'synthetic fixture, not science')
                result=dict(research_id='B001' if phase=='planning' else 'C001',n=3,kind=kind,seed=0,cases_sha256=digest(raw))
                fixture_review(dest/'review.json',result,status='measured_negative',identity=dict(n=3,kind=kind,seed=0))
        fixture_review(core/'outputs/gated/binding/slots_n3_s0/review.json',dict(research_id='C002',n=3,kind='slots',seed=0),'measured_negative')
        for phase in ['D001/adapter_check','D002/legacy']:fixture_review(core/'outputs/gated'/phase/'review.json',dict(research_id='D001'))
        for kind in ['o0','o1']:
            science=core/'outputs/gated/planning_full'/f'{kind}_n3_s0'
            science_hash=digest(science/'result.json');operation=core/'outputs/gated/U001/operations'/f'{kind}_n3_s0'
            op=dict(research_id='C009',budget_policy='U001',name=f'{kind}_n3_s0',returncode=0,child_terminal=True,inputs_unchanged=True,timed_out=False,actions=[],errors=[],child_artifact_sha256={'result.json':science_hash})
            fixture_review(operation/'review.json',op)
            checked=dict(research_id='U002',checks={'cases':3000},all_task_scoring_summary_checks_passed=True,result_sha256=science_hash,operational_result_sha256=digest(operation/'result.json'))
            # Keep the science result separate from the inspection's result path.
            atomic_json(science/'inspection.json',checked)
            fixture=core/'inspection_fixture'/kind/'review.json';fixture_review(fixture,checked)
            review=json.loads(fixture.read_text());review['result']='inspection.json';atomic_json(science/'inspection_review.json',review)
        assert len(core_reviews(core))==15
        inspection=core/'outputs/gated/planning_full/o0_n3_s0/inspection.json';review_path=inspection.with_name('inspection_review.json')
        saved=inspection.read_bytes();saved_review=review_path.read_bytes();value=json.loads(saved);value['checks']['cases']=55;atomic_json(inspection,value)
        review=json.loads(saved_review);review['result_sha256']=digest(inspection);atomic_json(review_path,review);refuses(lambda:core_reviews(core))
        inspection.write_bytes(saved);review_path.write_bytes(saved_review)
        full=inspection.parent;old_path=core/'outputs/gated/planning/o0_n3_s0';full.rename(old_path);refuses(lambda:core_reviews(core));old_path.rename(full)
        raw=full/'cases.jsonl';saved=raw.read_bytes();raw.write_bytes(b'changed');refuses(lambda:core_reviews(core));raw.write_bytes(saved)
        assert len(core_reviews(core))==15
        checks.append('fifteen_complete_core_reviews_negative_science_allowed_partial_old_path_tamper_refused')
    assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(p)==h for p,h in live_sources.items())
    checks.append('no_Torch_shared_ledger_or_active_U001_source_mutation')
    result=dict(research_id='U003',research_sha256=rh,d001_record_sha256=require_record('D001')[1],status='awaiting_literature_review',checks=checks,
        elapsed_seconds=time.monotonic()-began,source_sha256={f:digest(BASE/f) for f in HELPERS},
        repository_source_sha256={str(p):digest(p) for p in Path('go2nav').rglob('*.py')},config_sha256=original['config_sha256'],frozen_model_sha256=original['frozen_model_sha256'],
        a016_supervision_sha256=original['a016_supervision_sha256'],source_reconciliation=changed,original_D003_review_sha256=digest(ORIGINAL/'software/review.json'),
        U001_review_sha256=digest(UNLIMITED/'software/review.json'),prepared_input_files=len(prepared),actual_GPU_cases=0,
        limitation='CPU integration checks and inventory only. Actual complete core is still missing. No track export, S1 training, P26 comparison or full feature-byte hashing occurred.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','config_sha256','frozen_model_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='U003',status='technical_blocked',checks=checks,error=repr(exc)));raise
