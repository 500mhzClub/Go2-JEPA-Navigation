"""B004 software fixtures; no actual additional-seed experiment or automatic review."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import planning as p
import parents
from run_one import cpu_lock,supervise,owned_module
import verify_cases as inspector
from go2nav.eval.balls_planning import _evaluate
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
import numpy as np
import torch
import psutil

BASE=Path('outputs/gated/B004');started=time.monotonic();record,rh=require_record('B004')
out=BASE/'software';out.mkdir(exist_ok=False);checks={}
ledger=Path('outputs/gated/gpu_intervals.json');ledger_before=digest(ledger)
oldhelpers={str(f):digest(f) for n in ['B002','B003','C005','C006','C007','C008','C009','C010','C011','A016','A018','A019','D003'] for f in Path('outputs/gated',n).rglob('*.py')}

def rejects(call):
    try:call()
    except (ValueError,RuntimeError,KeyError,FileNotFoundError,AssertionError,BlockingIOError):return
    raise AssertionError('Expected invalid fixture refusal')


def fake_review(folder,raw,status='passed',**extra):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',result_sha256=digest(folder/'result.json'),
        reviewed_at='synthetic fixture only',status=status,sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],
        measurements='fixture only',uncertainty='not actual science',explanations=['fixture'],next_step='fixture only',**extra))


def probe_fixture(root,name):
    sf=root/'outputs/gated/probes'/name;sf.mkdir(parents=True)
    (sf/'cases.npz').write_bytes(b'synthetic complete probe cases')
    ck=root/'outputs/gated/A019/inference'/name/'calibrated.pt'
    atomic_json(sf/'manifest.json',dict(research_sha256=require_record('C011')[1],inputs={str(ck):digest(ck)}))
    fake_review(sf,dict(research_id='C011',kind=name[:2],n=3,seed=int(name[-1]),train_windows=4096,test_windows=5000,test_episodes=1000,
        checkpoint_sha256=digest(ck),manifest_sha256=digest(sf/'manifest.json'),cases_sha256=digest(sf/'cases.npz')),'measured_negative')
    op=root/'outputs/gated/C011/supervision'/name
    fake_review(op,dict(research_id='C011',name=name,returncode=0,child_terminal=True,signals=[],errors=[],stop_reason=None,
        inputs_unchanged=True,no_C011_gpu_intervals=True,child_artifact_sha256={f.name:digest(f) for f in sf.iterdir() if f.name!='review.json'}))


def planning_fixture(root,name):
    sf=root/'outputs/gated/planning'/name;sf.mkdir(parents=True);(sf/'cases.jsonl').write_bytes(b'synthetic complete planning cases')
    raw=dict(research_id='B004',kind=name[:2],n=3,seed=int(name[-1]),complete_cases=3000)
    atomic_json(sf/'result.json',raw)
    op=root/'outputs/gated/B004/operations'/name
    fake_review(op,dict(research_id='B004',name=name,returncode=0,child_terminal=True,signals=[],errors=[],stop_reason=None,
        inputs_unchanged=True,no_B004_gpu_intervals=True,elapsed_seconds=20,child_artifact_sha256={f.name:digest(f) for f in sf.iterdir()}))
    ins=root/'outputs/gated/B004/inspection'/name
    fake_review(ins,dict(research_id='B004',name=name,checks=dict(cases=3000),all_task_scoring_summary_checks_passed=True,
        saved_singleton_numerics_passed=True,scientific_result_sha256=digest(sf/'result.json'),operational_result_sha256=digest(op/'result.json')))
    fake_review(sf,raw,'measured_negative',inspection_review=str(ins/'review.json'),inspection_review_sha256=digest(ins/'review.json'))

try:
    torch.set_num_threads(1);p.guard()
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('B004','B003')
            return node
    def parse(path):return ast.parse(Path(path).read_text())
    def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=parse('outputs/gated/B003/planning.py');new=parse(BASE/'planning.py')
    for name in ['numerical_check','cases']:
        assert ast.dump(fn(old,name))==ast.dump(Normalize().visit(copy.deepcopy(fn(new,name))))
    oldbody=next(n for n in ast.walk(fn(old,'run')) if isinstance(n,ast.With))
    newbody=next(n for n in ast.walk(fn(new,'run')) if isinstance(n,ast.With))
    # Only stronger final input hashing changes the computational body.
    class InputCheck(Normalize):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='verify_files' and len(node.args)==2 and isinstance(node.args[1],ast.Subscript):
                if ast.unparse(node.args[1])=="ctx['consult']['inputs_sha256']":node.args[1]=ast.Name(id='data_hashes',ctx=ast.Load())
            return node
    assert ast.dump(oldbody)==ast.dump(InputCheck().visit(copy.deepcopy(newbody)))
    original_inspector=parse('outputs/gated/B003/inspection/v2/verify_cases.py');newins=parse(BASE/'verify_cases.py')
    for name in ['audit_rows','summary_from_rows','compare_summary']:
        assert ast.dump(fn(original_inspector,name))==ast.dump(Normalize().visit(copy.deepcopy(fn(newins,name))))
    checks['unchanged_B003_candidate_case_computation_and_independent_scoring']=True
    b003=json.loads(Path('outputs/gated/B003/software/result.json').read_text());verify_files(Path('outputs/gated/B003'),b003['source_sha256']);verify_files(Path.cwd(),b003['repository_source_sha256'])
    assert require_review(Path('outputs/gated/B003/inspection/v2/software/review.json'))['status']=='passed'
    assert digest('outputs/gated/B003/inspection/v2/verify_cases.py')==json.loads(Path('outputs/gated/B003/inspection/v2/software/result.json').read_text())['source_sha256']
    checks['reviewed_source_and_original_inspector_bindings']=True

    class Fake(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
        def encode(self,x):return x.mean((-3,-2,-1))[...,None].expand(-1,-1,2)
        def rollout(self,z,a,steps):return z[:,-1:]+a[:,-steps:].cumsum(1)
    frames=np.broadcast_to(np.arange(100,dtype=np.uint8)[None,:,None,None,None],(2,100,4,4,3)).copy()
    labels=dict(actions=np.full((2,100,2),.2,np.float32),red=np.broadcast_to([20.,20.],(2,100,2)).copy(),white=np.broadcast_to([40.,40.],(2,100,3,2)).copy())
    with tempfile.TemporaryDirectory(prefix='b004_fixtures_') as td:
        root=Path(td);original=root/'original';original.mkdir();atomic_json(root/'ledger.json',[])
        _evaluate(frames,labels,3,original,'B004',Fake(),dict(budget_hours=48,gpu_ledger=str(root/'ledger.json')),'l0')
        expected=[json.loads(x) for x in (original/'cases.jsonl').read_text().splitlines()];actual=list(p.cases(Fake(),frames,labels,3,'l0'))
        assert len(actual)==len(expected)==6
        for a,b in zip(actual,expected):assert {k:v for k,v in a.items() if k!='latency_s'}=={k:v for k,v in b.items() if k!='latency_s'}
        controls={(r['episode'],r['horizon']):{k:v for k,v in r.items() if k not in ['goal_frame','start_position','goal_position']} for r in expected}
        inspected=inspector.audit_rows(actual,controls,labels,3,'l0',2);assert inspected['cases']==6
        independent=inspector.summary_from_rows(actual,'l0');inspector.compare_summary(p.summarize(actual)['arms'],independent)
        for key,value in [('success',not actual[0]['success']),('endpoint_error',99.),('collision',not actual[0]['collision']),('goal_frame',99),('seed',[0])]:
            wrong=copy.deepcopy(actual);wrong[0][key]=value;rejects(lambda:inspector.audit_rows(wrong,controls,labels,3,'l0',2))
        wrong=copy.deepcopy(independent);wrong['l0']['4']['success']=.123;rejects(lambda:inspector.compare_summary(wrong,independent))
        checks['original_evaluator_six_cases_independent_scores_and_six_corruptions']=True
        inp=p.task_inputs(frames,labels['actions'],0,p.task_start(3,0,8),8)
        metrics,arrays=p.numerical_check(Fake(),inp,3);_,priorarrays=p.prior.candidate_check(Fake(),inp)
        assert all(v<1e-12 for v in metrics.values())
        for key in arrays:np.testing.assert_array_equal(arrays[key],priorarrays[key])
        checks['fixed_candidate_numerics_match_original_B002']=True

        # Reuse only declared synthetic fixture definitions, never execute C011's check module.
        fixturetree=parse('outputs/gated/C011/check.py')
        selected=[fn(fixturetree,n) for n in ['training_fixture','numerical_fixture','data_fixture']]
        ns=dict(Path=Path,atomic_json=atomic_json,digest=digest,fake_review=fake_review)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'reviewed_c011_fixture_definitions','exec'),ns)
        ns['data_fixture'](root)
        for name in parents.ORDER:
            ns['numerical_fixture'](root,name);probe_fixture(root,name)
        parentmod=parents.inherited()
        for name in parents.ORDER:
            ctx=parentmod.numerical_parent(root,name);files=parents.probe_parent(root,name,ctx);assert len(files)>=5
        checks['four_complete_numerical_and_measured_negative_probe_parent_chains']=True
        name=parents.ORDER[0];ctx=parentmod.numerical_parent(root,name);sf=root/'outputs/gated/probes'/name;op=root/'outputs/gated/C011/supervision'/name
        saved={f:f.read_bytes() for f in root.rglob('*') if f.is_file()}
        def restore():
            for f,value in saved.items():f.write_bytes(value)
        def change(path,keys,value):
            doc=json.loads(path.read_text());target=doc
            for key in keys[:-1]:target=target[key]
            target[keys[-1]]=value;atomic_json(path,doc)
            if path.name=='result.json':
                rv=path.with_name('review.json');v=json.loads(rv.read_text());v['result_sha256']=digest(path);atomic_json(rv,v)
        bad=[(sf/'review.json',['status'],'technical_blocked'),(sf/'result.json',['seed'],2),(sf/'result.json',['test_windows'],4999),
             (sf/'result.json',['checkpoint_sha256'],'wrong'),(sf/'result.json',['manifest_sha256'],'wrong'),
             (op/'result.json',['returncode'],1),(op/'result.json',['child_terminal'],False),(op/'result.json',['inputs_unchanged'],False),
             (op/'result.json',['no_C011_gpu_intervals'],False)]
        for path,keys,value in bad:change(path,keys,value);rejects(lambda:parents.probe_parent(root,name,ctx));restore()
        (sf/'cases.npz').write_bytes(b'changed');rejects(lambda:parents.probe_parent(root,name,ctx));restore()
        checks['ten_partial_wrong_or_changed_probe_refusals']=True
        datafiles,data=parentmod.original_probe_data(root);changed=root/'outputs/data/balls/n3/test/frames.npy';changed.write_bytes(b'changed original data')
        rejects(lambda:parentmod.original_probe_data(root));restore()
        full=parents.inputs(root,name);assert full['data']==data
        checks['full_actual_admission_route_and_original_data_refusal_fixture']=True
        software=root/'outputs/gated/B004/software';software.mkdir(parents=True);(software/'review.json').write_bytes(b'fixture software hash')
        pre=dict(name=name,research_sha256=rh,b001_record_sha256=digest('docs/research/records/B001.json'),checkpoint_sha256=digest(full['checkpoint']),
            inputs_sha256={str(f):digest(f) for f in sorted(full['files'])},data_sha256=full['data'],software_review_sha256=digest(software/'review.json'),
            consulted_at='fixture',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],consequences='fixture only')
        pp=root/'pre.json';atomic_json(pp,pre);assert parents.consultation(pp,root,name,full,rh)==pre
        for key,value in [('inputs_sha256',{}),('checkpoint_sha256','wrong'),('name','l0_n3_s2'),('data_sha256',{}),('sources',[{}])]:
            wrong=copy.deepcopy(pre);wrong[key]=value;atomic_json(pp,wrong);rejects(lambda:parents.consultation(pp,root,name,full,rh))
        checks['complete_consultation_and_five_stale_bindings_refused']=True
        rejects(lambda:parents.previous_reviews(root,parents.ORDER[1]));planning_fixture(root,name)
        assert parents.previous_reviews(root,parents.ORDER[1])
        previous=root/'outputs/gated/planning'/name/'review.json';backup=previous.read_bytes();change(previous,['inspection_review_sha256'],'wrong')
        rejects(lambda:parents.previous_reviews(root,parents.ORDER[1]));previous.write_bytes(backup)
        operation=root/'outputs/gated/B004/operations'/name/'result.json';change(operation,['elapsed_seconds'],25000)
        # Also update inspection's operation digest to isolate the CPU allowance check.
        ins=root/'outputs/gated/B004/inspection'/name;change(ins/'result.json',['operational_result_sha256'],digest(operation))
        change(previous,['inspection_review_sha256'],digest(ins/'review.json'))
        rejects(lambda:parents.previous_reviews(root,parents.ORDER[1]))
        checks['previous_measured_negative_with_inspection_admitted_missing_stale_overbudget_refused']=True
        (root/'outputs/gated/B003').mkdir(parents=True,exist_ok=True)
        def acquire():
            with cpu_lock(root):pass
        with cpu_lock(root):rejects(acquire)
        acquire();checks['same_CPU_planning_probe_lock_refusal_and_release']=True
        normal=root/'normal';normal.mkdir();got=supervise([sys.executable,'-c','print("fixture")'],root,normal,dict(os.environ),dict(fixture=True),work=2,poll=.01)
        assert got['returncode']==0 and got['child_terminal'] and not got['errors'] and not got['signals']
        timed=root/'timed';timed.mkdir();got=supervise([sys.executable,'-c','import time;time.sleep(20)'],root,timed,dict(os.environ),dict(fixture=True),work=.04,term=.2,kill=.2,poll=.01)
        assert got['child_terminal'] and got['status']=='budget_incomplete' and got['signals'][0]['signal']=='SIGTERM' and not got['errors']
        argv=[sys.executable,'-c','import time;time.sleep(20)'];child=subprocess.Popen(argv,cwd=root);proc=psutil.Process(child.pid);ct=proc.create_time();actions=[];owned=owned_module()
        try:rejects(lambda:owned.stop_owned(child,proc,ct+1,argv,root,actions,.2,.2));assert not actions and child.poll() is None
        finally:owned.stop_owned(child,proc,ct,argv,root,actions,.2,.2)
        checks['actual_owned_normal_deadline_wrong_identity_CPU_processes']=True

    try:parents.inputs(Path.cwd(),parents.ORDER[0])
    except FileNotFoundError as exc:refusal=str(exc)
    else:raise AssertionError('Missing actual training was accepted')
    assert 'outputs/gated/C001/l0_n3_s1/review.json' in refusal
    checks['actual_missing_training_refused_before_model_or_output']=refusal
    assert all(digest(f)==h for f,h in oldhelpers.items()) and not torch.cuda.is_initialized()
    assert not any(x['research_id']=='B004' for x in json.loads(ledger.read_text()))
    checks['no_CUDA_or_existing_helper_change_or_B004_GPU_interval']=True
    result=dict(research_id='B004',research_sha256=rh,status='awaiting_literature_review',checks=checks,
        source_sha256={f.name:digest(f) for f in BASE.glob('*.py')},repository_source_sha256=b003['repository_source_sha256'],
        inherited_source_sha256=oldhelpers,record_sha256={str(Path('docs/research/records')/(n+'.json')):digest(Path('docs/research/records')/(n+'.json')) for n in ['B001','B003','B004','C011','A019']},
        elapsed_seconds=time.monotonic()-started,cuda_initialized=False,actual_scientific_planning_cases=0,
        ledger_before_sha256=ledger_before,ledger_after_sha256=digest(ledger),
        limitation='Synthetic complete parent/probe and six-case original-evaluator fixtures only. No additional-seed model inference, actual planning or scientific comparison has run; actual full-case independent inspection and separate reviews remain mandatory.')
    atomic_json(out/'result.json',result);print(json.dumps({k:result[k] for k in ['research_id','checks','elapsed_seconds','actual_scientific_planning_cases']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='B004',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)));raise
