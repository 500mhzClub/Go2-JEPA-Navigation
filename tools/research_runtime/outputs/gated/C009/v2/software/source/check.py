"""Focused C009 CPU checks; synthetic reviews never approve real science."""
import ast
import copy
from contextlib import contextmanager
import importlib.util
import shutil
import torch
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
from go2nav.eval.slots_inference import IMPLEMENTATION_SOURCES
from supervision import admit,close_intervals,parent,remaining,stop_owned,wait_owned
from stages import arguments,consultation,identity,inputs,route

BASE=Path('outputs/gated/C009/v2')


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,FileNotFoundError,KeyError):return
    raise AssertionError('Expected refusal')


def fake_review(folder,raw):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',result_sha256=digest(folder/'result.json'),status='passed',reviewed_at='synthetic fixture only',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],measurements='synthetic only',uncertainty='not actual science',explanations=['fixture'],next_step='fixture only'))



def renew_review(folder, filename='review.json'):
    p=folder/filename;v=json.loads(p.read_text());v['result_sha256']=digest(folder/v['result']);atomic_json(p,v)


@contextmanager
def saved_tree(root):
    saved={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    try:yield
    finally:
        for p,data in saved.items():p.write_bytes(data)


def change(path,key,value):
    d=json.loads(path.read_text());d[key]=value;atomic_json(path,d)


def fixture(root,rh):
    num=root/'outputs/gated/C014';helper=num/'fixture.py';helper.parent.mkdir(parents=True,exist_ok=True);helper.write_text('fixture helper')
    code=root/'go2nav/fixture.py';code.parent.mkdir(exist_ok=True);code.write_text('fixture repository source')
    helpers={'fixture.py':digest(helper)}
    fake_review(num/'software_numerics',dict(research_id='C014',research_sha256=rh,helper_sha256=helpers))
    for split in ['train','test']:
        for f in ['meta.json','frames.npy','labels.npz']:
            p=root/'outputs/data/balls/n3'/split/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic dataset')
    baseline=root/'outputs/gated/probes/l0_n3_s0';pre=root/'outputs/gated/C006/pre_l0_n3_s0.json'
    original={str(f.relative_to(root)):digest(f) for f in (root/'outputs/data/balls/n3').rglob('*') if f.is_file()}
    atomic_json(pre,dict(data_sha256=original));atomic_json(baseline/'manifest.json',dict(inputs={str(pre):digest(pre)}))
    fake_review(baseline,dict(research_id='C001',n=3,kind='l0',seed=0,manifest_sha256=digest(baseline/'manifest.json')))
    controls=root/'outputs/gated/B001/n3';atomic_json(controls/'manifest.json',dict(fixture=True));(controls/'cases.jsonl').write_text('synthetic control')
    fake_review(controls,dict(research_id='B001',n=3,controls_passed=True,cases_sha256=digest(controls/'cases.jsonl')))
    def selected(kind,parents):
        name=f'{kind}_n3_s0';p=route(root,'binding' if kind=='slots' else 'probes',name);out=p['review'].parent;op=root/'outputs/gated/C014/operations'/name
        p['checkpoint'].parent.mkdir(parents=True,exist_ok=True);torch.save(dict(kind=kind,n=3,seed=0),p['checkpoint'])
        atomic_json(p['checkpoint'].with_name('config.json'),dict(kind=kind,n=3,seed=0,parents=parents))
        bound={str(p['checkpoint']):digest(p['checkpoint'])}
        atomic_json(out/'manifest.json',dict(inputs=bound,computational_source_sha256={'go2nav/fixture.py':digest(code)},helper_sha256=helpers))
        raw=dict(research_id='C014',research_sha256=rh,name=name,kind=kind,n=3,seed=0,status='awaiting_literature_review',selected_policy='encoder64_predictor64',checkpoint_sha256=digest(p['checkpoint']),complete=True,checks_passed=True,weights_unchanged=True,inputs_unchanged=True,comparison_count={'slots':1104,'o0':2772,'o1':4164}[kind],implementation_sha256={f:digest(f) for f in IMPLEMENTATION_SOURCES},inputs_sha256=bound)
        atomic_json(out/'result.json',raw)
        atomic_json(op/'manifest.json',dict(software_review_sha256=digest(num/'software_numerics/review.json'),helper_sha256=helpers))
        fake_review(op,dict(research_id='C014',name=name,child_terminal=True,returncode=0,timed_out=False,actions=[],errors=[],manifest_sha256=digest(op/'manifest.json'),child_artifact_sha256={f:digest(out/f) for f in ['result.json','manifest.json']}))
        iv=dict(research_id='C014',numerical_passed=True,result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'))
        atomic_json(out/'inspection.json',iv)
        fake_review(out,raw);ir=json.loads(p['review'].read_text());ir.update(result='inspection.json',result_sha256=digest(out/'inspection.json'));atomic_json(out/'inspection_review.json',ir)
        review=json.loads(p['review'].read_text());review['inspection_review_sha256']=digest(out/'inspection_review.json');atomic_json(p['review'],review)
        return p
    encoder=selected('slots',{})
    cache=root/'outputs/gated/C001/extract_n3_s0';cache.mkdir(parents=True)
    for split in ['train','val']:(cache/f'{split}.npy').write_bytes(b'synthetic cache')
    atomic_json(cache/'index.json',dict(n=3,seed=0,encoder=str(encoder['checkpoint']),encoder_sha256=digest(encoder['checkpoint']),splits={s:digest(cache/f'{s}.npy') for s in ['train','val']}))
    world=selected('o1',{str(cache/'index.json'):digest(cache/'index.json')})
    return encoder,world


def fixture_checks(root,checks,rh):
    encoder,world=fixture(root,rh)
    for p in [encoder,world,route(root,'planning',world['name'])]:
        files=inputs(p);assert p['checkpoint'] in files and p['review'].parent/'inspection_review.json' in files
    checks['unmocked_encoder_predictor_planning_synthetic_lineage']=True
    p=world;out=p['review'].parent;op=root/'outputs/gated/C014/operations'/p['name']
    for key,value in [('research_id','C004'),('selected_policy','encoder32_predictor64'),('complete',False),('checks_passed',False),('kind','o0'),('seed',1),('comparison_count',2772),('checkpoint_sha256','changed')]:
        with saved_tree(root):
            change(out/'result.json',key,value);renew_review(out);rejects(lambda:inputs(p))
    checks['eight_numerical_policy_identity_coverage_refusals']=True
    for key,value in [('research_id','C007'),('name','o0_n3_s0'),('child_terminal',False),('returncode',1),('timed_out',True),('actions',['signal']),('errors',['error'])]:
        with saved_tree(root):
            change(op/'result.json',key,value);renew_review(op);rejects(lambda:inputs(p))
    for folder in [op,out,root/'outputs/gated/C014/software_numerics']:
        with saved_tree(root):change(folder/'review.json','status','technical_blocked');rejects(lambda:inputs(p))
    checks['seven_operation_and_three_review_refusals']=True
    for key,value in [('numerical_passed',False),('result_sha256','stale'),('operational_result_sha256','stale')]:
        with saved_tree(root):
            change(out/'inspection.json',key,value);renew_review(out,'inspection_review.json');change(p['review'],'inspection_review_sha256',digest(out/'inspection_review.json'));rejects(lambda:inputs(p))
    for key,value in [('status','technical_blocked'),('research_id','C004')]:
        with saved_tree(root):
            change(out/'inspection_review.json',key,value);change(p['review'],'inspection_review_sha256',digest(out/'inspection_review.json'));rejects(lambda:inputs(p))
    with saved_tree(root):change(p['review'],'inspection_review_sha256','stale');rejects(lambda:inputs(p))
    checks['six_inspection_and_science_binding_refusals']=True
    for changed in [root/'go2nav/fixture.py',root/'outputs/gated/C014/fixture.py',root/'outputs/gated/C001/extract_n3_s0/train.npy',encoder['checkpoint']]:
        with saved_tree(root):changed.write_bytes(b'changed');rejects(lambda:inputs(p))
    with saved_tree(root):change(p['checkpoint'].with_name('config.json'),'n',1);rejects(lambda:inputs(p))
    checks['repository_helper_cache_encoder_and_training_identity_refusals']=True
    files=inputs(p);pre=root/'consultation.json'
    value=dict(stage=p['stage'],name=p['name'],research_sha256=rh,scientific_research_sha256='science',inputs_sha256={str(f.resolve()):digest(f) for f in files},consulted_at='synthetic',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],consequences='fixture only')
    atomic_json(pre,value);consultation(pre,p,files,rh,'science')
    for key in ['stage','name','research_sha256','scientific_research_sha256','inputs_sha256','sources']:
        wrong=copy.deepcopy(value);wrong[key]=None;atomic_json(pre,wrong);rejects(lambda:consultation(pre,p,files,rh,'science'))
    atomic_json(pre,value)
    with saved_tree(root):
        frames=root/'outputs/data/balls/n3/test/frames.npy';frames.write_bytes(b'changed original data')
        new=copy.deepcopy(value);new['inputs_sha256'][str(frames.resolve())]=digest(frames);atomic_json(pre,new)
        consultation(pre,p,files,rh,'science');rejects(lambda:inputs(p))
    controls=root/'outputs/gated/B001/n3'
    with saved_tree(root):
        change(controls/'result.json','controls_passed',False);renew_review(controls);rejects(lambda:inputs(route(root,'planning',p['name'])))
    with saved_tree(root):
        (controls/'cases.jsonl').write_bytes(b'changed');rejects(lambda:inputs(route(root,'planning',p['name'])))
    checks['consultation_schema_original_data_despite_fresh_hash_and_control_refusals']=True


def actual_checks(checks):
    root=Path.cwd();original=Path('outputs/gated/C009');pre=json.loads((BASE/'pre_implementation.json').read_text())
    verify_files(original,pre['inherited_c009_source_sha256']);assert digest(BASE/'supervision.py')==digest(original/'supervision.py')
    spec=importlib.util.spec_from_file_location('original_c009_routes',original/'stages.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    count=0
    for stage in ['planning','probes','binding']:
        for n in [1,3,6]:
            for seed in ([0,1,2] if n==3 else [0]):
                for kind in (['slots'] if stage=='binding' else ['o0','o1'] if n==3 else ['o1']):
                    name=f'{kind}_n{n}_s{seed}';p=route(root,stage,name);a=arguments(sys.executable,p);before=old.arguments(sys.executable,old.route(root,stage,name))
                    flag='--inference-review' if stage=='planning' else '--parent-review';before[before.index(flag)+1]=str(p['review']);assert a==before;count+=1
    assert count==21;checks['all21_scientific_commands_unchanged_except_C014_review_path']=True
    actual={}
    for stage,name in [('binding','slots_n3_s0'),('probes','o0_n3_s0'),('planning','o0_n3_s0'),('probes','o1_n3_s0'),('planning','o1_n3_s0')]:
        p=route(root,stage,name);files=inputs(p);actual[f'{stage}/{name}']=len(files)
    rejects(lambda:inputs(route(root,'probes','o1_n3_s1')))
    checks['five_real_reviewed_parent_routes_and_missing_future_seed_refusal']=actual
    assert not torch.cuda.is_initialized();checks['actual_admission_no_CUDA']=True



started=time.monotonic();_,rh=require_record('C014');_,legacy_rh=require_record('C009');out=BASE/'software';out.mkdir(exist_ok=False);checks={}
for name in ['supervision.py','stages.py','run_eval.py','check.py']:
    (out/'source').mkdir(exist_ok=True);shutil.copy2(BASE/name,out/'source'/name)
torch.set_num_threads(2)
ledger=Path('outputs/gated/gpu_intervals.json');ledger_before=digest(ledger)
unchanged={str(f):digest(f) for b in ['B002','B003','C005','C007','C008','D003','C014'] for f in Path(f'outputs/gated/{b}').rglob('*.py')}
try:
    for name in ['supervision.py','stages.py','run_eval.py','check.py']:ast.parse((BASE/name).read_text())
    rows=[dict(research_id=r,start=a,end=b) for r,a,b in [('B001',10,30),('B001',15,25),('C002',40,50),('C004',60,70)]]
    got=remaining(rows,100,'B001');assert got==dict(global_seconds=48*3600-40,bc_seconds=28*3600-30,science_seconds=7200-20)
    inherited=parent.remaining_seconds(rows,100);assert inherited['bc_seconds']==got['bc_seconds']
    checks['underlying_science_ids_nested_union_and_inherited_allocation']=True
    with tempfile.TemporaryDirectory(prefix='c009_fixture_') as tmp:
        root=Path(tmp);lp=root/'ledger.json'
        for science,cap in [('B001',7200),('C001',28*3600),('C002',3600)]:
            atomic_json(lp,[]);outer,left,work=admit(lp,root/'op',science);assert work==cap-45
            assert outer['research_id']==science and outer['supervisor_research_id']=='C009'
            old=digest(lp);rejects(lambda:admit(lp,root/'again',science));assert old==digest(lp)
            atomic_json(lp,[dict(id='used',research_id=science,start=0,end=cap,out='old')]);old=digest(lp)
            rejects(lambda:admit(lp,root/'exhausted',science));assert old==digest(lp)
        checks['three_existing_caps_and_open_exhausted_no_mutation_refusals']=True
        now=time.time();child=root/'science';child.mkdir();atomic_json(child/'manifest.json',dict(run_id='inner'))
        rows=[dict(id='outer',research_id='B001',supervisor_research_id='C009',start=now-2,end=None,out=str(root/'ops')),
              dict(id='inner',research_id='B001',start=now-1,end=None,out=str(child)),
              dict(id='foreign',research_id='C001',start=now-1,end=None,out='foreign')]
        atomic_json(lp,rows);closed=close_intervals(lp,'outer',None,child,root,'B001');assert set(closed)=={'outer','inner'}
        assert json.loads(lp.read_text())[2]==rows[2]
        class Live:
            def poll(self):return None
        atomic_json(lp,rows);old=digest(lp);rejects(lambda:close_intervals(lp,'outer',Live(),child,root,'B001'));assert old==digest(lp)
        bad=[]
        v=copy.deepcopy(rows);v[0]['supervisor_research_id']='foreign';bad.append(v)
        v=copy.deepcopy(rows);v[0]['research_id']='C001';bad.append(v)
        v=copy.deepcopy(rows);v[1]['start']=now-3;bad.append(v)
        v=copy.deepcopy(rows);v.append(dict(v[1],id='duplicate'));bad.append(v)
        v=copy.deepcopy(rows);v[1]['id']='wrong';bad.append(v)
        for v in bad:
            atomic_json(lp,v);old=digest(lp);rejects(lambda:close_intervals(lp,'outer',None,child,root,'B001'));assert old==digest(lp)
        checks['canonical_owned_closure_foreign_preservation_and_six_refusals']=True
        routes=[]
        for stage in ['planning','probes','binding']:
            for n in [1,3,6]:
                for seed in ([0,1,2] if n==3 else [0]):
                    for kind in (['slots'] if stage=='binding' else ['o0','o1'] if n==3 else ['o1']):
                        p=route(root,stage,f'{kind}_n{n}_s{seed}');a=arguments('/fixture/python',p)
                        assert a[:3]==['/fixture/python','-m','go2nav.eval.balls_'+stage]
                        assert a[a.index('--research')+1]=={'planning':'B001','probes':'C001','binding':'C002'}[stage]
                        assert ('--controls' in a)==(stage=='planning');routes.append(p)
        assert len(routes)==21
        for stage,name in [('train','o1_n3_s0'),('planning','l0_n3_s0'),('probes','o0_n1_s0'),('binding','slots_n1_s1'),('binding','o1_n3_s0')]:rejects(lambda:route(root,stage,name))
        checks['twenty_one_fixed_routes_exact_modules_and_unplanned_refusals']=True
        fixture_checks(root,checks,rh)
        for mode,duration,deadline in [('normal',.01,2),('deadline',10,.1)]:
            argv=[sys.executable,'-c',f'import time;time.sleep({duration})'];actions=[]
            with (root/f'{mode}.log').open('wb') as log:
                process=subprocess.Popen(argv,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                handle=psutil.Process(process.pid);created=handle.create_time()
                try:
                    timed=wait_owned(process,handle,created,argv,Path.cwd(),time.monotonic()+deadline,actions,poll_seconds=.02,reserve=0,term_seconds=.5,kill_seconds=.5)
                    assert (process.returncode,timed)==((0,False) if mode=='normal' else (-15,True))
                    checks['owned_CPU_'+mode]=dict(returncode=process.returncode,timed_out=timed,actions=actions)
                finally:
                    if process.poll() is None:stop_owned(process,handle,created,argv,Path.cwd(),actions,.5,.5)
    assert not torch.cuda.is_initialized() and digest(ledger)==ledger_before and all(digest(f)==h for f,h in unchanged.items())
    checks['no_CUDA_no_shared_ledger_or_existing_helper_mutation']=True
    actual_checks(checks)
    result=dict(research_id='C014',research_sha256=rh,c009_record_sha256=legacy_rh,status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-started,
        source_sha256={f:digest(BASE/f) for f in ['supervision.py','stages.py','run_eval.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{rid}.json':digest(f'docs/research/records/{rid}.json') for rid in ['B001','C001','C002','C005','C004','C009','C014']},
        actual_GPU_cases=0,cuda_initialized=torch.cuda.is_initialized(),pre_verification_sha256=digest(BASE/'pre_verification.json'),limitation='Real selected encoder/O0/O1 admission, disposable evidence fixtures and inherited CPU owned-process tests. No actual GPU scientific evaluation or GPU stall was tested.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C014',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
