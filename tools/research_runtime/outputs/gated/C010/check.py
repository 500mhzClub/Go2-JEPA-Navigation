"""CPU fixtures for C010 only; synthetic reviews never approve real results."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record
from supervision import admit,close_intervals,parent,remaining,stop_owned,wait_owned
from stages import ORDER,arguments,consultation,core_reviews,inputs,original_data,p26_reviews,prior_training,route

BASE=Path('outputs/gated/C010')


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,FileNotFoundError,KeyError):return
    raise AssertionError('Expected refusal')


def fake_review(folder,raw,status='passed',**extra):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',
        result_sha256=digest(folder/'result.json'),status=status,reviewed_at='synthetic fixture only',
        sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],
        measurements='synthetic only',uncertainty='not actual science',explanations=['fixture'],
        next_step='fixture only',**extra))


def core_fixture(root,seed):
    for kind in ['l0','l1','o0','o1']:
        for phase in ['planning','probes']:
            f=root/'outputs/gated'/phase/f'{kind}_n3_s{seed}';f.mkdir(parents=True)
            cases=f/('cases.jsonl' if phase=='planning' else 'cases.npz');cases.write_bytes(b'synthetic cases')
            raw=dict(research_id='B003' if phase=='planning' else 'C006',status='awaiting_literature_review',
                     n=3,kind=kind,seed=seed,cases_sha256=digest(cases))
            if phase=='planning':raw['arms']={kind:{str(h):dict(count=1000) for h in [4,6,8]}}
            fake_review(f,raw)
    f=root/f'outputs/gated/binding/slots_n3_s{seed}';f.mkdir(parents=True)
    (f/'cases.jsonl').write_bytes(b'synthetic binding')
    fake_review(f,dict(research_id='C002',n=3,kind='slots',seed=seed,cases_sha256=digest(f/'cases.jsonl')))


def p26_fixture(root):
    for stage,folder in [('export','corpus/track_run'),('s1','s1'),('compare','comparison')]:
        sf=root/'outputs/gated/D001'/folder
        fake_review(sf,dict(research_id='D001',status='awaiting_literature_review',step=20000),
                    'measured_negative' if stage=='compare' else 'passed')
        op=root/'outputs/gated/D003/runs'/stage
        atomic_json(op/'manifest.json',dict(synthetic=True))
        fake_review(op,dict(research_id='D003',stage=stage,returncode=0,child_terminal=True,timed_out=False,
            signals=[],errors=[],inputs_unchanged=True,manifest_sha256=digest(op/'manifest.json'),
            child_artifact_sha256={str(sf/'result.json'):digest(sf/'result.json')}))


def training_fixture(root,name):
    sf=root/'outputs/gated/C001'/name;sf.mkdir(parents=True)
    for f in ['model.pt','checkpoint.pt','manifest.json','config.json','metrics.jsonl']:
        (sf/f).write_bytes(b'synthetic training artifact')
    audit=root/'outputs/gated/fixture_final_audit'/name
    fake_review(audit,dict(research_id='fixture_audit',step=303000,
        model_sha256=digest(sf/'model.pt'),checkpoint_sha256=digest(sf/'checkpoint.pt')))
    fake_review(sf,dict(research_id='C001',kind=name[:2],n=3,seed=int(name[-1]),step=303000,
        checkpoint_sha256=digest(sf/'model.pt')),final_audit_review=str(audit/'review.json'),
        final_audit_review_sha256=digest(audit/'review.json'))
    op=root/'outputs/gated/C010/runs'/name;atomic_json(op/'manifest.json',dict(synthetic=True))
    fake_review(op,dict(research_id='C010',name=name,returncode=0,child_terminal=True,timed_out=False,
        actions=[],errors=[],inputs_unchanged=True,manifest_sha256=digest(op/'manifest.json'),
        child_artifact_sha256={f.name:digest(f) for f in sf.iterdir() if f.name!='review.json'}))


started=time.monotonic();_,rh=require_record('C010');out=BASE/'software';out.mkdir(exist_ok=False)
checks={};ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
unchanged={str(f):digest(f) for b in ['B002','B003','C005','C007','C008','C009','D003']
           for f in Path(f'outputs/gated/{b}').rglob('*.py')}
try:
    for f in ['supervision.py','stages.py','run_train.py','check.py']:ast.parse((BASE/f).read_text())
    original=ast.parse(subprocess.check_output(['git','show','7dfc83722da0a9fb805ccc879fe7e5459709a2fd:go2nav/training/balls.py'],text=True))
    current=ast.parse(Path('go2nav/training/balls.py').read_text())
    methods=['model_config','FourierHead','frequencies','fourier_target','to_pixels','Windows','terms','schedule','build','validate']
    def nodes(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert all(ast.dump(nodes(original)[k])==ast.dump(nodes(current)[k]) for k in methods)
    defaults=next(ast.literal_eval(n.value.args[0]) if n.value.args else {k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
                  for n in current.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULTS' for t in n.targets))
    assert defaults==json.loads(Path('outputs/balls/recovery_A014/l0_n3_s0/config.json').read_text())['train']
    checks['ten_original_computational_methods_and_original_defaults_unchanged']=True
    rows=[dict(research_id=r,start=a,end=b) for r,a,b in [('C001',10,30),('C001',15,25),('C004',40,50)]]
    left=remaining(rows,100,'C001');assert left==dict(global_seconds=48*3600-30,bc_seconds=28*3600-20,science_seconds=28*3600-20)
    assert parent.remaining_seconds(rows,100)['bc_seconds']==left['bc_seconds']
    checks['nested_union_and_existing_supervisor_allocation_recognition']=True
    with tempfile.TemporaryDirectory(prefix='c010_fixture_') as tmp:
        root=Path(tmp);lp=root/'ledger.json';atomic_json(lp,[])
        outer,left,work=admit(lp,root/'ops','C001');assert work==28*3600-45
        assert outer['research_id']=='C001' and outer['supervisor_research_id']=='C010'
        old=digest(lp);rejects(lambda:admit(lp,root/'again','C001'));assert digest(lp)==old
        atomic_json(lp,[dict(id='used',research_id='C001',start=0,end=28*3600,out='used')])
        old=digest(lp);rejects(lambda:admit(lp,root/'exhausted','C001'));assert digest(lp)==old
        checks['open_and_exhausted_admission_refuse_without_mutation']=True
        child=root/'ledger_child';child.mkdir();atomic_json(child/'manifest.json',dict(run_id='inner'));now=time.time()
        rows=[dict(id='outer',research_id='C001',supervisor_research_id='C010',start=now-2,end=None,out='ops'),
              dict(id='inner',research_id='C001',start=now-1,end=None,out=str(child)),
              dict(id='foreign',research_id='C002',start=now-1,end=None,out='foreign')]
        atomic_json(lp,rows);assert set(close_intervals(lp,'outer',None,child,root,'C001'))=={'outer','inner'}
        assert json.loads(lp.read_text())[2]==rows[2]
        class Live:
            def poll(self):return None
        atomic_json(lp,rows);old=digest(lp);rejects(lambda:close_intervals(lp,'outer',Live(),child,root,'C001'));assert digest(lp)==old
        bad=[]
        v=copy.deepcopy(rows);v[0]['supervisor_research_id']='C009';bad.append(v)
        v=copy.deepcopy(rows);v[1]['start']=now-3;bad.append(v)
        v=copy.deepcopy(rows);v.append(dict(v[1],id='duplicate'));bad.append(v)
        v=copy.deepcopy(rows);v[1]['id']='wrong';bad.append(v)
        for v in bad:
            atomic_json(lp,v);old=digest(lp);rejects(lambda:close_intervals(lp,'outer',None,child,root,'C001'));assert digest(lp)==old
        checks['owned_canonical_closure_foreign_preservation_and_five_refusals']=True
        for name in ORDER:
            p=route(root,name);argv=arguments('/fixture/python',p)
            assert argv==['/fixture/python','-m','go2nav.training.balls','train','--data',str(root/'outputs/data/balls'),
                          '--n','3','--model',name[:2],'--seed',name[-1],'--research','C001','--out',str(p['science'])]
        for name in ['l0_n3_s0','l1_n1_s1','o1_n3_s1','l0_n3_s3']:rejects(lambda:route(root,name))
        checks['four_exact_default_recipe_commands_and_scope_refusals']=True
        data={}
        for split in ['train','val']:
            for f in ['meta.json','labels.npz','frames.npy']:
                path=root/f'outputs/data/balls/n3/{split}/{f}';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original fixture data')
                data[str(path.relative_to(root))]=digest(path)
        fake_review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256=data))
        assert len(original_data(root))==8
        path=root/'outputs/data/balls/n3/train/frames.npy';path.write_bytes(b'changed data')
        rejects(lambda:original_data(root));path.write_bytes(b'original fixture data')
        checks['original_training_data_provenance_and_changed_bytes_refusal']=True
        rejects(lambda:p26_reviews(root));p26_fixture(root);assert p26_reviews(root)
        op=root/'outputs/gated/D003/runs/compare';good=json.loads((op/'result.json').read_text())
        for key,value in [('stage','s1'),('child_terminal',False),('returncode',1),('timed_out',True),('signals',['TERM']),('errors',['error']),('inputs_unchanged',False),('child_artifact_sha256',{})]:
            fake_review(op,dict(good,**{key:value}));rejects(lambda:p26_reviews(root))
        fake_review(op,good)
        checks['P26_measured_negative_allowed_but_eight_incomplete_operation_variants_refused']=True
        core_fixture(root,0);assert core_reviews(root,0)
        folder=root/'outputs/gated/planning/l0_n3_s0';good=json.loads((folder/'result.json').read_text())
        for key,value in [('seed',1),('status','budget_incomplete'),('cases_sha256','bad')]:
            fake_review(folder,dict(good,**{key:value}));rejects(lambda:core_reviews(root,0))
        bad=copy.deepcopy(good);bad['arms']['l0']['4']['count']=999
        fake_review(folder,bad);rejects(lambda:core_reviews(root,0));fake_review(folder,good)
        checks['complete_core_scope_and_four_identity_partial_tamper_refusals']=True
        d=root/'outputs/gated/D003';(d/'stages.py').write_text(Path('outputs/gated/D003/stages.py').read_text())
        fake_review(d/'software',dict(research_id='D003',source_sha256={'stages.py':digest(d/'stages.py')}))
        for rel,rid in [('D001/adapter_check','D001'),('D002/legacy','D002')]:fake_review(root/'outputs/gated'/rel,dict(research_id=rid))
        files=inputs(route(root,ORDER[0]));assert files
        rejects(lambda:inputs(route(root,ORDER[1])))
        training_fixture(root,ORDER[0]);assert inputs(route(root,ORDER[1]))
        training_fixture(root,ORDER[1]);rejects(lambda:inputs(route(root,ORDER[2])))
        core_fixture(root,1);assert inputs(route(root,ORDER[2]))
        training_fixture(root,ORDER[2]);assert inputs(route(root,ORDER[3]))
        checks['full_admission_fixture_P26_before_seed1_prior_training_and_seed1_core_before_seed2']=True
        sf=root/'outputs/gated/C001'/ORDER[0];(sf/'checkpoint.pt').write_bytes(b'changed full state')
        rejects(lambda:prior_training(root,ORDER[0]))
        checks['prior_completion_audit_binds_full_checkpoint']=True
        p=route(root,ORDER[0]);files=inputs(p);pre=root/'pre.json'
        value=dict(stage='train',name=p['name'],research_sha256='ops',scientific_research_sha256='science',
            inputs_sha256={str(f.resolve()):digest(f) for f in files},consulted_at='fixture',consequences='fixture only',
            sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')])
        atomic_json(pre,value);consultation(pre,p,files,'ops','science')
        for key in ['stage','name','research_sha256','scientific_research_sha256','inputs_sha256','sources']:
            wrong=copy.deepcopy(value);wrong[key]=None;atomic_json(pre,wrong);rejects(lambda:consultation(pre,p,files,'ops','science'))
        atomic_json(pre,value);path=root/'outputs/data/balls/n3/train/frames.npy';path.write_bytes(b'changed again')
        rejects(lambda:consultation(pre,p,files,'ops','science'))
        checks['complete_consultation_and_seven_stale_malformed_refusals']=True
        for mode,duration,deadline in [('normal',.01,2),('deadline',10,.1)]:
            argv=[sys.executable,'-c',f'import time;time.sleep({duration})'];actions=[]
            with (root/f'{mode}.log').open('wb') as log:
                child=subprocess.Popen(argv,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                handle=psutil.Process(child.pid);created=handle.create_time()
                try:
                    timed=wait_owned(child,handle,created,argv,Path.cwd(),time.monotonic()+deadline,actions,poll_seconds=.02,reserve=0,term_seconds=.5,kill_seconds=.5)
                    assert (child.returncode,timed)==((0,False) if mode=='normal' else (-15,True))
                    checks['owned_CPU_'+mode]=dict(returncode=child.returncode,timed_out=timed,actions=actions)
                finally:
                    if child.poll() is None:stop_owned(child,handle,created,argv,Path.cwd(),actions,.5,.5)
    rejects(lambda:inputs(route(Path.cwd(),ORDER[0])))
    checks['current_repo_missing_P26_refuses_admission']=True
    assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(f)==h for f,h in unchanged.items())
    checks['no_Torch_shared_ledger_or_live_helper_mutation']=True
    result=dict(research_id='C010',research_sha256=rh,status='awaiting_literature_review',checks=checks,
        elapsed_seconds=time.monotonic()-started,actual_GPU_runs=0,torch_imported=False,
        source_sha256={f:digest(BASE/f) for f in ['supervision.py','stages.py','run_train.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{rid}.json':digest(f'docs/research/records/{rid}.json') for rid in ['C001','C005','D001','D003']},
        limitation='Synthetic prerequisite/ledger/process fixtures and actual original source/data/missing-gate checks only. No actual extra-seed training or final artifact/numerical validation.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C010',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
