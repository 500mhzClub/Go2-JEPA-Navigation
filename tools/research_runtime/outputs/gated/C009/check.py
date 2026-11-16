"""Focused C009 CPU checks; synthetic reviews never approve real science."""
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
from stages import arguments,consultation,identity,inputs,route

BASE=Path('outputs/gated/C009')


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,FileNotFoundError):return
    raise AssertionError('Expected refusal')


def fake_review(folder,raw):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',result_sha256=digest(folder/'result.json'),status='passed',reviewed_at='synthetic fixture only',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],measurements='synthetic only',uncertainty='not actual science',explanations=['fixture'],next_step='fixture only'))


started=time.monotonic();_,rh=require_record('C009');out=BASE/'software';out.mkdir(exist_ok=False);checks={}
ledger=Path('outputs/gated/gpu_intervals.json');ledger_before=digest(ledger)
unchanged={str(f):digest(f) for b in ['B002','B003','C005','C007','C008','D003'] for f in Path(f'outputs/gated/{b}').rglob('*.py')}
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
        ev=dict(research_id='C004',kind='slots',n=3,seed=0,checks_passed=True);identity(ev,'slots',3,0)
        for key,value in [('research_id','C001'),('kind','o1'),('n',1),('seed',1),('checks_passed',False)]:
            wrong=dict(ev);wrong[key]=value;rejects(lambda:identity(wrong,'slots',3,0))
        checks['selected_parent_identity_and_numerical_gate']=True
        p=route(root,'binding','slots_n3_s0');p['checkpoint'].parent.mkdir(parents=True)
        p['checkpoint'].write_bytes(b'synthetic encoder');atomic_json(p['checkpoint'].with_name('config.json'),dict(kind='slots',n=3,seed=0,parents={}))
        code=root/'go2nav/fixture.py';code.parent.mkdir();code.write_text('synthetic source')
        c4=p['review'].parent;atomic_json(c4/'manifest.json',dict(inputs={},source_sha256={'go2nav/fixture.py':digest(code)}))
        fake_review(c4,dict(ev,checkpoint_sha256=digest(p['checkpoint'])))
        c7=root/'outputs/gated/C007/runs/slots_n3_s0';atomic_json(c7/'manifest.json',dict(synthetic=True))
        fake_review(c7,dict(research_id='C007',name='slots_n3_s0',child_terminal=True,returncode=0,manifest_sha256=digest(c7/'manifest.json'),child_artifact_sha256={f:digest(c4/f) for f in ['result.json','manifest.json']}))
        for file in ['meta.json','frames.npy','labels.npz']:
            q=root/'outputs/data/balls/n3/test'/file;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b'fixture')
        baseline=root/'outputs/gated/probes/l0_n3_s0';baseline_pre=root/'outputs/gated/C006/pre_l0_n3_s0.json'
        original={str(f.relative_to(root)):digest(f) for f in (root/'outputs/data/balls/n3/test').iterdir()}
        atomic_json(baseline_pre,dict(data_sha256=original))
        atomic_json(baseline/'manifest.json',dict(inputs={str(baseline_pre):digest(baseline_pre)}))
        fake_review(baseline,dict(research_id='C006',n=3,kind='l0',seed=0,manifest_sha256=digest(baseline/'manifest.json')))
        files=inputs(p);assert p['checkpoint'] in files and len(files)==15
        pre=root/'pre.json';value=dict(stage='binding',name='slots_n3_s0',research_sha256='ops',scientific_research_sha256='science',inputs_sha256={str(f.resolve()):digest(f) for f in files},consulted_at='fixture',consequences='fixture only',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')])
        atomic_json(pre,value);consultation(pre,p,files,'ops','science')
        for key in ['stage','name','research_sha256','scientific_research_sha256','inputs_sha256','sources']:
            wrong=copy.deepcopy(value);wrong[key]=None;atomic_json(pre,wrong);rejects(lambda:consultation(pre,p,files,'ops','science'))
        atomic_json(pre,value);p['checkpoint'].write_bytes(b'changed');rejects(lambda:consultation(pre,p,files,'ops','science'));rejects(lambda:inputs(p))
        checks['actual_binding_schema_fixture_complete_inputs_and_tamper_refusal']=True
        p['checkpoint'].write_bytes(b'synthetic encoder')
        frames=root/'outputs/data/balls/n3/test/frames.npy';frames.write_bytes(b'changed data')
        fresh=copy.deepcopy(value);fresh['inputs_sha256'][str(frames.resolve())]=digest(frames);atomic_json(pre,fresh)
        rejects(lambda:inputs(p))
        checks['changed_dataset_with_fresh_consultation_rejected_against_original_bytes']=True
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
    assert 'torch' not in sys.modules and digest(ledger)==ledger_before and all(digest(f)==h for f,h in unchanged.items())
    checks['no_torch_no_shared_ledger_or_existing_helper_mutation']=True
    result=dict(research_id='C009',research_sha256=rh,status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-started,
        source_sha256={f:digest(BASE/f) for f in ['supervision.py','stages.py','run_eval.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{rid}.json':digest(f'docs/research/records/{rid}.json') for rid in ['B001','C001','C002','C005','C004']},
        actual_GPU_cases=0,torch_imported=False,limitation='CPU fixtures and binding lineage fixture only. Actual trained predictors/encoders, GPU-stall handling and all scientific outcomes remain untested by this suite.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C009',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
