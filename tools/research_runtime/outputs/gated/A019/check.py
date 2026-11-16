"""Focused CPU fixtures; no synthetic review can approve an actual A019 run."""
import ast
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,verify_files
from stages import ORDER,arguments,consultation,inputs,route
from supervision import admit,close_intervals,parent,remaining,stop_owned,wait_owned
from verify_result import artifact_checks

BASE=Path('outputs/gated/A019')


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,FileNotFoundError,KeyError,AssertionError):return
    raise AssertionError('Expected invalid fixture refusal')


def fake_review(folder,raw,status='passed',**extra):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',
        result_sha256=digest(folder/'result.json'),status=status,reviewed_at='synthetic fixture only',
        sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],
        measurements='synthetic only',uncertainty='not actual science',explanations=['fixture'],
        next_step='fixture only',**extra))


def parent_fixture(root,name):
    sf=root/'outputs/gated/C001'/name;sf.mkdir(parents=True)
    for f in ['model.pt','checkpoint.pt','manifest.json','config.json','metrics.jsonl']:
        (sf/f).write_bytes(b'synthetic '+f.encode())
    raw=dict(research_id='C001',kind=name[:2],n=3,seed=int(name[-1]),step=303000,
             checkpoint_sha256=digest(sf/'model.pt'))
    atomic_json(sf/'result.json',raw)
    audit=root/'outputs/gated/A018/runs'/name
    atomic_json(audit/'manifest.json',dict(input_sha256={str(sf/'model.pt'):digest(sf/'model.pt')},
        snapshot_sha256={p.name:digest(p) for p in sf.iterdir()}))
    fake_review(audit,dict(research_id='A018',name=name,status='awaiting_literature_review',step=303000,
        model_sha256=digest(sf/'model.pt'),checkpoint_sha256=digest(sf/'checkpoint.pt'),manifest_sha256=digest(audit/'manifest.json')))
    fake_review(sf,raw,final_audit_review=str(audit/'review.json'),final_audit_review_sha256=digest(audit/'review.json'))
    op=root/'outputs/gated/C010/runs'/name;atomic_json(op/'manifest.json',dict(pid=999999999,creation_time=0))
    fake_review(op,dict(research_id='C010',name=name,returncode=0,child_terminal=True,timed_out=False,
        actions=[],errors=[],inputs_unchanged=True,manifest_sha256=digest(op/'manifest.json'),
        child_artifact_sha256={p.name:digest(p) for p in sf.iterdir() if p.name!='review.json'}))


def state_fixture(kind,seed):
    import torch
    from go2nav.training.balls import build,DEFAULTS
    model,head=build(kind,2026111500+seed,'cpu')
    source=dict(config=asdict(model.config),model_state_dict=copy.deepcopy(model.state_dict()),
                head_state_dict=None if head is None else copy.deepcopy(head.state_dict()),kind=kind,n=3,
                train=DEFAULTS,step=303000,research_id='C001',
                identity=dict(kind=kind,n=3,seed=seed,run_seed=2026111500+seed))
    template=json.loads(Path('outputs/gated/A003_recovered/l0_n3_s0/result.json').read_text())
    result=copy.deepcopy(template);result.update(research_id='A019',kind=kind,source_sha256='fixture_source')
    final={k:(v.double() if v.is_floating_point() else v.clone()) for k,v in source['model_state_dict'].items()}
    for p in ['projector.net.1','pred_proj.net.1']:
        final[p+'.running_mean']+=.5;final[p+'.running_var']+=.25;final[p+'.num_batches_tracked'].zero_()
        for field,key in [('running_mean','mean_change'),('running_var','variance_change')]:
            old,new=source['model_state_dict'][p+'.'+field].double(),final[p+'.'+field].float().double()
            delta=old-new
            result['calibration'][p][key]=dict(passed=bool(torch.allclose(old,new,rtol=1e-4,atol=1e-5)),
                max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()))
    derived={k:copy.deepcopy(source[k]) for k in ['config','kind','n','train','head_state_dict','step']}
    derived.update(model_state_dict=final,research_id='A019',calibration=copy.deepcopy(result['calibration']),
                   source_sha256='fixture_source',inference_only=True,inference_dtype='float64',attention_backend='math')
    return source,derived,result


started=time.monotonic();_,rh=require_record('A019');out=BASE/'software';out.mkdir(exist_ok=False)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger);checks={}
unchanged={str(f):digest(f) for b in ['B002','B003','C005','C007','C008','C009','C010','D003','A016','A018']
           for f in Path('outputs/gated',b).rglob('*.py')}
try:
    assert all(os.environ.get(k)=='' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=subprocess.check_output(['git','show','161fe8b89592516690fcbb8916fd14224b87dfe6:go2nav/eval/balls_inference.py'],text=True)
    assert ast.dump(ast.parse(original))==ast.dump(ast.parse(Path('go2nav/eval/balls_inference.py').read_text()))
    old=json.loads(Path('docs/research/records/A003.json').read_text())['protocol']
    new=json.loads(Path('docs/research/records/A019.json').read_text())['protocol']
    for k in ['diagnostic_windows','sampling_seed','splits','batch_modes','calibration_windows','calibration_seed',
              'calibration_split','calibration_batch','calibration_order','rtol','atol','inference_dtype','attention_backend']:
        assert old[k]==new[k]
    checks['original_A003_computation_AST_and_all_numerical_settings_unchanged']=True
    for name in ORDER:
        p=route(Path.cwd(),name);argv=arguments('fixture_python',p)
        assert argv==['fixture_python','-m','go2nav.eval.balls_inference','--checkpoint',str(p['checkpoint']),
                      '--data',str(p['data']),'--out',str(p['science']),'--research','A019']
    for name in ['l0_n3_s0','l1_n1_s1','o1_n3_s1','../l0_n3_s1']:rejects(lambda:route(Path.cwd(),name))
    checks['four_approved_routes_and_invalid_names_refused']=True
    rows=[dict(research_id=k,start=a,end=b) for k,a,b in [('A003',10,30),('A016',15,25),('A019',40,45),('C001',50,60)]]
    assert remaining(rows,100,'A019')==dict(global_seconds=48*3600-35,a_seconds=4*3600-25)
    checks['historical_and_new_A_nested_intervals_count_once']=True
    with tempfile.TemporaryDirectory(prefix='a019_fixture_') as td:
        root=Path(td);lp=root/'ledger.json';atomic_json(lp,[])
        outer,left,work=admit(lp,root/'ops','A019');assert work==1755
        saved=digest(lp);rejects(lambda:admit(lp,root/'again','A019'));assert digest(lp)==saved
        atomic_json(lp,[dict(research_id='A003',start=0,end=4*3600-50)])
        _,_,work=admit(lp,root/'near','A019');assert work==5
        for k,h in [('A003',4),('C001',48)]:
            atomic_json(lp,[dict(research_id=k,start=0,end=h*3600-44)])
            saved=digest(lp);rejects(lambda:admit(lp,root/'exhausted','A019'));assert digest(lp)==saved
        checks['case_cap_cleanup_and_both_exhaustion_guards']=True
        child=root/'child';child.mkdir();atomic_json(child/'manifest.json',dict(run_id='inner'));now=time.time()
        rows=[dict(id='outer',research_id='A019',supervisor_research_id='A019',start=now-2,end=None,out='ops'),
              dict(id='inner',research_id='A019',start=now-1,end=None,out=str(child)),
              dict(id='foreign',research_id='C001',start=now-1,end=None,out='foreign')]
        atomic_json(lp,rows);assert set(close_intervals(lp,'outer',None,child,root,'A019'))=={'outer','inner'}
        assert json.loads(lp.read_text())[2]==rows[2]
        class Live:
            def poll(self):return None
        for changed in [rows,rows+[copy.deepcopy(rows[1])]]:
            atomic_json(lp,changed);saved=digest(lp)
            rejects(lambda:close_intervals(lp,'outer',Live() if changed==rows else None,child,root,'A019'))
            assert digest(lp)==saved
        checks['owned_nested_closure_preserves_foreign_and_refuses_live_or_ambiguous']=True
        data={}
        for split in ['train','val']:
            for f in ['meta.json','labels.npz','frames.npy']:
                p=root/f'outputs/data/balls/n3/{split}'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original synthetic data '+str(p).encode());data[str(p)]=digest(p)
        fake_review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256={str(Path(k).relative_to(root)):v for k,v in data.items()}))
        for label in ['C010','A018']:
            folder=root/'outputs/gated'/label/'software';folder.mkdir(parents=True,exist_ok=True)
            for f in ['review.json','result.json']:(folder/f).write_bytes(b'bound fixture software only')
        for name in ORDER:
            parent_fixture(root,name);assert len(inputs(route(root,name)))>=20
        checks['all_four_complete_training_audit_and_original_data_chains']=True
        p=route(root,ORDER[0]);sf=root/'outputs/gated/C001'/ORDER[0];audit=root/'outputs/gated/A018/runs'/ORDER[0]
        # Every corruption changes actual evidence, not just a name or status flag.
        for file,replace in [(sf/'model.pt',b'changed'),(sf/'checkpoint.pt',b'changed'),
                             (audit/'manifest.json',b'{}'),(sf/'result.json',b'{}'),
                             (audit/'review.json',b'{}'),(root/'outputs/data/balls/n3/train/frames.npy',b'changed')]:
            saved=file.read_bytes();file.write_bytes(replace);rejects(lambda:inputs(p));file.write_bytes(saved)
        checks['six_changed_or_missing_parent_data_evidence_refusals']=True
        software=root/'outputs/gated/A019/software';software.mkdir(parents=True);(software/'review.json').write_bytes(b'fixture software review')
        files=inputs(p);pre=dict(name=p['name'],research_sha256=rh,scientific_research_sha256=rh,
            inputs_sha256={str(f):digest(f) for f in files},software_review_sha256=digest(software/'review.json'),
            consulted_at='fixture',sources=['fixture'],consequences='fixture only')
        f=root/'consultation.json';atomic_json(f,pre);assert consultation(f,p,files,rh,rh)==pre
        for key,value in [('name','l1_n3_s2'),('inputs_sha256',{}),('sources',[]),('software_review_sha256','changed')]:
            bad=copy.deepcopy(pre);bad[key]=value;atomic_json(f,bad);rejects(lambda:consultation(f,p,files,rh,rh))
        checks['complete_consultation_and_four_stale_scope_input_refusals']=True
        actions=[];argv=[sys.executable,'-c','pass'];childp=subprocess.Popen(argv,cwd=root)
        proc=psutil.Process(childp.pid);ct=proc.create_time()
        assert not wait_owned(childp,proc,ct,argv,root,time.monotonic()+2,actions,poll_seconds=.01) and childp.returncode==0 and not actions
        argv=[sys.executable,'-c','import time; time.sleep(20)'];childp=subprocess.Popen(argv,cwd=root)
        proc=psutil.Process(childp.pid);ct=proc.create_time();actions=[]
        try:
            rejects(lambda:stop_owned(childp,proc,ct+1,argv,root,actions,.1,.1));assert not actions and childp.poll() is None
            assert wait_owned(childp,proc,ct,argv,root,time.monotonic()+.04,actions,poll_seconds=.01,term_seconds=.2,kill_seconds=.2)
            assert childp.poll() is not None and actions[0]['signal']=='SIGTERM'
        finally:
            if childp.poll() is None:stop_owned(childp,proc,ct,argv,root,actions,.2,.2)
        checks['actual_owned_normal_deadline_and_wrong_identity_processes']=True
    for kind in ['l0','l1']:
        for seed in [1,2]:
            source,derived,result=state_fixture(kind,seed)
            checked=artifact_checks(source,derived,result,kind,seed,'fixture_source')
            assert checked['reported_comparisons']==270 and len(checked['changed_buffers'])==4
    checks['four_default_architecture_exports_and_complete_270_comparison_schemas']=True
    source,derived,result=state_fixture('l1',1);corruptions=[]
    for key,value in [('inference_only',False),('inference_dtype','float32'),('attention_backend','flash'),('source_sha256','wrong')]:
        bad=copy.deepcopy(derived);bad[key]=value;corruptions.append((source,bad,result))
    bad=copy.deepcopy(derived);bad['optimizer']={};corruptions.append((source,bad,result))
    weight=next(k for k in source['model_state_dict'] if k.endswith('weight'))
    for field in [weight,'projector.net.1.running_var']:
        bad=copy.deepcopy(derived);bad['model_state_dict'][field].flatten()[0]=float('nan');corruptions.append((source,bad,result))
    bad=copy.deepcopy(derived);bad['model_state_dict'][weight].flatten()[0]+=.1;corruptions.append((source,bad,result))
    bad=copy.deepcopy(derived);bad['head_state_dict'][next(iter(bad['head_state_dict']))].flatten()[0]+=.1;corruptions.append((source,bad,result))
    bad=copy.deepcopy(source);bad['identity']['seed']=2;corruptions.append((bad,derived,result))
    for section in ['before','after']:
        bad=copy.deepcopy(result);bad[section]['parity']['gradients'].pop(next(iter(bad[section]['parity']['gradients'])));corruptions.append((source,derived,bad))
        bad=copy.deepcopy(result);bad[section]['invariance']['single']['encoded']['passed']=False;corruptions.append((source,derived,bad))
        bad=copy.deepcopy(result);bad[section]['parity']['forward']['predicted']['max_abs']=float('nan');corruptions.append((source,derived,bad))
    bad=copy.deepcopy(result);bad['calibration']['projector.net.1']['count']=4095;d=copy.deepcopy(derived);d['calibration']=bad['calibration'];corruptions.append((source,d,bad))
    bad=copy.deepcopy(result);bad['calibration']['projector.net.1']['mean_change']['max_abs']=12.;d=copy.deepcopy(derived);d['calibration']=bad['calibration'];corruptions.append((source,d,bad))
    for s,d,r in corruptions:rejects(lambda:artifact_checks(s,d,r,'l1',1,'fixture_source'))
    checks['eighteen_invalid_numerical_identity_weight_head_dtype_and_calibration_fixtures']=len(corruptions)
    assert len(corruptions)==18
    try:inputs(route(Path.cwd(),'l0_n3_s1'))
    except FileNotFoundError as exc:refusal=str(exc)
    else:raise AssertionError('Expected actual missing extra-seed prerequisite')
    assert 'outputs/gated/C001/l0_n3_s1/review.json' in refusal
    checks['actual_current_missing_training_review_refusal_before_output']=refusal
    assert not torch.cuda.is_initialized() and digest(ledger)==before
    assert all(digest(f)==h for f,h in unchanged.items())
    checks['CPU_only_and_existing_live_helpers_and_shared_ledger_unchanged']=True
    repository=json.loads(Path('outputs/gated/C010/software/result.json').read_text())['repository_source_sha256']
    verify_files(Path.cwd(),repository)
    result=dict(research_id='A019',research_sha256=rh,status='awaiting_literature_review',checks=checks,
        elapsed_seconds=time.monotonic()-started,source_sha256={p.name:digest(p) for p in BASE.glob('*.py')},
        repository_source_sha256=repository,record_sha256={str(p):digest(p) for p in [Path('docs/research/records')/(n+'.json') for n in ['A003','A016','A018','A019','C010']]},
        actual_extra_seed_evaluations=0,cuda_initialized=False,
        limitation='Synthetic artifacts and CPU process fixtures only. Numerical comparisons copied from a prior valid schema are fixture values, not new GPU measurements. No actual extra-seed checkpoint evaluated.')
    atomic_json(out/'result.json',result);print(json.dumps({k:result[k] for k in ['research_id','status','checks','elapsed_seconds','actual_extra_seed_evaluations']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='A019',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)));raise
