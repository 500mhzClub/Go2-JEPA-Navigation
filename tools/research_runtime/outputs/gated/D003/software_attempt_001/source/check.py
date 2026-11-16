"""Focused D003 CPU fixtures; no actual P26 stage or shared-ledger changes."""
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
from supervision import admit,close_intervals,remaining_seconds,stop_owned,union_seconds,wait_owned
from stages import arguments,consultation,core_reviews,inventory,paths,stage_parents

base=Path('outputs/gated/D003');_,rh=require_record('D003');out=base/'software';out.mkdir(exist_ok=False)
started=time.monotonic();checks={};ledger_path=Path('outputs/gated/gpu_intervals.json');before=digest(ledger_path)
untouched={str(p):digest(p) for name in ['B002','B003','C005','C007'] for p in Path(f'outputs/gated/{name}').glob('*.py')}


def rejects(call,error=ValueError):
    try:call()
    except error:return
    raise AssertionError('Expected refusal')


def fixture_review(path,result,status='passed',identity=None):
    atomic_json(path.parent/'result.json',result)
    review=dict(research_id=result['research_id'],result='result.json',result_sha256=digest(path.parent/'result.json'),
                reviewed_at='synthetic',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture only')],
                measurements='synthetic fixture, not actual approval',uncertainty='synthetic only',explanations=['fixture'],
                next_step='fixture only',status=status)
    if identity:review['run_identity']=identity
    atomic_json(path,review)


try:
    for f in ['supervision.py','stages.py','run_stage.py','check.py']:ast.parse((base/f).read_text())
    rows=[dict(research_id=rid,start=a,end=b) for rid,a,b in [('A003',0,10),('D003',20,50),('D001',25,40),('D002',60,70),('C004',80,90)]]
    assert remaining_seconds(rows,100)==dict(global_seconds=48*3600-60,d_seconds=8*3600-40)
    checks['nested_union_D_global_and_other_allocations']=True
    with tempfile.TemporaryDirectory(prefix='d003_fixture_') as tmp:
        root=Path(tmp);ledger=root/'ledger.json';atomic_json(ledger,[])
        outer,_,work=admit(ledger,root/'op');assert work==8*3600-45
        saved=digest(ledger);rejects(lambda:admit(ledger,root/'duplicate'));assert digest(ledger)==saved
        for rid,total in [('D001',8*3600),('A003',48*3600)]:
            now=time.time();atomic_json(ledger,[dict(id='old',research_id=rid,out='old',start=now-total,end=now)])
            saved=digest(ledger);rejects(lambda:admit(ledger,root/'exhausted'));assert digest(ledger)==saved
            atomic_json(ledger,[dict(id='old',research_id=rid,out='old',start=now-(total-60),end=now)])
            _,_,work=admit(ledger,root/'remaining');assert abs(work-15)<1e-6
        checks['eight_hour_global_cap_cleanup_and_no_mutation_refusals']=True
        real=root/'real';real.mkdir();(root/'alias').symlink_to(real,target_is_directory=True)
        child_out=real/'child';child_out.mkdir();atomic_json(child_out/'manifest.json',dict(run_id='inner'))
        now=time.time();fixture=[dict(id='outer',research_id='D003',out='op',start=now-10,end=None),
            dict(id='inner',research_id='D001',out='alias/child',start=now-9,end=None),
            dict(id='other',research_id='C001',out='unrelated',start=now-11,end=None)]
        class Terminal:
            def poll(self):return 0
        class Live:
            def poll(self):return None
        atomic_json(ledger,fixture);saved=digest(ledger)
        rejects(lambda:close_intervals(ledger,'outer',Live(),child_out,root),RuntimeError);assert digest(ledger)==saved
        assert set(close_intervals(ledger,'outer',Terminal(),child_out,root))=={'outer','inner'}
        closed=json.loads(ledger.read_text());assert closed[2]==fixture[2] and closed[0]['end']==closed[1]['end']
        assert abs(union_seconds(closed[:2],closed[0]['end'])-(closed[0]['end']-fixture[0]['start']))<1e-6
        for kind in ['historical','duplicate','manifest','outer']:
            bad=copy.deepcopy(fixture)
            if kind=='historical':bad[1]['start']=bad[0]['start']-1
            elif kind=='duplicate':bad.append(dict(bad[1],id='another'))
            elif kind=='manifest':bad[1]['id']='wrong'
            else:bad[0]['research_id']='C005'
            atomic_json(ledger,bad);saved=digest(ledger)
            rejects(lambda:close_intervals(ledger,'outer',Terminal(),child_out,root));assert digest(ledger)==saved
        checks['canonical_terminal_owned_only_closure_and_invalid_rows']=True

        corpus=root/'outputs/gated/D001/corpus';features=root/'features';features.mkdir()
        units={s:[f'units/lewm_{s}/{i:04d}_fixture' for i in range(count)] for s,count in [('train',600),('val',100)]}
        atomic_json(features/'index.json',dict(units={'lewm_'+s:v for s,v in units.items()},grid=[12,16]))
        fh=digest(features/'index.json')
        atomic_json(corpus/'prepare_manifest.json',dict(inputs={str(features/'index.json'):fh}))
        atomic_json(corpus/'index.json',dict(features_index_sha256=fh,units=units))
        for split,values in units.items():
            for i,unit in enumerate(values):
                meta=corpus/split/f'{i:05d}.json';atomic_json(meta,dict(unit=unit,maze=1))
                meta.with_suffix('.npz').write_bytes(b'fixture')
                f=features/('lewm_'+split)/(Path(unit).name+'.npy');f.parent.mkdir(exist_ok=True);f.write_bytes(b'fixture')
            for name in ['pose','requested','ranges','starts']:
                f=corpus/'packed'/f'{split}_{name}.npy';f.parent.mkdir(exist_ok=True);f.write_bytes(b'fixture')
        atomic_json(root/'data/mazes/fixture/registry.json',dict(fixture=True))
        for name in ['s0/model.pt','s0/config.json']:
            f=root/'outputs/camera_slots'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'fixture')
        p=paths(root,'export');inputs=inventory(p);assert len(inputs)==2114
        cmd=arguments('/fixture/python',p)
        assert cmd==['/fixture/python','-m','go2nav.data.slot_tracks','export','--corpus',str(corpus),'--features',str(features),
                    '--checkpoint',str(p['s0']),'--mazes',str(root/'data/mazes'),'--research','D001']
        for stage in ['s1','compare']:
            q=paths(root,stage);a=arguments('/fixture/python',q)
            assert a[:3]==['/fixture/python','-m','go2nav.training.object_jepa' if stage=='s1' else 'go2nav.eval.slot_dynamics']
            assert a.count('--research')==1 and a[a.index('--research')+1]=='D001'
            if stage=='s1':assert a[a.index('--set')+1]=='seed=0' and a[a.index('--config')+1]==str(root/'configs/object/m5.yaml')
            else:assert a[a.index('--m5')+1]==str(q['m5']) and a[a.index('--s1')+1]==str(q['s1'])
        rejects(lambda:paths(root,'prepare'))
        checks['fixed_three_stage_routing_and_prepared_inventory']=True
        value=dict(stage='export',d003_record_sha256='ops',d001_record_sha256='science',inputs_sha256={str(f.resolve()):digest(f) for f in inputs},reviews_sha256={},
                   consulted_at='fixture',consequences='synthetic only',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')])
        pre=root/'pre.json';atomic_json(pre,value);consultation(pre,p,inputs,[],'ops','science')
        for key in ['stage','d003_record_sha256','d001_record_sha256','inputs_sha256','reviews_sha256','sources']:
            wrong=copy.deepcopy(value);wrong[key]=None;atomic_json(pre,wrong)
            rejects(lambda:consultation(pre,p,inputs,[],'ops','science'))
        atomic_json(pre,value);inputs[0].write_bytes(b'changed');rejects(lambda:consultation(pre,p,inputs,[],'ops','science'))
        checks['stale_or_incomplete_stage_source_parent_inputs_refused']=True

        core=root/'core'
        for kind in ['l0','l1','o0','o1']:
            for phase in ['planning','probes']:
                folder=core/'outputs/gated'/phase/f'{kind}_n3_s0';folder.mkdir(parents=True)
                raw=folder/('cases.jsonl' if phase=='planning' else 'cases.npz');raw.write_bytes(b'synthetic only')
                result=dict(research_id='fixture',n=3,kind=kind,seed=0,cases_sha256=digest(raw))
                identity=None
                if kind=='o0' and phase=='planning':
                    result.pop('kind');result.pop('seed');identity=dict(n=3,kind=kind,seed=0)
                fixture_review(folder/'review.json',result,status='measured_negative' if kind=='l1' else 'passed',identity=identity)
        fixture_review(core/'outputs/gated/binding/slots_n3_s0/review.json',dict(research_id='fixture',n=3,kind='slots',seed=0))
        for phase in ['D001/adapter_check','D002/legacy']:fixture_review(core/'outputs/gated'/phase/'review.json',dict(research_id='fixture'))
        assert len(core_reviews(core))==11
        target=core/'outputs/gated/probes/l0_n3_s0/review.json'
        fixture_review(target,dict(research_id='fixture',n=1,kind='l0',seed=0,cases_sha256=digest(target.parent/'cases.npz')))
        rejects(lambda:core_reviews(core))
        checks['core_identity_legacy_seed_and_measured_negative_gates']=True

        stage_root=root/'stage_parents';c=stage_root/'outputs/gated/D001/corpus';atomic_json(c/'prepare_manifest.json',dict(inputs={str(features/'index.json'):fh}))
        atomic_json(c/'tracks_index.json',dict(files={}))
        fixture_review(c/'track_run/review.json',dict(research_id='D001',tracks_sha256=digest(c/'tracks_index.json')))
        fixture_review(stage_root/'outputs/gated/D003/runs/export/review.json',dict(research_id='D003',stage='export',child_terminal=True,returncode=0,child_artifact_sha256={}))
        assert len(stage_parents(paths(stage_root,'s1')))==2
        q=paths(stage_root,'compare');q['s1'].parent.mkdir(parents=True);q['s1'].write_bytes(b'selected fixture')
        q['m5'].parent.mkdir(parents=True);q['m5'].write_bytes(b'original fixture')
        fixture_review(q['s1'].parent/'review.json',dict(research_id='D001',step=20000,checkpoint_sha256=digest(q['s1'])))
        fixture_review(stage_root/'outputs/gated/D003/runs/s1/review.json',dict(research_id='D003',stage='s1',child_terminal=True,returncode=0,child_artifact_sha256={}))
        atomic_json(stage_root/'outputs/gated/D002/legacy/manifest.json',dict(inputs={'outputs/object_dynamics/m5/model.pt':digest(q['m5'])}))
        assert len(stage_parents(q))==4
        fixture_review(q['s1'].parent/'review.json',dict(research_id='D001',step=19999,checkpoint_sha256=digest(q['s1'])))
        rejects(lambda:stage_parents(q))
        checks['actual_stage_review_and_full_S1_step_guards_on_fixtures']=True

        for mode,duration,deadline in [('normal',.02,2),('deadline',10,.1)]:
            argv=[sys.executable,'-c',f'import time;time.sleep({duration})'];signals=[]
            with (root/f'{mode}.log').open('wb') as log:
                child=subprocess.Popen(argv,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                process=psutil.Process(child.pid);identity=process.create_time()
                try:
                    timed=wait_owned(child,process,identity,argv,Path.cwd(),time.monotonic()+deadline,signals,poll_seconds=.02,reserve=0,term_seconds=.5,kill_seconds=.5)
                    assert child.poll() is not None
                    assert (child.returncode,timed)==((0,False) if mode=='normal' else (-15,True))
                    checks[f'owned_CPU_{mode}']=dict(returncode=child.returncode,timed_out=timed,signals=signals)
                finally:
                    if child.poll() is None:stop_owned(child,process,identity,argv,Path.cwd(),signals,.5,.5)

    real=paths(Path.cwd(),'export');actual_inputs=inventory(real)
    checks['actual_prepared700_units_inventory_without_GPU']=dict(files=len(actual_inputs),feature_index=str(real['feature_index']))
    assert digest(ledger_path)==before and all(digest(f)==h for f,h in untouched.items()) and 'torch' not in sys.modules
    checks['shared_ledger_and_live_sources_unchanged_no_torch']=True
    result=dict(research_id='D003',research_sha256=rh,d001_record_sha256=digest('docs/research/records/D001.json'),status='awaiting_literature_review',
        checks=checks,elapsed_seconds=time.monotonic()-started,actual_GPU_cases=0,torch_imported=False,
        source_sha256={f:digest(base/f) for f in ['supervision.py','stages.py','run_stage.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in sorted(Path('go2nav').rglob('*.py'))},
        config_sha256={str(f):digest(f) for f in sorted(Path('configs').rglob('*.yaml'))},
        frozen_model_sha256={str(f.relative_to(Path.cwd())):digest(f) for f in [real['s0'],real['s0'].with_name('config.json'),real['m5'],real['m5'].with_name('config.json')]},
        a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'),
        limitation='CPU fixtures and current inventory/source checks only. No actual n3-dependent P26 stage or full pre-admission data hashing performed.')
    atomic_json(out/'result.json',result);print(json.dumps(dict(checks=checks,elapsed_seconds=result['elapsed_seconds'])),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='D003',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
