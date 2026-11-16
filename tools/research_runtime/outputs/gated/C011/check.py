"""C011 CPU fixtures; no actual additional-seed probe or automatic scientific review."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
from analysis import sample,targets,features,fit_linear,apply_linear,residual,finite,guard,np,torch
from go2nav.eval.balls_probes import probe_summary
from parents import ORDER,numerical_parent,original_probe_data,previous_reviews,consultation
from run_one import cpu_lock,supervise

BASE=Path('outputs/gated/C011')

def rejects(call,exception=(ValueError,RuntimeError,FileNotFoundError,KeyError,AssertionError,BlockingIOError)):
    try:call()
    except exception:return
    raise AssertionError('Expected fixture refusal')

def fake_review(folder,raw,status='passed',**extra):
    atomic_json(folder/'result.json',raw)
    atomic_json(folder/'review.json',dict(research_id=raw['research_id'],result='result.json',
        result_sha256=digest(folder/'result.json'),status=status,reviewed_at='synthetic fixture only',
        sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')],
        measurements='synthetic only',uncertainty='not actual science',explanations=['fixture'],next_step='fixture only',**extra))

def training_fixture(root,name):
    sf=root/'outputs/gated/C001'/name;sf.mkdir(parents=True)
    for f in ['model.pt','checkpoint.pt','manifest.json','metrics.jsonl']:(sf/f).write_bytes(b'synthetic '+f.encode())
    atomic_json(sf/'config.json',dict(kind=name[:2],n=3,seed=int(name[-1]),train={'fixture':True}))
    raw=dict(research_id='C001',kind=name[:2],n=3,seed=int(name[-1]),step=303000,checkpoint_sha256=digest(sf/'model.pt'))
    atomic_json(sf/'result.json',raw);audit=root/'outputs/gated/A018/runs'/name
    atomic_json(audit/'manifest.json',dict(input_sha256={str(sf/'model.pt'):digest(sf/'model.pt')},snapshot_sha256={p.name:digest(p) for p in sf.iterdir()}))
    fake_review(audit,dict(research_id='A018',name=name,status='awaiting_literature_review',step=303000,
        model_sha256=digest(sf/'model.pt'),checkpoint_sha256=digest(sf/'checkpoint.pt'),manifest_sha256=digest(audit/'manifest.json')))
    fake_review(sf,raw,final_audit_review=str(audit/'review.json'),final_audit_review_sha256=digest(audit/'review.json'))
    op=root/'outputs/gated/C010/runs'/name;atomic_json(op/'manifest.json',dict(pid=999999999,creation_time=0))
    fake_review(op,dict(research_id='C010',name=name,returncode=0,child_terminal=True,timed_out=False,
        actions=[],errors=[],inputs_unchanged=True,manifest_sha256=digest(op/'manifest.json'),
        child_artifact_sha256={p.name:digest(p) for p in sf.iterdir() if p.name!='review.json'}))

def numerical_fixture(root,name):
    training_fixture(root,name);sf=root/'outputs/gated/C001'/name
    folder=root/'outputs/gated/A019/inference'/name;folder.mkdir(parents=True)
    (folder/'source.pt').write_bytes((sf/'model.pt').read_bytes());(folder/'calibrated.pt').write_bytes(b'calibrated fixture '+name.encode())
    atomic_json(folder/'manifest.json',dict(research_id='A019'))
    raw=dict(research_id='A019',kind=name[:2],n=3,step=303000,status='awaiting_literature_review',weights_unchanged=True,invalid_state=[],
        source_sha256=digest(folder/'source.pt'),calibrated_sha256=digest(folder/'calibrated.pt'))
    atomic_json(folder/'result.json',raw)
    op=root/'outputs/gated/A019/runs'/name
    atomic_json(op/'manifest.json',dict(preconsultation=dict(inputs_sha256={str(sf/'model.pt'):digest(sf/'model.pt')})))
    fake_review(op,dict(research_id='A019',name=name,returncode=0,child_terminal=True,timed_out=False,actions=[],errors=[],inputs_unchanged=True,
        manifest_sha256=digest(op/'manifest.json'),child_artifact_sha256={p.name:digest(p) for p in folder.iterdir()}))
    vf=root/'outputs/gated/A019/verification'/name
    fake_review(vf,dict(research_id='A019',name=name,kind=name[:2],n=3,seed=int(name[-1]),
        checks=dict(reported_comparisons=270,gradient_tensors_per_phase=127,learned_weights_and_head_exact=True,finite_float64_math_inference_only=True,cuda_initialized=False),
        calibrated_sha256=digest(folder/'calibrated.pt'),scientific_result_sha256=digest(folder/'result.json'),operational_result_sha256=digest(op/'result.json')))
    fake_review(folder,raw,run_identity=dict(kind=name[:2],n=3,seed=int(name[-1])),verification_review=str(vf/'review.json'),verification_review_sha256=digest(vf/'review.json'))

def data_fixture(root):
    hashes={}
    for split in ['train','val','test']:
        for name in ['meta.json','labels.npz','frames.npy']:
            p=root/f'outputs/data/balls/n3/{split}'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original fixture '+str(p).encode());hashes[str(p.relative_to(root))]=digest(p)
    fake_review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256=hashes))
    for label in ['A018','C010']:
        f=root/'outputs/gated'/label/'software';f.mkdir(parents=True)
        for name in ['result.json','review.json']:(f/name).write_bytes(b'bound synthetic software')
    pre=root/'outputs/gated/C006/pre_l0_n3_s0.json';atomic_json(pre,dict(data_sha256=hashes))
    baseline=root/'outputs/gated/probes/l0_n3_s0';baseline.mkdir(parents=True)
    (baseline/'cases.npz').write_bytes(b'original synthetic cases');atomic_json(baseline/'manifest.json',dict(inputs={str(pre):digest(pre)}))
    fake_review(baseline,dict(research_id='C006',n=3,kind='l0',seed=0,manifest_sha256=digest(baseline/'manifest.json'),cases_sha256=digest(baseline/'cases.npz')))
    controls=root/'outputs/gated/B001/n3';controls.mkdir(parents=True);(controls/'cases.jsonl').write_bytes(b'original control cases')
    atomic_json(controls/'manifest.json',dict(inputs=hashes))
    fake_review(controls,dict(research_id='B001',n=3,controls_passed=True,cases_sha256=digest(controls/'cases.jsonl')))
    return hashes

started=time.monotonic();_,rh=require_record('C011');out=BASE/'software';out.mkdir(exist_ok=False);checks={}
ledger=Path('outputs/gated/gpu_intervals.json');ledger_before=digest(ledger)
unchanged={str(p):digest(p) for b in ['B002','B003','C005','C006','C007','C008','C009','C010','A016','A018','A019','D003'] for p in Path('outputs/gated',b).rglob('*.py')}
try:
    torch.set_num_threads(1)
    for p in BASE.glob('*.py'):ast.parse(p.read_text())
    old=ast.parse(Path('outputs/gated/C006/analysis.py').read_text());new=ast.parse((BASE/'analysis.py').read_text())
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('C011','C006').replace('inputs_sha256','data_sha256')
            return node
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ['guard','finite','sample','residual']:
        assert ast.dump(function(old,name))==ast.dump(Normalize().visit(copy.deepcopy(function(new,name))))
    oldbody=next(n for n in ast.walk(function(old,'run')) if isinstance(n,ast.With))
    newbody=next(n for n in ast.walk(function(new,'run')) if isinstance(n,ast.With))
    assert ast.dump(oldbody)==ast.dump(Normalize().visit(copy.deepcopy(newbody)))
    inherited=json.loads(Path('outputs/gated/C006/software/result.json').read_text());verify_files(Path('outputs/gated/C006'),inherited['source_sha256'])
    verify_files(Path.cwd(),inherited['repository_source_sha256'])
    checks['unchanged_C006_actual_computational_body_helpers_and_imported_science']=True
    guard()
    for n in [3]:
        for split in ['train', 'test']:
            e, t = sample(1000, 100, n, split)
            rng = np.random.default_rng([2026111511, n, split == 'test'])
            expected_e = rng.integers(1000, size=4096) if split == 'train' else np.repeat(np.arange(1000), 5)
            expected_t = rng.integers(2, 96, size=len(expected_e))
            assert np.array_equal(e, expected_e) and np.array_equal(t, expected_t)
            assert t.min() >= 2 and t.max()+4 < 100
            if split == 'test':
                assert np.array_equal(np.bincount(e), np.full(1000, 5))
    rejects(lambda: sample(100, 100, 3, 'val'), ValueError)
    checks['exact_C001_n3_sampling_and_bounds'] = True

    red = np.array([[2., 3.], [4., 5.]])
    white = np.array([[[4., 2.], [1., 3.], [1., 1.]], [[3., 4.], [2., 1.], [3., 2.]]])
    np.testing.assert_array_equal(targets(red, white), targets(red, white[:, [2, 0, 1]]))
    np.testing.assert_array_equal(targets(red, white)[0], [2, 3, 1, 1, 1, 3, 4, 2])
    checks['white_lexicographic_sort_and_permutation_invariance'] = True

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.observations = []
            self.actions = []

        def encode(self, pixels):
            self.observations.append(pixels.clone())
            return pixels.mean((-2, -1))[..., :2]

        def rollout(self, z, a, steps):
            self.actions.append(a.clone())
            # Order-sensitive output exposes accidental future-action permutations.
            weights = torch.arange(1, 7, dtype=a.dtype).reshape(1, 6, 1)
            end = z[:, -1]+(a*weights).sum(1)
            return end[:, None].expand(-1, steps, -1)

    frames = np.zeros((2, 12, 4, 4, 3), dtype=np.uint8)
    for ep in range(2):
        for t in range(12):
            frames[ep, t] = ep*100+t
    actions = np.arange(48, dtype=np.float64).reshape(2, 12, 2)
    episodes, starts = np.array([0, 1, 0]), np.array([3, 6, 5])
    model = RecordingModel()
    values = features(model, frames, actions, episodes, starts, batch=32)
    factual, reversed_actions = model.actions
    expected = torch.from_numpy(actions[episodes[:, None], starts[:, None]+np.arange(-2, 4)])/3
    torch.testing.assert_close(factual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reversed_actions[:, :2], factual[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(reversed_actions[:, 2:], factual[:, 2:].flip(1), rtol=0, atol=0)
    assert not torch.equal(values['rollout'], values['shuffled'])
    assert torch.equal(values['current'], values['no_rollout'])
    altered = frames.copy()
    # This separate single case alters every future image but retains its history.
    altered[0, 4:] = 255
    before = features(RecordingModel(), frames, actions, episodes[:1], starts[:1])
    after = features(RecordingModel(), altered, actions, episodes[:1], starts[:1])
    assert all(torch.equal(before[k], after[k]) for k in before)
    checks['past_actions_preserved_future_reversed_no_future_image_leakage'] = True
    singles = features(RecordingModel(), frames, actions, episodes, starts, batch=1)
    reverse = features(RecordingModel(), frames, actions, episodes[::-1].copy(), starts[::-1].copy(), batch=3)
    for key, value in values.items():
        torch.testing.assert_close(value, singles[key], rtol=0, atol=0)
        torch.testing.assert_close(value, reverse[key].flip(0), rtol=0, atol=0)
    checks['feature_batch_and_permutation_consistency_fake_model'] = True

    rng = np.random.default_rng(42)
    x = rng.normal(size=(257, 8)); x[:, -1] = 3  # Constant column exercises std floor.
    y = x@rng.normal(size=(8, 4))+rng.normal(size=4)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    decoder = fit_linear(xt, yt)
    mean, std, w = decoder
    design = np.column_stack(((x-mean.numpy())/std.numpy(), np.ones(len(x))))
    penalty = np.diag([.001]*8+[0.])
    augmented = np.vstack((design/np.sqrt(len(x)), np.sqrt(penalty)))
    rhs = np.vstack((y/np.sqrt(len(x)), np.zeros((9, 4))))
    reference = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    np.testing.assert_allclose(w.numpy(), reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(apply_linear(xt, decoder).numpy(), design@reference, rtol=1e-10, atol=1e-10)
    ridge_residual = residual(xt, yt, decoder)
    assert ridge_residual <= 1e-8
    checks['ridge_augmented_lstsq_constant_column_and_residual'] = True

    for value in [float('nan'), np.array([np.inf]), torch.tensor([float('nan')]), {'nested': [float('inf')]}]:
        rejects(lambda: finite(value), ValueError)
    saved = os.environ.pop('HIP_VISIBLE_DEVICES')
    try:
        rejects(guard, RuntimeError)
    finally:
        os.environ['HIP_VISIBLE_DEVICES'] = saved
    guard()
    checks['nonfinite_and_visible_GPU_environment_rejected'] = True

    truth=np.arange(16,dtype=np.float64).reshape(8,2);pred=truth.copy();pred[[2,3,6,7]]+=np.array([4.,-3.])
    episodes=np.repeat([10,20,30,40],2);summary=probe_summary(pred,truth,episodes)
    rng=np.random.default_rng(2026111510);values=[]
    for _ in range(2000):
        chosen=rng.integers(4,size=4);x=pred.reshape(4,2,2)[chosen].reshape(8,2);y=truth.reshape(4,2,2)[chosen].reshape(8,2)
        values.append([float(np.corrcoef(x[:,k],y[:,k])[0,1]) if np.std(x[:,k])>0 and np.std(y[:,k])>0 else 0. for k in range(2)])
    np.testing.assert_allclose(summary['pearson_ci95'],np.quantile(values,[.025,.975],axis=0).T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(summary['rmse_px'],np.sqrt(np.mean((pred-truth)**2,axis=0)),rtol=1e-12,atol=1e-12)
    checks['independent_whole_episode_bootstrap_and_point_error_fixture']=True

    with tempfile.TemporaryDirectory(prefix='c011_fixture_') as td:
        root=Path(td);data_fixture(root)
        for name in ORDER:
            numerical_fixture(root,name);ctx=numerical_parent(root,name)
            assert ctx['config']['seed']==int(name[-1]) and len(ctx['files'])>=30
        checks['four_complete_training_and_three_numerical_review_chains']=True
        datafiles,data=original_probe_data(root);assert len(data)==6
        saved={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
        def restore():
            for p,value in saved.items():p.write_bytes(value)
        def change(path,keys,value):
            doc=json.loads(path.read_text());target=doc
            for key in keys[:-1]:target=target[key]
            target[keys[-1]]=value;atomic_json(path,doc)
            if path.name=='result.json':
                review=path.with_name('review.json');r=json.loads(review.read_text());r['result_sha256']=digest(path);atomic_json(review,r)
                if path.parent.parent.name=='verification':
                    science=root/'outputs/gated/A019/inference'/path.parent.name/'review.json';r=json.loads(science.read_text())
                    r['verification_review_sha256']=digest(review);atomic_json(science,r)
        name=ORDER[0];science=root/'outputs/gated/A019/inference'/name;vf=root/'outputs/gated/A019/verification'/name;op=root/'outputs/gated/A019/runs'/name
        bad=[(science/'review.json',['status'],'measured_negative'),(science/'review.json',['run_identity','seed'],2),
             (science/'review.json',['verification_review_sha256'],'changed'),(science/'result.json',['kind'],'l1'),
             (vf/'result.json',['seed'],2),(vf/'result.json',['checks','reported_comparisons'],269),
             (vf/'result.json',['checks','cuda_initialized'],True),(op/'result.json',['returncode'],1),
             (op/'result.json',['child_terminal'],False),(op/'result.json',['inputs_unchanged'],False)]
        for path,keys,value in bad:
            change(path,keys,value);rejects(lambda:numerical_parent(root,name));restore()
        for path in [science/'calibrated.pt',science/'source.pt']:
            path.write_bytes(b'changed');rejects(lambda:numerical_parent(root,name));restore()
        checks['twelve_wrong_or_partial_numerical_review_and_artifact_refusals']=True
        path=root/'outputs/data/balls/n3/test/frames.npy';path.write_bytes(b'changed test with fresh consultation proposed')
        fresh=digest(path);assert fresh!=data[str(path.relative_to(root))]
        rejects(lambda:original_probe_data(root));restore()
        control=root/'outputs/gated/B001/n3/result.json';change(control,['controls_passed'],False)
        rejects(lambda:original_probe_data(root));restore()
        checks['changed_original_test_data_and_failed_controls_refused']=True
        ctx=numerical_parent(root,name);files=ctx['files']|datafiles
        software=root/'outputs/gated/C011/software';software.mkdir(parents=True);(software/'review.json').write_bytes(b'synthetic software binding')
        pre=dict(name=name,research_sha256=rh,inputs_sha256={str(f):digest(f) for f in sorted(files)},data_sha256=data,
            software_review_sha256=digest(software/'review.json'),consulted_at='fixture',consequences='fixture only',
            sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')])
        pp=root/'pre.json';atomic_json(pp,pre);assert consultation(pp,root,name,files,data,rh)==pre
        for key,value in [('inputs_sha256',{}),('name','l1_n3_s2'),('data_sha256',{}),('sources',[{}])]:
            altered=copy.deepcopy(pre);altered[key]=value;atomic_json(pp,altered);rejects(lambda:consultation(pp,root,name,files,data,rh))
        checks['complete_input_consultation_and_four_stale_scope_refusals']=True
        assert previous_reviews(root,ORDER[0])==set();rejects(lambda:previous_reviews(root,ORDER[1]))
        sf=root/'outputs/gated/probes'/ORDER[0];sf.mkdir(parents=True);(sf/'cases.npz').write_bytes(b'complete synthetic cases')
        fake_review(sf,dict(research_id='C011',kind='l0',n=3,seed=1,train_windows=4096,test_windows=5000,test_episodes=1000,cases_sha256=digest(sf/'cases.npz')),'measured_negative')
        ops=root/'outputs/gated/C011/supervision'/ORDER[0]
        fake_review(ops,dict(research_id='C011',name=ORDER[0],returncode=0,child_terminal=True,signals=[],errors=[],stop_reason=None,
            inputs_unchanged=True,elapsed_seconds=20,child_artifact_sha256={p.name:digest(p) for p in sf.iterdir() if p.name!='review.json'}))
        assert previous_reviews(root,ORDER[1]);change(ops/'result.json',['elapsed_seconds'],3000)
        rejects(lambda:previous_reviews(root,ORDER[1]));checks['previous_complete_measured_negative_admitted_missing_or_overbudget_refused']=True
        lockdir=root/'outputs/gated/B003';lockdir.mkdir(parents=True)
        def acquire():
            with cpu_lock(root):pass
        with cpu_lock(root):rejects(acquire,BlockingIOError)
        acquire();checks['shared_CPU_planning_lock_refusal_and_release']=True
        normal=root/'normal';normal.mkdir();env=dict(os.environ)
        got=supervise([sys.executable,'-c','print("normal fixture")'],root,normal,env,dict(fixture=True),work=2,poll=.01)
        assert got['returncode']==0 and got['child_terminal'] and not got['errors'] and not got['signals']
        timed=root/'timed';timed.mkdir()
        got=supervise([sys.executable,'-c','import time;time.sleep(20)'],root,timed,env,dict(fixture=True),work=.04,term=.2,kill=.2,poll=.01)
        assert got['child_terminal'] and got['status']=='budget_incomplete' and got['signals'][0]['signal']=='SIGTERM' and not got['errors']
        from run_one import owned_module
        import psutil
        argv=[sys.executable,'-c','import time;time.sleep(20)'];child=subprocess.Popen(argv,cwd=root);proc=psutil.Process(child.pid);ct=proc.create_time();actions=[];owned=owned_module()
        try:
            rejects(lambda:owned.stop_owned(child,proc,ct+1,argv,root,actions,.2,.2));assert not actions and child.poll() is None
        finally:owned.stop_owned(child,proc,ct,argv,root,actions,.2,.2)
        checks['actual_owned_normal_deadline_and_wrong_identity_process_fixtures']=True

    original_files,original_hashes=original_probe_data(Path.cwd());assert len(original_hashes)==6
    checks['actual_original_n3_data_and_reviewed_controls_bound_without_inference']=True
    try:numerical_parent(Path.cwd(),ORDER[0])
    except FileNotFoundError as exc:refusal=str(exc)
    else:raise AssertionError('Expected missing actual additional-seed completion')
    assert 'outputs/gated/C001/l0_n3_s1/review.json' in refusal
    checks['current_actual_missing_parent_refusal_before_output']=refusal
    assert not torch.cuda.is_initialized(), 'Unexpected CUDA initialization'
    assert digest(ledger)==ledger_before, 'Shared GPU ledger changed'
    assert all(digest(p)==h for p,h in unchanged.items()), 'Existing helper changed'
    checks['CPU_only_no_existing_live_helper_or_shared_ledger_change']=True
    result=dict(research_id='C011',research_sha256=rh,status='awaiting_literature_review',checks=checks,
        source_sha256={p.name:digest(p) for p in BASE.glob('*.py')},repository_source_sha256=inherited['repository_source_sha256'],
        record_sha256={str(p):digest(p) for p in [Path('docs/research/records')/(n+'.json') for n in ['C001','C006','C011','A019']]},
        original_data_sha256=original_hashes,a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'),
        elapsed_seconds=time.monotonic()-started,ridge_relative_residual=ridge_residual,cuda_initialized=False,scientific_cases_executed=0,
        limitation='Synthetic parent/process/readout fixtures plus actual original data hashes. No additional-seed model inference, probe fit or scientific score has run; actual parent/model/numerical checks and result review remain required.')
    atomic_json(out/'result.json',result);print(json.dumps({k:result[k] for k in ['research_id','status','checks','elapsed_seconds','ridge_relative_residual','scientific_cases_executed']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C011',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)));raise

