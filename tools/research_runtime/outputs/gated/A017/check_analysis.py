"""CPU tests for the new A017 sampling, paired statistics and solver path."""
import ast
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record
from analysis import sample,paired_gain,action_controls,fit_linear,apply_linear,np,torch

require_record('A017');base=Path('outputs/gated/A017');out=base/'software';out.mkdir(exist_ok=False)
started=time.time();checks={}
try:
    for name in ['analysis.py','check_analysis.py','run_stage.py']:
        ast.parse((base/name).read_text())
    for split,count in [('train',1000),('val',100)]:
        e,t=sample(count,100,split);e2,t2=sample(count,100,split)
        assert np.array_equal(e,e2) and np.array_equal(t,t2) and t.min()>=2 and t.max()+4<100
        assert len(e)==(4096 if split=='train' else 500)
        if split=='val': assert np.array_equal(np.bincount(e),np.full(100,5))
    checks['deterministic_train_val_sampling_and_history_future_bounds']=True
    try:sample(1000,100,'test')
    except ValueError:checks['test_split_refused']=True
    else:raise AssertionError('test accepted')
    a=torch.arange(24,dtype=torch.float64).reshape(2,6,2);original=a.clone();controls=action_controls(a)
    assert torch.equal(a,original) and all(torch.equal(v[:,:2],a[:,:2]) for v in controls.values())
    assert torch.equal(controls['reversed'][:,2:],a[:,2:].flip(1)) and (controls['zero'][:,2:]==0).all()
    checks['action_controls_preserve_history_and_input']=True
    truth=np.zeros((3,4));baseline=np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4]],dtype=float)
    result=paired_gain(baseline,truth,truth,np.array([0,0,1]))
    assert result['red']['mean']==10 and result['white']['mean']==20
    reverse=paired_gain(truth,baseline,truth,np.array([0,0,1]))
    assert reverse['red']['mean']==-10 and reverse['white']['mean']==-20
    checks['paired_n1_gain_sign_and_equal_episode_weight']=True
    rng=np.random.default_rng(42);x=rng.normal(size=(257,8));y=x@rng.normal(size=(8,4))+rng.normal(size=4)
    decoder=fit_linear(torch.from_numpy(x),torch.from_numpy(y));mean,std,w=decoder
    z=np.column_stack(((x-mean.numpy())/std.numpy(),np.ones(len(x))))
    penalty=np.diag([.001]*8+[0.])
    augmented=np.row_stack((z/np.sqrt(len(x)),np.sqrt(penalty)))
    rhs=np.row_stack((y/np.sqrt(len(x)),np.zeros((9,4))))
    reference=np.linalg.lstsq(augmented,rhs,rcond=None)[0]
    np.testing.assert_allclose(w.numpy(),reference,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(apply_linear(torch.from_numpy(x),decoder).numpy(),z@reference,rtol=1e-10,atol=1e-10)
    checks['ridge_matches_independent_augmented_lstsq']=True
    assert not torch.cuda.is_initialized()
    atomic_json(out/'result.json',dict(research_id='A017',status='passed',checks=checks,elapsed_seconds=time.time()-started,
        source_sha256={n:digest(base/n) for n in ['analysis.py','check_analysis.py','run_stage.py']},cuda_initialized=False,
        scientific_cases_executed=0,limitation='Helper tests; real checkpoint numerical and probe validity require phase results.'))
    print(json.dumps(dict(status='passed',checks=len(checks))))
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='A017',status='technical_blocked',checks=checks,error=repr(exc)))
    raise
