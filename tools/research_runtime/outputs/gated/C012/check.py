"""Focused CPU checks for diagnostic scoring, feedback and resource admission."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from common import BASE, admit, close, inherited, inputs
from diagnostic import commands, compare, grouped
from go2nav.models.balls_slots import ObjectMaskedPredictor
from go2nav.research import atomic_json,digest,require_record


def reject(call, exception):
    try:call()
    except exception:return
    raise AssertionError('Expected refusal')


def run():
    start=time.monotonic();_,rh=require_record('C012');checks=[]
    a=torch.zeros(2,8,1,2);b=torch.zeros_like(a);a[:,6]=2e-5
    d=compare(a,b);assert not d['passed'] and d['failed_elements']==4
    assert [x['failed_elements'] for x in d['by_step']]==[0,0,0,0,0,0,4,0]
    expected=np.isclose(a.numpy(),b.numpy(),rtol=1e-4,atol=1e-5)
    assert int((~expected).sum())==d['failed_elements']
    assert compare(torch.ones_like(a)*1000.05,torch.ones_like(a)*1000)['passed']
    for v in [float('nan'),float('inf'),-float('inf')]:
        corrupt=a.clone();corrupt[0,0,0,0]=v;reject(lambda:compare(corrupt,b),FloatingPointError)
    reject(lambda:compare(a,b[:1]),ValueError)
    checks.append('Mixed absolute/relative tolerance, zero-reference failure counts and exact step attribution agree with NumPy; shape/nonfinite refusals')

    class Model:
        def __init__(self,mix=False):self.mix=mix
        def __call__(self,h,u):
            x=h[:,-1:]+u[:,-1:,0,None,None]
            if self.mix:x=x+x.mean(0,keepdim=True)
            return torch.cat((h,x),1),None
        def rollout(self,h,u,steps):
            values=[]
            for i in range(steps):
                x=self(h[:,-3:],u[:,i:i+3])[0][:,-1:];values.append(x);h=torch.cat((h,x),1)
            return torch.cat(values,1)
    h=torch.zeros(10,3,1,1);u=torch.arange(200).reshape(10,10,2).float()/100
    for mix in [False,True]:
        model=Model(mix);batched=grouped(model,h,u,'batched');rev=grouped(model,h,u,'reversed');pairs=grouped(model,h,u,'separate_pairs')
        assert compare(batched,rev)['passed']
        assert compare(batched,pairs)['passed'] == (not mix)
    checks.append('Grouping reverses sample order correctly and detects deliberate cross-candidate dependence')
    model=Model();fixed=torch.zeros(10,8,1,1)
    common=grouped(model,h,u,'batched',common=fixed)
    assert torch.equal(common,u[:,2:,0,None,None])
    assert not torch.equal(common,grouped(model,h,u,'batched'))
    checks.append('Common-history mode uses supplied states instead of recursively feeding variant outputs')

    torch.manual_seed(7);original=ObjectMaskedPredictor(dim=8,depth=1,heads=2,head_dim=4,mlp=16).eval()
    saved=copy.deepcopy(original.state_dict());promoted=copy.deepcopy(original).double()
    assert all(torch.equal(v,saved[k].double()) for k,v in promoted.state_dict().items())
    history=torch.randn(2,3,5,8);action=torch.randn(2,10,2)
    assert torch.isfinite(grouped(promoted,history.double(),action.double(),'batched')).all()
    assert all(torch.equal(v,saved[k]) for k,v in original.state_dict().items())
    checks.append('CPU fixture strict float64 promotion preserves learned values and leaves original model unchanged')

    labels=np.arange(2*100*2,dtype=np.float32).reshape(2,100,2)/100
    for split in ['train','val']:
        values=commands(labels,1,4,split)
        assert values.shape==(100,10,2) and np.array_equal(values[:,:2].numpy(),np.broadcast_to(labels[1,4:6]/3,(100,2,2)))
        assert float(torch.linalg.vector_norm(values[:,2:],dim=-1).max()) <= 1.000001
        assert torch.equal(values,commands(labels,1,4,split))
    assert not torch.equal(commands(labels,1,4,'train'),commands(labels,1,4,'val'))
    checks.append('Fixed train/val command cohorts preserve executed past, norm-capped future and deterministic split-specific seeds')

    class Child:
        def __init__(self,value):self.value=value
        def poll(self):return self.value
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);ledger=root/'ledger.json';ledger.write_text('[]')
        row,remaining,work=admit(ledger,root/'out');assert work==255 and row['research_id']=='C004' and row['supervisor']=='C012'
        reject(lambda:admit(ledger,root/'next'),ValueError)
        reject(lambda:close(ledger,row['id'],Child(None)),RuntimeError)
        reject(lambda:close(ledger,'wrong-id',Child(0)),ValueError)
        close(ledger,row['id'],Child(0));assert json.loads(ledger.read_text())[0]['end'] is not None
        reject(lambda:close(ledger,row['id'],Child(0)),ValueError)
        now=time.time();ledger.write_text(json.dumps([dict(id='spent',research_id='C004',supervisor='C012',out='x',start=now-28800,end=now)]))
        rem=inherited().remaining_seconds(json.loads(ledger.read_text()),now)
        assert rem['reserve_seconds']==0
        reject(lambda:admit(ledger,root/'next'),ValueError)
    checks.append('Single C004 row charges existing reserve; open/exhausted admission and live/wrong/already-closed interval closure refuse')

    bound,source=inputs();assert len(bound)>=20 and source
    assert not torch.cuda.is_initialized()
    checks.append('Actual exact parent/input inventory and failed-run computational source verified without CUDA initialization')
    return dict(research_id='C012',research_sha256=rh,status='awaiting_literature_review',checks=checks,check_groups=len(checks),
        source_sha256={n:digest(BASE/n) for n in ['common.py','diagnostic.py','run.py','check.py']},
        repository_source_sha256=source,input_sha256=bound,elapsed_seconds=time.monotonic()-start,cuda_initialized=False,
        limitation='CPU software fixtures and byte checks only; no actual GPU precision or accumulation finding and no scientific admission.')


if __name__=='__main__':
    out=BASE/'software';out.mkdir(exist_ok=False)
    try:result=run()
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C012',status='technical_blocked',error=repr(exc)))
        raise
    atomic_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','input_sha256']}),flush=True)
