"""Focused CPU checks for C013 policy boundaries, evidence completeness and supervision."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from torch import nn
import common
from policy import POLICIES,PrecisionWorld,choose,load_policy
from validation import metric,comparison,pipeline_checks,qualify,commands_for,unchanged,encode_frames
from go2nav.models.balls_slots import SlotWorld
from go2nav.eval.balls_probes import features
from go2nav.eval.balls_planning import Inputs,learned_actions
from go2nav.research import atomic_json,digest,require_record
from go2nav.eval.slots_checkpoint import windows


class Encoder(nn.Module):
    def __init__(self,coupled=False):
        super().__init__();self.scale=nn.Parameter(torch.arange(1,5,dtype=torch.float32));self.calls=[];self.coupled=coupled
    def forward(self,pixels):
        self.calls.append(tuple(pixels.shape))
        values=pixels.mean((-3,-2,-1))[...,None,None]*self.scale[None,None,None,:]
        values=values.expand(-1,-1,5,-1).clone()
        if self.coupled:values=values+values.mean(0,keepdim=True)*.25
        return values


class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.ones(4))
    def rollout(self,h,a,steps):
        result=[];z=h[:,-1:]
        for j in range(steps):
            z=z+a[:,j+2:j+3].sum(-1)[...,None,None]*self.scale
            result.append(z)
        return torch.cat(result,1)


def refuse(call,types=(ValueError,RuntimeError,FloatingPointError)):
    try:call()
    except types:return
    raise AssertionError('Expected refusal')


def fixture():
    ok=dict(passed=True)
    def numerical(names,n):
        return dict(outputs={k:ok.copy() for k in names},gradients={str(k):ok.copy() for k in range(n)},
                    grouping={k:{m:ok.copy() for m in ('repeat','reversed','separate')} for k in names},
                    missing_gradients={'cpu':[],'cuda:0':[]},passed=True)
    row=dict(episodes=[0,1],starts=[2,3],encoder=numerical(['slots','reconstruction','masks'],57),predictor=numerical(['predicted'],83),
        pipeline=dict(checks={m:{k:ok.copy() for k in ['history','goal','predicted','cost']} for m in ['cpu','repeat','reversed','separate']}))
    plan=dict(encoding={k:ok.copy() for k in ['history','goal']},checks={m:dict(cost=ok.copy(),rollout=dict(passed=True,by_step=[ok.copy() for _ in range(8)])) for m in ['cpu','repeat','reversed','separate']})
    return dict(complete=True,windows={s:[copy.deepcopy(row) for _ in range(8)] for s in ['train','val']},planning={s:copy.deepcopy(plan) for s in ['train','val']})


def main():
    start=time.monotonic();checks=[];record,rh=require_record('C013');torch.set_num_threads(2)
    torch.manual_seed(2026111516)
    world=SlotWorld(Encoder(),Predictor());original={k:v.clone() for k,v in world.state_dict().items()}
    pixels=torch.randn(2,3,3,8,8);goal=torch.randn(2,1,3,8,8);actions=torch.randn(2,6,2)
    for policy in POLICIES:
        p=PrecisionWorld(world,policy)
        assert next(p.encoder.parameters()).dtype==(torch.float32 if policy==POLICIES[0] else torch.float64)
        assert p.encode(pixels).dtype==torch.float64 and p.rollout(p.encode(pixels),actions,4).dtype==torch.float64
        assert unchanged(p,original) and unchanged(world,original)
    refuse(lambda:PrecisionWorld(world,'unknown'))
    checks.append('Both explicit policies preserve original values and expose float64 latents/rollout; unknown policy refuses')
    a=torch.tensor([0.,1.,2.]);b=a.clone();b[0]=2e-5
    assert not metric(a,b)['passed'] and metric(a,b)['failed_elements']==1
    assert np.allclose(a.numpy(),b.numpy(),rtol=1e-4,atol=1e-5)==metric(a,b)['passed']
    refuse(lambda:metric(a,torch.tensor([float('nan'),1.,2.])))
    refuse(lambda:metric(a,b[:2]))
    v=torch.zeros(2,8,5,4);w=v.clone();w[0,6,0,0]=2e-5
    assert [x['failed_elements'] for x in comparison(v,w,True)['by_step']]==[0,0,0,0,0,0,1,0]
    checks.append('Mixed tolerance, zero-reference failure, per-step attribution and nonfinite/shape refusals')
    raw=torch.randint(0,256,(2,3,8,8,3),dtype=torch.uint8);raw_goal=torch.randint(0,256,(2,1,8,8,3),dtype=torch.uint8)
    for policy in POLICIES:
        p=PrecisionWorld(world,policy);out=pipeline_checks(p,raw,raw_goal,actions[:,:3],lambda:None);assert out['passed']
        assert (2,3,3,8,8) in p.encoder.calls and (2,1,3,8,8) in p.encoder.calls
        bad=PrecisionWorld(SlotWorld(Encoder(True),Predictor()),policy)
        assert not pipeline_checks(bad,raw,raw_goal,actions[:,:3],lambda:None)['passed']
        refuse(lambda:encode_frames(p,raw.float()))
        from go2nav.training.balls import to_pixels
        with patch('validation.to_pixels',wraps=to_pixels) as preprocessing:
            encode_frames(p,raw)
            assert preprocessing.call_count==1 and preprocessing.call_args.args[0].dtype==torch.uint8
            assert preprocessing.call_args.args[0].device==next(p.parameters()).device
    checks.append('Complete raw-image pipeline grouping catches cross-example dependence; independent history/goal calls normalize uint8 after transfer to model device')
    good=fixture();assert qualify(good)
    bad=copy.deepcopy(good);bad['windows']['train'][0]['encoder']['missing_gradients']['cpu']=['scale'];assert not qualify(bad)
    bad=copy.deepcopy(good);bad['planning']['val']['checks']['cpu']['rollout']['passed']=False;assert not qualify(bad)
    bad=copy.deepcopy(good);bad['windows']['val'].pop();refuse(lambda:qualify(bad))
    bad=copy.deepcopy(good);bad['windows']['val'][0]['predictor']['gradients'].pop('0');refuse(lambda:qualify(bad))
    assert choose({p:dict(complete=True,checks_passed=True) for p in POLICIES})==POLICIES[0]
    assert choose({p:dict(complete=True,checks_passed=p==POLICIES[1]) for p in POLICIES})==POLICIES[1]
    assert choose({p:dict(complete=True,checks_passed=False) for p in POLICIES}) is None
    refuse(lambda:choose({POLICIES[0]:dict(complete=True,checks_passed=True)}))
    checks.append('Policy qualification rejects missing gradients/coverage and failed rollouts; fixed preference requires both complete outcomes')
    frames=np.random.default_rng(11).integers(0,256,(2,10,8,8,3),dtype=np.uint8)
    acts=np.random.default_rng(12).normal(size=(2,10,2)).astype(np.float32)
    for policy in POLICIES:
        p=PrecisionWorld(world,policy);f=features(p,frames,acts,np.array([0,1]),np.array([2,3]),batch=2)
        assert set(f)=={'current','rollout','shuffled','no_rollout'} and all(x.shape==(2,20) and x.dtype==torch.float64 for x in f.values())
        inp=Inputs(frames[0,:3],acts[0,:2],frames[0,6]);plan=learned_actions(p,inp,4,2026111504)
        assert plan.shape==(4,2) and np.isfinite(plan).all() and np.linalg.norm(plan,axis=-1).max()<=3.00001
        assert np.array_equal(plan,learned_actions(p,inp,4,2026111504))
    checks.append('Unchanged probe features and full default100candidate/20iteration CEM caller execute both policies with finite repeatable plans on CPU fixtures')
    for dtype in [torch.float32,torch.float64]:
        u=commands_for(acts,0,0,'train',dtype)
        assert u.dtype==dtype and u.shape==(100,10,2) and torch.equal(u[0,:2],torch.from_numpy(acts[0,:2]).to(dtype)/3)
        assert torch.linalg.vector_norm(u[:,2:],dim=-1).max()<1.000001
        assert torch.equal(u,commands_for(acts,0,0,'train',dtype))
    checks.append('Physical candidate cohort and executed past commands are deterministic; action normalization follows deployed encoder dtype')
    with tempfile.TemporaryDirectory() as tmp:
        ledger=Path(tmp)/'ledger.json';atomic_json(ledger,[])
        row,remaining,work=common.admit(ledger,Path(tmp)/'out');assert work==1755 and row['supervisor']=='C013' and row['research_id']=='C004'
        refuse(lambda:common.admit(ledger,Path(tmp)/'other'))
        class Live:
            def poll(self):return None
        refuse(lambda:common.close(ledger,row['id'],Live()))
        refuse(lambda:common.close(ledger,'wrong',None))
        common.close(ledger,row['id'],None);refuse(lambda:common.close(ledger,row['id'],None))
        atomic_json(ledger,[dict(research_id='C004',start=0,end=8*3600)])
        refuse(lambda:common.admit(ledger,Path(tmp)/'exhausted'))
    checks.append('Single existing-reserve allocation, open/exhausted admission and live/wrong/double-close refusals')
    bound,source=common.inputs()
    for policy in POLICIES:
        p=load_policy('outputs/gated/C001/o0_n3_s0/model.pt',policy)
        assert len(list(p.encoder.parameters()))==57 and len(list(p.predictor.parameters()))==83
        assert all(b.dtype==next(p.encoder.parameters()).dtype for b in p.encoder.buffers() if b.is_floating_point())
        if policy==POLICIES[0]:original={k:v.clone() for k,v in p.state_dict().items()}
        else:assert unchanged(p,original)
        for split in ['train','val']:
            real=np.load(f'outputs/data/balls/n3/{split}/frames.npy',mmap_mode='r')
            episodes,starts=windows(real.shape,3,split);e,t=episodes[:2,None],starts[:2,None]
            raw=torch.from_numpy(np.array(real[e,t+np.arange(3)]));goal=torch.from_numpy(np.array(real[e,t+3]))
            assert raw.shape==(2,3,64,64,3) and goal.shape==(2,1,64,64,3)
            with torch.no_grad():
                assert encode_frames(p,goal).shape==(2,1,5,128)
            refuse(lambda:encode_frames(p,goal[:,None]))
            with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as f:commands=torch.from_numpy(np.array(f['actions'][e,t+np.arange(3)]))
            # Actual indexed pipeline executes all grouping routes on CPU; this is a software fixture, not CPU/GPU parity.
            assert pipeline_checks(p,raw,goal,commands,lambda:None)['passed']
    assert not torch.cuda.is_initialized()
    checks.append('Actual reviewed C012/C004/training/data/cache lineage, exact real-model precision promotion including buffers, no CUDA')
    checks.append('Actual first train/val paired-index extraction and complete CPU pipeline execute both policies; redundant goal dimension rejects')
    out=common.BASE/'software';out.mkdir(exist_ok=False)
    result=dict(research_id='C013',research_sha256=rh,status='awaiting_literature_review',checks=checks,check_groups=len(checks),
        source_sha256={n:digest(common.BASE/n) for n in ['common.py','policy.py','validation.py','run.py','check.py']},
        repository_source_sha256=source,input_count=len(bound),elapsed_seconds=time.monotonic()-start,cuda_initialized=False)
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='repository_source_sha256'}))


if __name__=='__main__':main()
