"""C011: unchanged final C001 probes on CPU, with explicit numerical checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from go2nav.data.balls import load as load_data
from go2nav.eval.balls_probes import features, targets, fit_linear, apply_linear, probe_summary
from go2nav.research import atomic_json, digest, provenance, require_record, require_review, reviewed_artifact, verify_files
from go2nav.training.balls import load
from parents import context

BASE=Path('outputs/gated/C011')
DATA=Path('outputs/data/balls')
torch.set_num_threads(8)
torch.set_num_interop_threads(1)


def guard():
    if any(os.environ.get(k) != '' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']):
        raise RuntimeError('C011 requires hidden GPU devices')
    if torch.cuda.is_initialized():
        raise RuntimeError('C011 must never initialize CUDA')
    if psutil.Process().memory_info().rss > 16*2**30:
        raise MemoryError('C011 CPU16GiB RSS cap')


def finite(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite or non-CPU tensor')
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():raise ValueError('Nonfinite array')
    elif isinstance(value, dict):
        for v in value.values():finite(v)
    elif isinstance(value, (list,tuple)):
        for v in value:finite(v)
    elif isinstance(value, (int,float)) and not np.isfinite(value):
        raise ValueError('Nonfinite scalar')


def sample(count, frames, n, split):
    if split not in ['train','test'] or frames < 7:
        raise ValueError('C011 requires train/test and three-history/four-future windows')
    rng=np.random.default_rng([2026111511,n,split=='test'])
    episodes=rng.integers(count,size=4096) if split=='train' else np.repeat(np.arange(count),5)
    return episodes,rng.integers(2,frames-4,size=len(episodes))


def residual(x,y,decoder):
    mean,std,w=decoder
    design=torch.cat(((x.double()-mean)/std,torch.ones(len(x),1,dtype=torch.float64)),1)
    penalty=torch.eye(design.shape[1],dtype=torch.float64)*.001;penalty[-1,-1]=0
    a=design.T@design/len(x)+penalty;b=design.T@y.double()/len(x)
    return float(torch.linalg.vector_norm(a@w-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))


def run(name):
    guard();ctx=context(Path.cwd(),name)
    rh=ctx['research_sha256'];software=ctx['software'];consult=ctx['consult'];pre=ctx['pre']
    config=ctx['config'];checkpoint=ctx['checkpoint'];parent=ctx['parent'];evidence=ctx['evidence']
    n,kind,seed=config['n'],config['kind'],config['seed'];inputs=sorted(ctx['files'])+[pre]
    out=Path('outputs/gated/probes')/name;out.mkdir(parents=True,exist_ok=False)
    manifest=provenance('C011',dict(name=name,n=n,kind=kind,seed=seed,device='cpu',dtype='float64',train_windows=4096,test_windows=5000),inputs)
    manifest.update(analysis_source_sha256=software['source_sha256'],software_review_sha256=digest(BASE/'software/review.json'))
    atomic_json(out/'manifest.json',manifest)
    started=time.monotonic();result=dict(research_id='C011',n=n,kind=kind,seed=seed,device='cpu',status='awaiting_literature_review')
    try:
        with sdpa_kernel(SDPBackend.MATH):
            model,head,state=load(checkpoint,'cpu')
            assert state['n']==n and state['kind']==kind and state['step']==303000
            assert state['inference_only'] and state['inference_dtype']=='float64' and state['attention_backend']=='math'
            assert state['train']==config['train']
            finite(model.state_dict())
            if head is not None:finite(head.state_dict())
            frozen={k:v.clone() for k,v in model.state_dict().items()}
            dataset={};indices={}
            for split in ['train','test']:
                guard();frames,labels=load_data(DATA,n,split)
                assert len(frames)==1000 and frames.shape[1]==100
                episodes,starts=sample(len(frames),frames.shape[1],n,split)
                indices[split]=(episodes,starts)
                print(json.dumps(dict(phase='features',split=split,windows=len(episodes))),flush=True)
                x=features(model,frames,labels['actions'],episodes,starts,batch=32)
                finite(x)
                y=targets(labels['red'][episodes,starts],labels['white'][episodes,starts])
                future=targets(labels['red'][episodes,starts+4],labels['white'][episodes,starts+4])
                dataset[split]=(x,y,future,episodes)
                if split=='test':
                    single=features(model,frames,labels['actions'],episodes[:8],starts[:8],batch=1)
                    reverse=features(model,frames,labels['actions'],episodes[:8][::-1].copy(),starts[:8][::-1].copy(),batch=8)
                    grouping={}
                    for key,value in x.items():
                        torch.testing.assert_close(value[:8],single[key],rtol=1e-4,atol=1e-5)
                        torch.testing.assert_close(value[:8],reverse[key].flip(0),rtol=1e-4,atol=1e-5)
                        grouping[key]=dict(single_max_abs=float((value[:8]-single[key]).abs().max()),reversed_max_abs=float((value[:8]-reverse[key].flip(0)).abs().max()))
            guard();train,y,_,_=dataset['train'];decoder=fit_linear(train['current'],torch.from_numpy(y));finite(decoder)
            error=residual(train['current'],torch.from_numpy(y),decoder);assert error<=1e-8
            torch.save(dict(mean=decoder[0],std=decoder[1],weights=decoder[2],research_id='C011',checkpoint_sha256=digest(checkpoint)),out/'probe.pt')
            x,current,future,episodes=dataset['test'];report={};predictions={}
            for key,value in x.items():
                guard();predicted=apply_linear(value,decoder).numpy();finite(predicted)
                truth=current if key=='current' else future
                report[key]=probe_summary(predicted,truth,episodes);finite(report[key]);predictions[key]=predicted
                for label,sl,count in [('red',slice(0,2),1),('white',slice(2,None),n)]:
                    report[key][label+'_mean_pearson']=float(np.mean(report[key]['pearson'][sl]))
                    report[key][label+'_position_rmse_px']=float(np.sqrt(np.mean(((predicted[:,sl]-truth[:,sl])**2).reshape(len(truth),count,2).sum(-1))))
                print(json.dumps(dict(phase='summary',mode=key)),flush=True)
            np.savez_compressed(out/'cases.npz',episodes=episodes,starts=indices['test'][1],current_truth=current,future_truth=future,**predictions)
            np.savez_compressed(out/'features.npz',train_episodes=indices['train'][0],train_starts=indices['train'][1],train_current=train['current'].numpy(),train_truth=y,**{f'test_{k}':v.numpy() for k,v in x.items()})
            assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
            assert digest(checkpoint)==evidence['calibrated_sha256']
            verify_files(Path.cwd(),consult['inputs_sha256']);verify_files(Path.cwd(),software['repository_source_sha256']);guard()
            result.update(measures=report,checkpoint_sha256=digest(checkpoint),parent_review_sha256=digest(parent/'review.json'),
                cases_sha256=digest(out/'cases.npz'),features_sha256=digest(out/'features.npz'),probe_sha256=digest(out/'probe.pt'),
                train_windows=4096,test_windows=5000,test_episodes=1000,bootstrap_repeats=2000,numerical_grouping=grouping,
                ridge_relative_residual=error,weights_unchanged=True,cuda_initialized=False,manifest_sha256=digest(out/'manifest.json'),elapsed_seconds=time.monotonic()-started)
            atomic_json(out/'result.json',result);print(json.dumps(dict(name=name,status=result['status'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as exc:
        result.update(status='technical_blocked',error=repr(exc),elapsed_seconds=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
        atomic_json(out/'result.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
