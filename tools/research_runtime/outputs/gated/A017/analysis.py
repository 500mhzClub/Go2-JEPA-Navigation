"""A017 CPU reference: two immutable n1 L1 checkpoints, calibration then probes."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
assert os.environ.get('HIP_VISIBLE_DEVICES') == os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import psutil
from scipy.stats import spearmanr
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from go2nav.data.balls import load as load_data
from go2nav.eval.balls_inference import batch_modes, calibrate, invariance, outputs, windows
from go2nav.eval.balls_interim import action_controls, encode
from go2nav.eval.balls_probes import apply_linear, fit_linear, probe_summary, targets
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files
from go2nav.training.balls import load, schedule, to_pixels

BASE = Path('outputs/gated/A017')
DATA = Path('outputs/data/balls')
torch.set_num_threads(8)
torch.set_num_interop_threads(1)


def sample(count, frames, split):
    if split not in ('train','val'):
        raise ValueError('A017 forbids test data')
    rng = np.random.default_rng([2026111520,1,split=='val'])
    e = rng.integers(count,size=4096) if split=='train' else np.repeat(np.arange(count),5)
    return e,rng.integers(2,frames-4,size=len(e))


def paired_gain(baseline, predicted, truth, episodes):
    delta = ((baseline-truth)**2-(predicted-truth)**2).reshape(len(truth),-1,2).sum(-1)
    by_episode = np.stack([delta[episodes==e].mean(0) for e in np.unique(episodes)])
    rng = np.random.default_rng(2026111520)
    draws = by_episode[rng.integers(len(by_episode),size=(2000,len(by_episode)))].mean(1)
    return {name:dict(mean=float(by_episode[:,idx].mean()),ci95=np.quantile(draws[:,idx].mean(-1),[.025,.975]).tolist())
            for name,idx in [('red',[0]),('white',list(range(1,delta.shape[1]))) ]}


def guard():
    assert not torch.cuda.is_initialized()
    if psutil.Process().memory_info().rss > 16*2**30:
        raise MemoryError('A01716GB RSS cap')


def finite(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=='cpu' and torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values(): finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value: finite(v)
    elif isinstance(value,(int,float)):
        assert np.isfinite(value)


def assert_invariance(result):
    assert all(v['passed'] for mode in result.values() for v in mode.values())


def gradient_check(model,p,a):
    model.zero_grad(set_to_none=True)
    z,pred=outputs(model,p[:4],a[:4]);loss=(pred-z[:,1:]).square().mean();loss.backward()
    grads={n:v.grad for n,v in model.named_parameters() if v.grad is not None}
    assert grads and all(torch.isfinite(v).all() for v in grads.values())
    count=len(grads);model.zero_grad(set_to_none=True)
    return dict(finite=True,populated_parameter_gradients=count,reference_loss=float(loss.detach()))


def calibration_phase(manifest,out):
    report={}
    initial=torch.load(BASE/'snapshots/before/checkpoint.pt',map_location='cpu',weights_only=False)
    for name in ['before','current']:
        guard();folder=BASE/'snapshots'/name;checkpoint=folder/'checkpoint.pt'
        model,head,state=load(checkpoint,'cpu');model.double();finite(state)
        assert state['kind']=='l1' and state['n']==1
        cfg=json.loads((folder/'config.json').read_text())
        assert json.dumps(state['config'],sort_keys=True)==json.dumps(cfg['model'],sort_keys=True)
        assert state['train']==cfg['train'] and state['scheduler']['last_epoch']==state['step']
        assert state['optimiser']['param_groups'][0]['params']==initial['optimiser']['param_groups'][0]['params']
        assert state['optimiser']['state'].keys()==initial['optimiser']['state'].keys()
        counters=[]
        for key,values in state['optimiser']['state'].items():
            old=initial['optimiser']['state'][key]['step'].item();new=values['step'].item()
            assert new==old+state['step']-initial['step'];counters.append(new)
        expected_lr=cfg['train']['lr']*schedule(cfg['train']['warmup'],cfg['total_steps'])(state['step'])
        assert state['scheduler']['_last_lr']==[expected_lr] and state['optimiser']['param_groups'][0]['lr']==expected_lr
        weights={k:v.detach().clone() for k,v in model.named_parameters()}
        result=dict(step=state['step'],checkpoint_sha256=digest(checkpoint),optimizer_min_step=min(counters),
                    optimizer_max_step=max(counters),finite_full_state=True,before={},after={})
        for split in ['train','val']:
            p,a=next(windows(DATA,1,split,64,42,device='cpu'))
            result['before'][split]=batch_modes(model,p,a)
        result['before']['invariance']=invariance(model,p[:8],a[:8]);assert_invariance(result['before']['invariance'])
        result['before']['gradients']=gradient_check(model,p,a)
        result['calibration']=calibrate(model,lambda:windows(DATA,1,'train',4096,17,64,device='cpu'))
        for split in ['train','val']:
            p,a=next(windows(DATA,1,split,64,42,device='cpu'))
            result['after'][split]=batch_modes(model,p,a)
        result['after']['invariance']=invariance(model,p[:8],a[:8]);assert_invariance(result['after']['invariance'])
        result['after']['gradients']=gradient_check(model,p,a)
        assert all(torch.equal(v,weights[k]) for k,v in model.named_parameters())
        derived={k:state[k] for k in ['config','kind','n','train','head_state_dict','step']}
        derived.update(model_state_dict=model.state_dict(),inference_only=True,inference_dtype='float64',
                       attention_backend='math',research_id='A017',source_sha256=digest(checkpoint))
        target=out/(name+'_calibrated.pt');torch.save(derived,target)
        result.update(calibrated_sha256=digest(target),weights_unchanged=True,cuda_initialized=False)
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line]
        finite(rows);train=[r for r in rows if 'validation' not in r]
        val=[r for r in rows if 'validation' in r];lastval=val[-1]
        recent=[r['pred'] for r in train if lastval['step']-5000<=r['step']<=lastval['step']]
        result['curves']=dict(training_rows=len(train),validation_rows=len(val),last_train=train[-1],
                            last_validation=lastval,recent_train_pred_median=float(np.median(recent)),
                            raw_validation_train_pred_ratio=float(lastval['validation']['pred']/np.median(recent)))
        report[name]=result
        print(json.dumps(dict(case=name,phase='calibration_finished',step=state['step'])),flush=True)
    return dict(cases=report,meaning='CPU reference validity and train-only calibration; no GPU parity or final gate',cuda_initialized=False)


@torch.no_grad()
def future_encoded(model,frames,e,t,batch=32):
    values=[]
    for i in range(0,len(e),batch):
        p=to_pixels(torch.from_numpy(np.array(frames[e[i:i+batch,None],t[i:i+batch,None]+4]))).double()
        values.append(model.encode(p)[:,0])
    return torch.cat(values)


def geometry(features,truth,episodes):
    rng=np.random.default_rng(2026111521);left=[];right=[]
    while len(left)<2000:
        i,j=rng.integers(len(episodes),size=2)
        if episodes[i]!=episodes[j]:left.append(i);right.append(j)
    x=features.numpy();latent=np.linalg.norm(x[left]-x[right],axis=-1)
    red=np.linalg.norm(truth[left,:2]-truth[right,:2],axis=-1)
    white=np.linalg.norm(truth[left,2:]-truth[right,2:],axis=-1)
    return dict(red_spearman=float(spearmanr(latent,red).statistic),white_spearman=float(spearmanr(latent,white).statistic),
                latent_coordinate_sd_mean=float(x.std(0,ddof=1).mean()),latent_pair_l2_mean=float(latent.mean())),dict(
                pair_left=np.array(left),pair_right=np.array(right),latent_distance=latent,red_distance=red,white_distance=white)


def probe_phase(manifest,out):
    assert require_review(BASE/'calibration/review.json')['status']=='passed'
    calibration=json.loads((BASE/'calibration/result.json').read_text())
    train_frames,train_labels=load_data(DATA,1,'train');frames,labels=load_data(DATA,1,'val')
    te,tt=sample(len(train_frames),train_frames.shape[1],'train');ve,vt=sample(len(frames),frames.shape[1],'val')
    y=targets(train_labels['red'][te,tt],train_labels['white'][te,tt])
    truth=targets(labels['red'][ve,vt],labels['white'][ve,vt]);future=targets(labels['red'][ve,vt+4],labels['white'][ve,vt+4])
    report={};all_predictions={}
    for name in ['before','current']:
        guard();checkpoint=BASE/'calibration'/(name+'_calibrated.pt')
        assert digest(checkpoint)==calibration['cases'][name]['calibrated_sha256']
        model,_,state=load(checkpoint,'cpu');frozen={k:v.clone() for k,v in model.state_dict().items()}
        train=encode(model,train_frames,train_labels['actions'],te,tt,future=False)['current']
        decoder=fit_linear(train,torch.from_numpy(y))
        design=torch.cat(((train-decoder[0])/decoder[1],torch.ones(len(train),1,dtype=torch.float64)),-1)
        penalty=torch.eye(design.shape[1],dtype=torch.float64)*.001;penalty[-1,-1]=0
        a=design.T@design/len(train)+penalty;b=design.T@torch.from_numpy(y).double()/len(train)
        residual=float(torch.linalg.vector_norm(a@decoder[2]-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
        assert residual<=1e-8;finite(decoder)
        x=encode(model,frames,labels['actions'],ve,vt,future=True)
        x['future_encoded']=future_encoded(model,frames,ve,vt)
        single=encode(model,frames,labels['actions'],ve[:8],vt[:8],future=True,batch=1)
        reverse=encode(model,frames,labels['actions'],ve[:8][::-1].copy(),vt[:8][::-1].copy(),future=True,batch=8)
        single['future_encoded']=future_encoded(model,frames,ve[:8],vt[:8],batch=1)
        reverse['future_encoded']=future_encoded(model,frames,ve[:8][::-1].copy(),vt[:8][::-1].copy(),batch=8)
        grouping={}
        for key,value in x.items():
            torch.testing.assert_close(value[:8],single[key],rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(value[:8],reverse[key].flip(0),rtol=1e-4,atol=1e-5)
            grouping[key]=dict(single_max_abs=float((value[:8]-single[key]).abs().max()),
                               reversed_max_abs=float((value[:8]-reverse[key].flip(0)).abs().max()))
        predictions={k:apply_linear(v,decoder).numpy() for k,v in x.items()}
        measures={}
        for key,pred in predictions.items():
            assert np.isfinite(pred).all();target=truth if key=='current' else future
            measures[key]=probe_summary(pred,target,ve)
            for label,idx in [('red',slice(0,2)),('white',slice(2,4))]:
                measures[key][label+'_mean_pearson']=float(np.mean(measures[key]['pearson'][idx]))
                measures[key][label+'_position_rmse_px']=float(np.sqrt(np.mean(((pred[:,idx]-target[:,idx])**2).sum(-1))))
        gains={k:paired_gain(predictions[k],predictions['rollout'],future,ve) for k in ['no_rollout','zero','reversed']}
        geom,pairs=geometry(x['current'],truth,ve)
        path=out/(name+'_cases.npz');np.savez_compressed(path,train_episodes=te,train_starts=tt,train_features=train.numpy(),
            train_truth=y,episodes=ve,starts=vt,current_truth=truth,future_truth=future,**predictions,**pairs)
        torch.save(dict(mean=decoder[0],std=decoder[1],weights=decoder[2]),out/(name+'_probe.pt'))
        assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
        assert digest(checkpoint)==calibration['cases'][name]['calibrated_sha256']
        report[name]=dict(step=state['step'],measures=measures,paired_control_gains=gains,geometry=geom,
                         numerical_grouping=grouping,ridge_relative_residual=residual,weights_unchanged=True,
                         cases_sha256=digest(path),calibrated_sha256=digest(checkpoint))
        all_predictions[name]=predictions
        print(json.dumps(dict(case=name,phase='probes_finished',step=state['step'])),flush=True)
    temporal={k:paired_gain(all_predictions['before'][k],all_predictions['current'][k],truth if k=='current' else future,ve)
              for k in all_predictions['before']}
    return dict(cases=report,paired_temporal_gains=temporal,train_windows=4096,validation_windows=500,
                validation_episodes=100,bootstrap_repeats=2000,cuda_initialized=False,
                meaning='Single-seed temporal CPU reference, not final test/planning or GPU validity')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['calibration','probes']);args=parser.parse_args()
    _,rh=require_record('A017');manifest=json.loads((BASE/'snapshots/manifest.json').read_text())
    verify_files(Path.cwd(),manifest['source_sha256']);verify_files(Path.cwd(),manifest['data_sha256'])
    verify_files(BASE/'snapshots',manifest['snapshot_sha256'])
    assert require_review(BASE/'software/review.json')['status']=='passed'
    software=json.loads((BASE/'software/result.json').read_text());verify_files(BASE,software['source_sha256'])
    out=BASE/args.phase;out.mkdir(exist_ok=False);started=time.time()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            result=(calibration_phase if args.phase=='calibration' else probe_phase)(manifest,out)
        guard();verify_files(BASE/'snapshots',manifest['snapshot_sha256']);verify_files(Path.cwd(),manifest['data_sha256'])
        result.update(research_id='A017',phase=args.phase,status='awaiting_literature_review',research_sha256=rh,
                      elapsed_seconds=time.time()-started,snapshot_manifest_sha256=digest(BASE/'snapshots/manifest.json'),
                      software_review_sha256=digest(BASE/'software/review.json'),cpu_threads=torch.get_num_threads())
        atomic_json(out/'result.json',result)
        print(json.dumps(dict(phase=args.phase,status=result['status'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='A017',phase=args.phase,status='technical_blocked',error=repr(exc),
                    elapsed_seconds=time.time()-started,snapshot_manifest_sha256=digest(BASE/'snapshots/manifest.json')))
        raise


if __name__=='__main__':main()
