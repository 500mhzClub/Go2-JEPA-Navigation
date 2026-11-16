"""Frozen O0 precision/feedback diagnostic; never an inference-admission pass."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from common import BASE, NAME, inputs
from go2nav.eval.balls_planning import cap
from go2nav.eval.slots_checkpoint import windows
from go2nav.models.balls_slots import ObjectMaskedPredictor
from go2nav.research import atomic_json, dedicated_device, digest, provenance, require_record, verify_files
from go2nav.training.balls import to_pixels
from go2nav.training.balls_slots import load_encoder


def compare(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point():
        raise ValueError('Invalid comparison shape/dtype')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite diagnostic output')
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    def summary(x, y):
        diff = (x-y).abs()
        failed = ~torch.isclose(x, y, rtol=1e-4, atol=1e-5)
        ratio = (x.double()-y.double()).abs()/(1e-5+1e-4*y.double().abs())
        return dict(passed=bool(torch.allclose(x,y,rtol=1e-4,atol=1e-5)),
                    failed_elements=int(failed.sum()), max_abs=float(diff.max()),
                    rms=float(diff.double().square().mean().sqrt()), max_tolerance_ratio=float(ratio.max()))
    return dict(**summary(a,b), by_step=[summary(a[:,i],b[:,i]) for i in range(a.shape[1])])


@torch.no_grad()
def grouped(model, history, actions, mode, check=lambda: None, common=None):
    check()
    if mode == 'reversed':
        h,a = history.flip(0),actions.flip(0)
        c = None if common is None else common.flip(0)
    else:
        h,a,c = history,actions,common
    size = 2 if mode == 'separate_pairs' else len(h)
    parts=[]
    for start in range(0,len(h),size):
        check(); x,u=h[start:start+size],a[start:start+size]
        if c is None:
            value=model.rollout(x,u,8)
        else:
            sequence=torch.cat((x,c[start:start+size]),1)
            values=[]
            for step in range(8):
                check()
                values.append(model(sequence[:,step:step+3],u[:,step:step+3])[0][:,-1:])
            value=torch.cat(values,1)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite rollout')
        parts.append(value.cpu())
    value=torch.cat(parts)
    return value.flip(0) if mode == 'reversed' else value


def commands(actions, episode, start, split):
    rng=np.random.default_rng((2026111516,3,('train','val').index(split),100))
    future=cap((2*rng.standard_normal((100,8,2))).astype(np.float32))
    past=np.broadcast_to(actions[episode,start:start+2],(100,2,2))
    return torch.from_numpy(np.concatenate((past,future),1))/3


def run(out, work):
    start=time.monotonic();record,rh=require_record('C012');bound,source=inputs()
    out.mkdir(exist_ok=False)
    manifest=provenance('C012',record['protocol'],bound)
    manifest['helper_sha256']={n:digest(BASE/n) for n in ['common.py','diagnostic.py','run.py','check.py']}
    atomic_json(out/'manifest.json',manifest)
    result=dict(research_id='C012',name=NAME,research_sha256=rh,status='awaiting_literature_review',splits={})
    def check():
        if time.monotonic()-start >= work:
            raise TimeoutError('C012 work deadline')
    try:
        dedicated_device(.5)
        torch.cuda.set_per_process_memory_fraction(.5)
        state=torch.load('outputs/gated/C001/o0_n3_s0/model.pt',map_location='cpu',weights_only=False)
        assert (state['kind'],state['n'],state['seed']) == ('o0',3,0)
        original=ObjectMaskedPredictor(**state['config']).eval()
        original.load_state_dict(state['model'],strict=True)
        encoder,es=load_encoder('outputs/gated/C001/slots_n3_s0/model.pt')
        flags=dict(cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=torch.backends.cudnn.benchmark,
                   cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_precision=torch.get_float32_matmul_precision(),
                   attention='MATH forced in original model and diagnostic context',torch=torch.__version__,hip=torch.version.hip)
        result['backend_flags']=flags
        old=json.loads(Path('outputs/gated/C004/o0_n3_s0/result.json').read_text())
        for split in ['train','val']:
            check();frames=np.load(f'outputs/data/balls/n3/{split}/frames.npy',mmap_mode='r')
            with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as labels:actions=labels['actions']
            episodes,starts=windows(frames.shape,3,split);e,t=int(episodes[0]),int(starts[0])
            assert (e,t)==(old['windows'][split][0]['episodes'][0],old['windows'][split][0]['starts'][0])
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                pixels=to_pixels(torch.from_numpy(np.array(frames[e:e+1,t:t+3])).cuda())
                history=encoder(pixels).expand(100,-1,-1,-1).cpu()
            action=commands(actions,e,t,split)
            arrays=dict(history=history.numpy(),actions=action.numpy());reports={};times={}
            cpu=copy.deepcopy(original).double().eval();before=time.monotonic()
            with sdpa_kernel(SDPBackend.MATH):
                reference=grouped(cpu,history.double(),action.double(),'batched',check)
            times['cpu_float64_reference']=time.monotonic()-before
            arrays['cpu64_batched']=reference.numpy();del cpu
            for label,dtype in [('gpu32',torch.float32),('gpu64',torch.float64)]:
                check();model=copy.deepcopy(original).to(device='cuda:0',dtype=dtype).eval()
                assert all(torch.equal(v.cpu(),state['model'][k].to(dtype)) for k,v in model.state_dict().items())
                h,a=history.to(device='cuda:0',dtype=dtype),action.to(device='cuda:0',dtype=dtype)
                outputs={};before=time.monotonic()
                with sdpa_kernel(SDPBackend.MATH):
                    for mode in ['batched','repeat','reversed','separate_pairs']:
                        outputs[mode]=grouped(model,h,a,mode,check)
                        arrays[f'{label}_{mode}']=outputs[mode].numpy()
                    reports[label]={mode:compare(outputs['batched'],outputs[mode]) for mode in ['repeat','reversed','separate_pairs']}
                    reports[label]['cpu64_reference']=compare(outputs['batched'],reference)
                    if label=='gpu32':
                        common=reference.to(device='cuda:0',dtype=torch.float32)
                        shared={mode:grouped(model,h,a,mode,check,common) for mode in ['batched','reversed','separate_pairs']}
                        arrays.update({f'common32_{mode}':v.numpy() for mode,v in shared.items()})
                        reports['common32']={mode:compare(shared['batched'],shared[mode]) for mode in ['reversed','separate_pairs']}
                times[label]=time.monotonic()-before
                assert all(torch.equal(v.cpu(),state['model'][k].to(dtype)) for k,v in model.state_dict().items())
                del model,h,a,outputs
            for value in arrays.values():assert np.isfinite(value).all()
            payload=out/f'{split}_arrays.npz';np.savez(payload,**arrays)
            result['splits'][split]=dict(episode=e,start=t,comparisons=reports,seconds=times,
                arrays_sha256=digest(payload),array_shapes={k:list(v.shape) for k,v in arrays.items()})
            atomic_json(out/'result.json',result)
        assert all(torch.equal(v.cpu(),es['model'][k]) for k,v in encoder.state_dict().items())
        verify_files(Path.cwd(),bound);verify_files(Path.cwd(),source)
        result.update(complete=True,weights_unchanged=True,inputs_unchanged=True,elapsed_seconds=time.monotonic()-start,
            limitation='Precision/feedback diagnostic only. No C004 admission, full gradient/encoder precision validation, checkpoint selection or scientific result.')
    except TimeoutError as exc:
        result.update(status='budget_incomplete',error=str(exc),complete=False)
    except BaseException as exc:
        result.update(status='technical_blocked',error=repr(exc),complete=False)
        atomic_json(out/'result.json',result)
        raise
    atomic_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='splits'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--work-seconds',type=float,required=True)
    args=parser.parse_args();assert 0<args.work_seconds<=255
    run(args.out,args.work_seconds)
