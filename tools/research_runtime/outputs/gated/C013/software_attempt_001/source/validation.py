"""C013 complete real-input numerical validation, retaining all policy failures."""
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
from policy import POLICIES, choose, load_policy
from go2nav.eval.balls_planning import cap, slot_distance
from go2nav.eval.slots_checkpoint import windows
from go2nav.eval.slots_recovery import numerics
from go2nav.research import atomic_json, dedicated_device, digest, provenance, require_record, verify_files
from go2nav.training.balls import to_pixels


def metric(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point():
        raise ValueError('Comparison shape/dtype mismatch')
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite numerical evidence')
    delta = (a-b).abs()
    return dict(passed=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-5)),
                failed_elements=int((~torch.isclose(a,b,rtol=1e-4,atol=1e-5)).sum()),
                max_abs=float(delta.max()), rms=float(delta.double().square().mean().sqrt()))


def comparison(a, b, steps=False):
    result = metric(a,b)
    if steps:
        result['by_step'] = [metric(a[:,s],b[:,s]) for s in range(a.shape[1])]
    return result


@torch.no_grad()
def pipeline(world, pixels, goal, commands):
    history = world.encode(pixels)
    target = world.encode(goal)[:,0]
    predicted = world.rollout(history, commands, 1)[:,0]
    return dict(history=history, goal=target, predicted=predicted, cost=slot_distance(predicted,target))


@torch.no_grad()
def pipeline_checks(world, pixels, goals, commands, check):
    check(); cpu = copy.deepcopy(world).cpu().eval()
    reference = pipeline(cpu,pixels,goals,commands)
    check(); batched = pipeline(world,pixels,goals,commands)
    variants = {'cpu':reference,'repeat':pipeline(world,pixels,goals,commands)}
    rev = pipeline(world,pixels.flip(0),goals.flip(0),commands.flip(0))
    variants['reversed'] = {k:v.flip(0) for k,v in rev.items()}
    separate = [pipeline(world,pixels[j:j+1],goals[j:j+1],commands[j:j+1]) for j in range(len(pixels))]
    variants['separate'] = {k:torch.cat([v[k] for v in separate]) for k in batched}
    checks = {mode:{k:comparison(batched[k],v[k]) for k in batched} for mode,v in variants.items()}
    return dict(checks=checks,passed=all(x['passed'] for row in checks.values() for x in row.values()))


@torch.no_grad()
def grouped(world, history, commands, mode, check):
    if mode == 'reversed':
        history,commands = history.flip(0),commands.flip(0)
    size = 2 if mode == 'separate' else len(history)
    values=[]
    for j in range(0,len(history),size):
        check();values.append(world.rollout(history[j:j+size],commands[j:j+size],8).cpu())
    value=torch.cat(values)
    return value.flip(0) if mode == 'reversed' else value


def commands_for(actions,e,t,split,dtype):
    rng=np.random.default_rng((2026111516,3,('train','val').index(split),100))
    future=cap((2*rng.standard_normal((100,8,2))).astype(np.float32))
    past=np.broadcast_to(actions[e,t:t+2],(100,2,2))
    return torch.from_numpy(np.concatenate((past,future),1)).to(dtype)/3


@torch.no_grad()
def planning_checks(world, frames, actions, split, e, t, out, check):
    cpu=copy.deepcopy(world).cpu().eval()
    dtype=next(world.parameters()).dtype
    pixels=to_pixels(torch.from_numpy(np.array(frames[e:e+1,t:t+3]))).to(dtype)
    goals=to_pixels(torch.from_numpy(np.array(frames[e:e+1,t+3:t+4]))).to(dtype)
    commands=commands_for(actions,e,t,split,dtype)
    cpu_h,cpu_g=cpu.encode(pixels),cpu.encode(goals)[:,0]
    gpu_h,gpu_g=world.encode(pixels),world.encode(goals)[:,0]
    arrays=dict(commands=commands.cpu(),cpu_history=cpu_h.cpu(),gpu_history=gpu_h.cpu(),
                cpu_goal=cpu_g.cpu(),gpu_goal=gpu_g.cpu())
    arrays['cpu_rollout']=grouped(cpu,cpu_h.expand(100,-1,-1,-1),commands,'batched',check)
    arrays['cpu_cost']=slot_distance(arrays['cpu_rollout'][:,-1],cpu_g)
    for mode in ('batched','repeat','reversed','separate'):
        value=grouped(world,gpu_h.expand(100,-1,-1,-1),commands,mode,check)
        arrays[f'{mode}_rollout']=value
        arrays[f'{mode}_cost']=slot_distance(value[:,-1].to(gpu_g.device),gpu_g).cpu()
    checks={mode:{'rollout':comparison(arrays['batched_rollout'],arrays[f'{mode}_rollout'],True),
                  'cost':comparison(arrays['batched_cost'],arrays[f'{mode}_cost'])}
            for mode in ('cpu','repeat','reversed','separate')}
    enc={k:comparison(arrays[f'gpu_{k}'],arrays[f'cpu_{k}']) for k in ('history','goal')}
    for a in arrays.values():
        if not torch.isfinite(a).all():raise FloatingPointError('Nonfinite saved array')
    np.savez(out,**{k:v.numpy() for k,v in arrays.items()})
    return dict(episode=e,start=t,checks=checks,encoding=enc,arrays_sha256=digest(out),
                array_shapes={k:list(v.shape) for k,v in arrays.items()},
                passed=all(x['passed'] for row in checks.values() for x in row.values()) and all(x['passed'] for x in enc.values()))


def unchanged(world, original):
    return all(torch.equal(v.cpu(),original[k].to(v.dtype)) for k,v in world.state_dict().items())


def qualify(result):
    if not result.get('complete') or set(result['windows'])!={'train','val'} or set(result['planning'])!={'train','val'}:
        raise ValueError('Incomplete policy coverage')
    checks=[]
    for rows in result['windows'].values():
        if len(rows)!=8:raise ValueError('Expected eight paired batches per split')
        for row in rows:
            if len(row['episodes'])!=2 or len(row['starts'])!=2:raise ValueError('Expected two examples')
            for key,names,count in [('encoder',{'slots','reconstruction','masks'},57),('predictor',{'predicted'},83)]:
                value=row[key]
                if set(value['outputs'])!=names or set(value['grouping'])!=names or len(value['gradients'])!=count:
                    raise ValueError('Incomplete forward/gradient coverage')
                checks.extend(x['passed'] for x in value['outputs'].values())
                checks.extend(x['passed'] for x in value['gradients'].values())
                if set(value['missing_gradients'])!={'cpu','cuda:0'}:raise ValueError('Missing device gradient report')
                checks.append(not any(value['missing_gradients'].values()))
                for groups in value['grouping'].values():
                    if set(groups)!={'repeat','reversed','separate'}:raise ValueError('Missing grouping')
                    checks.extend(x['passed'] for x in groups.values())
                checks.append(value['passed'])
            if set(row['pipeline']['checks'])!={'cpu','repeat','reversed','separate'}:raise ValueError('Incomplete pipeline')
            for groups in row['pipeline']['checks'].values():
                if set(groups)!={'history','goal','predicted','cost'}:raise ValueError('Incomplete pipeline outputs')
                checks.extend(x['passed'] for x in groups.values())
    for value in result['planning'].values():
        if set(value['checks'])!={'cpu','repeat','reversed','separate'} or set(value['encoding'])!={'history','goal'}:
            raise ValueError('Incomplete planning coverage')
        checks.extend(x['passed'] for x in value['encoding'].values())
        for groups in value['checks'].values():
            if set(groups)!={'rollout','cost'} or len(groups['rollout']['by_step'])!=8:raise ValueError('Incomplete rollout/cost')
            checks.extend(x['passed'] for x in groups.values())
    return all(checks)


def run(out,work):
    start=time.monotonic();record,rh=require_record('C013');bound,source=inputs()
    out.mkdir(exist_ok=False)
    manifest=provenance('C013',record['protocol'],bound)
    manifest['helper_sha256']={n:digest(BASE/n) for n in ('common.py','policy.py','validation.py','run.py','check.py')}
    atomic_json(out/'manifest.json',manifest)
    r=dict(research_id='C013',name=NAME,n=3,seed=0,kind='o0',checkpoint_sha256=record['protocol']['checkpoint_sha256'],
           research_sha256=rh,status='awaiting_literature_review',policies={},complete=False)
    def check():
        if time.monotonic()-start>=work:raise TimeoutError('C013 work deadline')
    try:
        dedicated_device(.5);torch.cuda.set_per_process_memory_fraction(.5)
        checkpoint='outputs/gated/C001/o0_n3_s0/model.pt'
        raw_world=load_policy(checkpoint,POLICIES[0])
        original={k:v.detach().clone() for k,v in raw_world.state_dict().items()}
        del raw_world
        for policy in POLICIES:
            check();world=load_policy(checkpoint,policy,'cuda:0');assert unchanged(world,original)
            dtype=next(world.parameters()).dtype;result=dict(windows={},planning={},complete=False)
            r['policies'][policy]=result
            for split in ('train','val'):
                frames=np.load(f'outputs/data/balls/n3/{split}/frames.npy',mmap_mode='r')
                z=np.load(f'outputs/gated/C001/extract_n3_s0/{split}.npy',mmap_mode='r')
                with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as f:actions=f['actions']
                episodes,starts=windows(frames.shape,3,split);result['windows'][split]=[]
                for j in range(0,len(episodes),2):
                    check();e,t=episodes[j:j+2,None],starts[j:j+2,None]
                    pixels=to_pixels(torch.from_numpy(np.array(frames[e,t+np.arange(3)]))).to(dtype)
                    goal=to_pixels(torch.from_numpy(np.array(frames[e,t+3]))[:,None]).to(dtype)
                    commands=torch.from_numpy(np.array(actions[e,t+np.arange(3)])).to(dtype)/3
                    states=torch.from_numpy(np.array(z[e,t+np.arange(4)])).double()
                    cached_actions=(torch.from_numpy(np.array(actions[e,t+np.arange(3)]))/3).double()
                    row=dict(episodes=e[:,0].tolist(),starts=t[:,0].tolist())
                    row['encoder']=numerics('rgb_slots',world.encoder,check,[pixels])
                    row['predictor']=numerics('predictor',world.predictor,check,
                        [states,cached_actions,torch.zeros(2,5,dtype=torch.bool)])
                    row['pipeline']=pipeline_checks(world,pixels,goal,commands,check)
                    result['windows'][split].append(row);atomic_json(out/'result.json',r)
                check();e,t=int(episodes[0]),int(starts[0])
                result['planning'][split]=planning_checks(world,frames,actions,split,e,t,out/f'{policy}_{split}.npz',check)
                atomic_json(out/'result.json',r)
            assert unchanged(world,original)
            result.update(complete=True,weights_unchanged=True)
            result['checks_passed']=qualify(result)
            atomic_json(out/'result.json',r);print(dict(policy=policy,checks_passed=result['checks_passed']),flush=True)
            del world
        verify_files(Path.cwd(),bound);verify_files(Path.cwd(),source)
        r.update(complete=True,inputs_unchanged=True,selected_policy=choose(r['policies']),elapsed_seconds=time.monotonic()-start)
        r['checks_passed']=r['selected_policy'] is not None
    except TimeoutError as exc:r.update(status='budget_incomplete',error=str(exc),checks_passed=False)
    except BaseException as exc:
        r.update(status='technical_blocked',error=repr(exc),checks_passed=False);atomic_json(out/'result.json',r);raise
    atomic_json(out/'result.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='policies'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-seconds',type=float,required=True)
    a=p.parse_args();assert 0<a.work_seconds<=1755
    with sdpa_kernel(SDPBackend.MATH):run(a.out,a.work_seconds)
