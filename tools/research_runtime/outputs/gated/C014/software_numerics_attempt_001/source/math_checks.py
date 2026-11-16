"""C014 math adapted from immutable C013 validation; only count is generalized."""
import copy
import numpy as np
import torch
from go2nav.eval.balls_planning import cap,slot_distance
from go2nav.training.balls import to_pixels
from go2nav.research import digest


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


def encode_frames(world, frames):
    if frames.dtype!=torch.uint8 or frames.ndim!=5 or frames.shape[-1]!=3:
        raise ValueError('Pipeline requires raw uint8 B,T,H,W,3 frames')
    parameter=next(world.parameters())
    return world.encode(to_pixels(frames.to(parameter.device)).to(parameter.dtype))


@torch.no_grad()
def pipeline(world, pixels, goal, commands):
    parameter=next(world.parameters())
    history = encode_frames(world,pixels)
    target = encode_frames(world,goal)[:,0]
    normalized=commands.to(device=parameter.device,dtype=parameter.dtype)/3
    predicted = world.rollout(history, normalized, 1)[:,0]
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


def commands_for(actions,e,t,split,dtype,device='cpu',n=3):
    rng=np.random.default_rng((2026111516,n,('train','val').index(split),100))
    future=cap((2*rng.standard_normal((100,8,2))).astype(np.float32))
    past=np.broadcast_to(actions[e,t:t+2],(100,2,2))
    return torch.from_numpy(np.concatenate((past,future),1)).to(device=device,dtype=dtype)/3


@torch.no_grad()
def planning_checks(world, frames, actions, split, e, t, out, check, n):
    cpu=copy.deepcopy(world).cpu().eval()
    dtype=next(world.parameters()).dtype
    pixels=torch.from_numpy(np.array(frames[e:e+1,t:t+3]))
    goals=torch.from_numpy(np.array(frames[e:e+1,t+3:t+4]))
    commands=commands_for(actions,e,t,split,dtype,n=n)
    gpu_commands=commands_for(actions,e,t,split,dtype,next(world.parameters()).device,n=n)
    cpu_h,cpu_g=encode_frames(cpu,pixels),encode_frames(cpu,goals)[:,0]
    gpu_h,gpu_g=encode_frames(world,pixels),encode_frames(world,goals)[:,0]
    arrays=dict(cpu_commands=commands.cpu(),gpu_commands=gpu_commands.cpu(),cpu_history=cpu_h.cpu(),gpu_history=gpu_h.cpu(),
                cpu_goal=cpu_g.cpu(),gpu_goal=gpu_g.cpu())
    arrays['cpu_rollout']=grouped(cpu,cpu_h.expand(100,-1,-1,-1),commands,'batched',check)
    arrays['cpu_cost']=slot_distance(arrays['cpu_rollout'][:,-1],cpu_g)
    for mode in ('batched','repeat','reversed','separate'):
        value=grouped(world,gpu_h.expand(100,-1,-1,-1),gpu_commands,mode,check)
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
