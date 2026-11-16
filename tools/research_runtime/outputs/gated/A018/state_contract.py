"""Fresh baseline completion checks; no forward, backward or optimizer update."""
import copy
from dataclasses import asdict
import json
import math
import torch
from go2nav.training import balls


def need(condition,message):
    if not condition:raise ValueError(message)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def same_json(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)


def finite(value):
    if isinstance(value,torch.Tensor):need(value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Nonfinite/non-CPU state')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,float):need(math.isfinite(value),'Nonfinite scalar')


def construct(kind):
    need(kind in ['l0','l1'],'Unplanned model kind')
    model=balls.LeWorldModel(balls.model_config())
    head=balls.FourierHead() if kind=='l1' else None
    params=list(model.parameters())+([] if head is None else list(head.parameters()))
    optimizer=torch.optim.AdamW(params,lr=balls.DEFAULTS['lr'],weight_decay=balls.DEFAULTS['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,balls.schedule(balls.DEFAULTS['warmup'],303000))
    return model,head,params,optimizer,scheduler


def state_checks(full,final,cfg,rows,raw,shape):
    kind,seed=cfg['kind'],cfg['seed']
    need(kind in ['l0','l1'] and cfg['n']==3 and seed in [1,2],'Unplanned identity')
    need(tuple(shape)==(1000,100,64,64,3),'Original training shape changed')
    batches=shape[0]*(shape[1]-balls.WINDOW+1)//balls.DEFAULTS['batch']
    total=batches*balls.DEFAULTS['epochs'];need(total==303000,'Unexpected total updates')
    expected=dict(kind=kind,n=3,seed=seed,run_seed=2026111500+seed,train=balls.DEFAULTS,
                  model=asdict(balls.model_config()),total_steps=total)
    need(same_json(cfg,expected),'Original recipe/identity changed')
    need((raw['research_id'],raw['kind'],raw['n'],raw['seed'],raw['step'])==('C001',kind,3,seed,total),'Raw completion mismatch')
    for label,state in [('full',full),('final',final)]:
        need(state.get('research_id')=='C001' and state.get('step')==total and not state.get('inference_only'),'Incomplete/inference-only '+label)
        need(same_json(state.get('identity'),cfg),'State identity mismatch '+label)
        for key,c in [('kind','kind'),('n','n'),('train','train'),('config','model')]:
            need(same_json(state.get(key),cfg[c]),'State configuration mismatch '+label+'.'+key)
        finite(state)
    for key in ['model_state_dict','head_state_dict']:
        need(equal(full[key],final[key]),'Final/full state differs: '+key)
    model,head,params,optimizer,scheduler=construct(kind)
    model.load_state_dict(full['model_state_dict'],strict=True)
    need(equal(model.state_dict(),full['model_state_dict']),'Model restore changed state')
    if head is None:need(full['head_state_dict'] is None,'Unexpected L0 head')
    else:
        head.load_state_dict(full['head_state_dict'],strict=True)
        need(equal(head.state_dict(),full['head_state_dict']),'Head restore changed state')
    opt=full['optimiser'];expected_opt=optimizer.state_dict();group=copy.deepcopy(expected_opt['param_groups'][0])
    final_lr=balls.DEFAULTS['lr']*balls.schedule(balls.DEFAULTS['warmup'],total)(total)
    group['lr']=final_lr
    need(equal(opt['param_groups'],[group]),'Optimizer groups/ordered IDs/hyperparameters changed')
    ids=group['params']
    need(set(opt['state'])==set(ids) and len(ids)==len(params),'Incomplete optimizer state coverage')
    for key,param in zip(ids,params):
        value=opt['state'][key]
        need(set(value)=={'step','exp_avg','exp_avg_sq'},'Wrong optimizer state fields')
        step=value['step']
        need(isinstance(step,torch.Tensor) and step.shape==torch.Size([]) and step.dtype==torch.float32
             and float(step)==total,'Optimizer counter mismatch')
        for field in ['exp_avg','exp_avg_sq']:
            need(value[field].shape==param.shape and value[field].dtype==param.dtype==torch.float32,'Wrong optimizer moment shape/dtype')
        need(bool((value['exp_avg_sq']>=0).all()),'Negative second moment')
    expected_scheduler=scheduler.state_dict() | dict(last_epoch=total,_step_count=total+1,_last_lr=[final_lr])
    need(equal(full['scheduler'],expected_scheduler),'Scheduler state/cursor/rate mismatch')
    scheduler.load_state_dict(copy.deepcopy(full['scheduler']))
    optimizer.load_state_dict(copy.deepcopy(opt))
    need(equal(optimizer.state_dict(),opt) and equal(scheduler.state_dict(),full['scheduler']),'Optimizer/scheduler restore mismatch')
    need(scheduler.get_last_lr()==[final_lr] and optimizer.param_groups[0]['lr']==final_lr,'Restored rate mismatch')
    for key in ['rng','cuda_rng']:
        value=full[key]
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8
             and value.ndim==1 and value.numel()>0,'Invalid RNG payload '+key)
    generator=torch.Generator(device='cpu');generator.set_state(full['rng'])
    need(torch.equal(generator.get_state(),full['rng']),'CPU RNG restore mismatch')
    finite(rows)
    train_steps=list(range(500,total+1,500));val_steps=list(range(20000,total+1,20000))+[total]
    need([r['step'] for r in rows if 'validation' not in r]==train_steps,'Training log coverage/order')
    need([r['step'] for r in rows if 'validation' in r]==val_steps,'Validation log coverage/order')
    order=[(r['step'],int('validation' in r)) for r in rows]
    need(order==sorted(order),'Combined metric ordering mismatch')
    losses=['pred','sigreg','loss']+(['aux_encoded','aux_predicted'] if kind=='l1' else [])
    for r in rows:
        need(r['research_id']=='C001','Wrong log research identity')
        validation='validation' in r;v=r['validation'] if validation else r
        required=losses if validation else losses+['epoch','lr','grad','steps_per_s']
        need(all(k in v and isinstance(v[k],(int,float)) and not isinstance(v[k],bool) for k in required),'Incomplete metric fields')
        if not validation:
            need(r['epoch']==(r['step']-1)//batches,'Wrong logged epoch')
            expected_lr=balls.DEFAULTS['lr']*balls.schedule(balls.DEFAULTS['warmup'],total)(r['step'])
            need(math.isclose(r['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15),'Logged schedule mismatch')
    need(not torch.cuda.is_initialized(),'Unexpected GPU initialization')
    return dict(step=total,optimizer_parameters=len(params),optimizer_populated_states=len(ids),
        optimizer_step_min=total,optimizer_step_max=total,full_state_restored_exactly=True,final_matches_full=True,
        all_state_and_logged_values_finite=True,training_rows=len(train_steps),validation_rows=len(val_steps),
        cpu_rng_restored=True,gpu_rng_structure_checked_only=True,cuda_initialized=False,optimizer_updates_executed=0)
