"""C008 focused CPU fixtures; no actual learned-model audit or update."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import psutil
import torch
from audit import BASE,counts,equal,make_model,state_checks,terminal_checks
from go2nav.research import atomic_json,digest,require_record,verify_files


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,KeyError):return
    raise AssertionError('Expected refusal')


def fixture(kind,n,protocol,research):
    model=make_model(kind,n);shape=(2,7,64,64,3) if kind=='slots' else (2,7,n+2,128)
    _,_,steps=counts(kind,protocol,shape);epochs=protocol['epochs']
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    opt=optimizer.state_dict();lr=0. if kind=='slots' else protocol['lr'];opt['param_groups'][0]['lr']=lr
    for key,param in zip(opt['param_groups'][0]['params'],model.parameters()):
        opt['state'][key]=dict(step=torch.tensor(float(steps)),exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
    schedule=scheduler.state_dict();schedule.update(last_epoch=steps,_step_count=steps+1,_last_lr=[lr])
    identity=dict(kind=kind,n=n,seed=0,research_sha256=research,protocol=protocol,parents={})
    rows=[dict(research_id='C001',epoch=e,train_loss=2.,validation=1.,best=1.) for e in range(epochs)]
    selected=dict(kind=kind,n=n,seed=0,research_id='C001',identity=identity,epoch=0,
                  config=asdict(model.config) if kind=='slots' else model.config,model=copy.deepcopy(model.state_dict()))
    final=dict(identity=identity,model=model.state_dict(),optimizer=opt,scheduler=schedule,epoch=epochs,batch_index=0,
               best=1.,epoch_loss_sum=0.,rng=torch.get_rng_state(),cuda_rng=torch.tensor([1,2,3],dtype=torch.uint8))
    raw=dict(research_id='C001',kind=kind,n=n,seed=0,best_validation=1.)
    return identity,rows,final,selected,raw,shape


started=time.monotonic();record,rh=require_record('C008');science,ch=require_record('C001')
out=BASE/'software';out.mkdir(exist_ok=False);checks={}
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
live={str(p):digest(p) for folder in ['B002','B003','C005','C007','D003'] for p in Path(f'outputs/gated/{folder}').glob('*.py')}
try:
    for kind,n in [('slots',1),('slots',3),('slots',6),('o0',3),('o1',3)]:
        protocol=science['protocol']['slot_training' if kind=='slots' else 'predictor_training']
        values=fixture(kind,n,protocol,ch)
        got=state_checks(*values)
        assert got['selected_epoch']==0 and got['total_updates']==protocol['epochs']
        checks[f'known_{kind}_n{n}_full_restore_first_tie']=got
        del values
    protocol=science['protocol']['slot_training'];values=fixture('slots',3,protocol,ch)
    # These deliberately corrupted fixtures are tiny encoder state copies, not training.
    bad=[]
    def change_final(key,value):
        v=copy.deepcopy(values);v[2][key]=value;return v
    bad += [change_final('epoch',7),change_final('batch_index',1),change_final('best',.5),change_final('epoch_loss_sum',1.)]
    for target,key,value in [('scheduler','last_epoch',7),('scheduler','_step_count',8),('scheduler','_last_lr',[1.])]:
        v=copy.deepcopy(values);v[2][target][key]=value;bad.append(v)
    v=copy.deepcopy(values);v[2]['optimizer']['state'][0]['step'].fill_(7);bad.append(v)
    v=copy.deepcopy(values);v[2]['optimizer']['state'].pop(0);bad.append(v)
    v=copy.deepcopy(values);v[2]['optimizer']['state'][0]['exp_avg']=torch.zeros(1);bad.append(v)
    v=copy.deepcopy(values);v[2]['optimizer']['state'][0]['exp_avg_sq'].fill_(-1);bad.append(v)
    v=copy.deepcopy(values);v[2]['optimizer']['param_groups'][0]['lr']=.01;bad.append(v)
    v=copy.deepcopy(values);v[3]['epoch']=1;bad.append(v)
    v=copy.deepcopy(values);v[3]['config']['slots']=6;bad.append(v)
    v=copy.deepcopy(values);v[3]['model']['mu'].fill_(float('nan'));bad.append(v)
    v=copy.deepcopy(values);v[3]['model'].pop('mu');bad.append(v)
    v=copy.deepcopy(values);v[1].pop();bad.append(v)
    v=copy.deepcopy(values);v[1][1]['epoch']=0;bad.append(v)
    v=copy.deepcopy(values);v[1][0]['validation']=float('inf');bad.append(v)
    v=copy.deepcopy(values);v[1][0]['best']=.5;bad.append(v)
    v=copy.deepcopy(values);v[2]['rng']=torch.ones(2,dtype=torch.uint8);bad.append(v)
    v=copy.deepcopy(values);v[2]['cuda_rng']=torch.tensor([],dtype=torch.uint8);bad.append(v)
    for value in bad:rejects(lambda:state_checks(*value))
    checks['twenty_two_corrupt_state_log_selection_cases_rejected']=len(bad);assert len(bad)==22
    del bad
    v=copy.deepcopy(values);v[1][-1].update(validation=.5,best=.5);v[3]['epoch']=7;v[2]['best']=v[4]['best_validation']=.5
    assert state_checks(*v)['selected_is_final']
    v[3]['model']['mu'].add_(1);rejects(lambda:state_checks(*v))
    checks['last_epoch_selected_final_weight_consistency']=True
    with tempfile.TemporaryDirectory(prefix='c008_fixture_') as tmp:
        root=Path(tmp);ops=root/'ops';child=root/'science';now=time.time()
        # Nonexistent pid verified here; never signal any process.
        missing=2147483647;assert not psutil.pid_exists(missing)
        om=dict(pid=missing,creation_time=0,cwd=str(root),outer_interval_id='outer')
        sm=dict(run_id='inner')
        op=dict(child_terminal=True,returncode=0,timed_out=False,actions=[],errors=[],raw_status='awaiting_literature_review',closed_interval_ids=['outer'])
        rows=[dict(id='outer',research_id='C005',out=str(ops),start=now-2,end=now),dict(id='inner',research_id='C001',out=str(child),start=now-1,end=now)]
        terminal_checks(op,om,sm,rows,child,ops)
        for key,value in [('child_terminal',False),('returncode',1),('timed_out',True),('actions',['signal']),('errors',['error']),('raw_status','budget_incomplete'),('closed_interval_ids',['unrelated'])]:
            wrong=copy.deepcopy(op);wrong[key]=value;rejects(lambda:terminal_checks(wrong,om,sm,rows,child,ops))
        wrong=copy.deepcopy(rows);wrong[1]['end']=None;rejects(lambda:terminal_checks(op,om,sm,wrong,child,ops))
        wrong=copy.deepcopy(rows);wrong[1]['out']=str(root/'other');rejects(lambda:terminal_checks(op,om,sm,wrong,child,ops))
        wrong=dict(om,pid=psutil.Process().pid,creation_time=psutil.Process().create_time());rejects(lambda:terminal_checks(op,wrong,sm,rows,child,ops))
        checks['terminal_closed_owned_rows_and_ten_invalid_operation_cases']=True
        source=root/'source.py';source.write_text('fixture');bound={str(source):digest(source)};verify_files(root,bound)
        source.write_text('tampered');rejects(lambda:verify_files(root,bound))
        checks['stale_source_digest_rejected']=True
    assert not torch.cuda.is_initialized() and digest(ledger)==before
    verify_files(Path.cwd(),live)
    checks['CPU_only_no_live_source_or_shared_ledger_mutation']=True
    result=dict(research_id='C008',research_sha256=rh,c001_record_sha256=ch,status='awaiting_literature_review',checks=checks,
        source_sha256={f:digest(BASE/f) for f in ['audit.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in Path('go2nav').rglob('*.py')},
        elapsed_seconds=time.monotonic()-started,cuda_initialized=False,actual_training_cases=0,
        limitation='Synthetic terminal-state fixtures and pure admission checks only. No actual completed C001 checkpoint audited, no forward pass or optimizer step, no GPU RNG restoration.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='repository_source_sha256'}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C008',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
