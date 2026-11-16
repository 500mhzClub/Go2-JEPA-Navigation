"""Synthetic complete-state and deliberate-corruption fixtures; no real approval."""
import ast
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from go2nav.research import atomic_json,digest,require_record,verify_files
from go2nav.training import balls
from state_contract import construct,state_checks
from audit import admission,source_checks,terminal_checks

BASE=Path('outputs/gated/A018')
SHAPE=(1000,100,64,64,3)


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,KeyError,FileNotFoundError):return
    raise AssertionError('Expected refusal')


def fixture(kind,seed):
    model,head,params,optimizer,scheduler=construct(kind)
    total=303000;opt=optimizer.state_dict();opt['param_groups'][0]['lr']=0.
    opt['state']={key:dict(step=torch.tensor(float(total)),exp_avg=torch.zeros_like(param),
                           exp_avg_sq=torch.ones_like(param)) for key,param in zip(opt['param_groups'][0]['params'],params)}
    sched=scheduler.state_dict() | dict(last_epoch=total,_step_count=total+1,_last_lr=[0.])
    cfg=dict(kind=kind,n=3,seed=seed,run_seed=2026111500+seed,train=copy.deepcopy(balls.DEFAULTS),
             model=asdict(balls.model_config()),total_steps=total)
    final=dict(config=cfg['model'],model_state_dict=copy.deepcopy(model.state_dict()),kind=kind,n=3,
               train=cfg['train'],head_state_dict=None if head is None else copy.deepcopy(head.state_dict()),
               step=total,identity=cfg,research_id='C001')
    full=copy.deepcopy(final);full.update(optimiser=opt,scheduler=sched,
        rng=torch.Generator(device='cpu').manual_seed(seed).get_state(),cuda_rng=torch.zeros(16,dtype=torch.uint8))
    rows=[];losses=dict(pred=1.,sigreg=1.,loss=1.)
    if kind=='l1':losses.update(aux_encoded=1.,aux_predicted=1.)
    for step in range(500,total+1,500):
        rows.append(dict(research_id='C001',step=step,epoch=(step-1)//1515,
                    lr=balls.DEFAULTS['lr']*balls.schedule(2000,total)(step),grad=1.,steps_per_s=1.,**losses))
        if step%20000==0 or step==total:rows.append(dict(research_id='C001',step=step,validation=dict(losses)))
    raw=dict(research_id='C001',kind=kind,n=3,seed=seed,step=total)
    return full,final,cfg,rows,raw


started=time.monotonic();_,rh=require_record('A018');out=BASE/'software';out.mkdir(exist_ok=False);checks={}
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
unchanged={str(f):digest(f) for b in ['A015','B002','B003','C005','C008','C009','C010','D003'] for f in Path(f'outputs/gated/{b}').rglob('*.py')}
try:
    assert all(os.environ.get(k)=='' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    for f in ['state_contract.py','audit.py','check.py']:ast.parse((BASE/f).read_text())
    for kind,seed in [('l0',1),('l1',1),('l0',2),('l1',2)]:
        values=fixture(kind,seed);checked=state_checks(*values,SHAPE)
        assert checked['optimizer_parameters']==(127 if kind=='l0' else 131)
        checks[f'{kind}_seed{seed}_complete_default_architecture_fixture']=checked
    values=fixture('l1',1)
    full,final,cfg,rows,raw=values;first=next(iter(full['optimiser']['state']))
    mutations=[
        ('unfinished',lambda a:a[0].update(step=302999)),
        ('inference_only',lambda a:a[1].update(inference_only=True)),
        ('wrong_seed',lambda a:a[2].update(seed=0)),
        ('wrong_run_seed',lambda a:a[2].update(run_seed=0)),
        ('wrong_raw_seed',lambda a:a[4].update(seed=2)),
        ('wrong_architecture',lambda a:a[0]['config'].update(width=32)),
        ('nonfinite_weight',lambda a:next(iter(a[0]['model_state_dict'].values())).fill_(float('nan'))),
        ('different_final_weight',lambda a:next(iter(a[1]['model_state_dict'].values())).add_(1)),
        ('missing_head',lambda a:a[1].update(head_state_dict=None)),
        ('missing_optimizer_state',lambda a:a[0]['optimiser']['state'].pop(first)),
        ('wrong_parameter_order',lambda a:a[0]['optimiser']['param_groups'][0]['params'].reverse()),
        ('wrong_weight_decay',lambda a:a[0]['optimiser']['param_groups'][0].update(weight_decay=.1)),
        ('wrong_optimizer_counter',lambda a:a[0]['optimiser']['state'][first]['step'].fill_(302999)),
        ('wrong_moment_shape',lambda a:a[0]['optimiser']['state'][first].update(exp_avg=torch.zeros(1))),
        ('wrong_moment_dtype',lambda a:a[0]['optimiser']['state'][first].update(exp_avg=a[0]['optimiser']['state'][first]['exp_avg'].double())),
        ('negative_second_moment',lambda a:a[0]['optimiser']['state'][first]['exp_avg_sq'].fill_(-1)),
        ('nonfinite_moment',lambda a:a[0]['optimiser']['state'][first]['exp_avg_sq'].fill_(float('inf'))),
        ('scheduler_cursor',lambda a:a[0]['scheduler'].update(last_epoch=302999)),
        ('scheduler_count',lambda a:a[0]['scheduler'].update(_step_count=303000)),
        ('scheduler_lr',lambda a:a[0]['scheduler'].update(_last_lr=[1.])),
        ('invalid_CPU_rng',lambda a:a[0].update(rng=torch.zeros(1,dtype=torch.uint8))),
        ('invalid_GPU_rng',lambda a:a[0].update(cuda_rng=torch.zeros(2,2,dtype=torch.uint8))),
        ('missing_metric',lambda a:a[3].pop(0)),
        ('duplicate_metric',lambda a:a[3].append(copy.deepcopy(a[3][-1]))),
        ('nonfinite_metric',lambda a:a[3][0].update(loss=float('nan'))),
        ('wrong_log_lr',lambda a:a[3][0].update(lr=1.)),
        ('wrong_log_epoch',lambda a:a[3][0].update(epoch=200)),
        ('wrong_log_research',lambda a:a[3][0].update(research_id='C010')),
    ]
    refused=[]
    for name,change in mutations:
        broken=copy.deepcopy(values);change(broken)
        rejects(lambda:state_checks(*broken,SHAPE));refused.append(name)
    rejects(lambda:state_checks(*values,(999,100,64,64,3)))
    checks['twenty_nine_state_recipe_optimizer_rng_log_shape_corruptions_refused']=refused+['wrong_data_shape']
    with tempfile.TemporaryDirectory(prefix='a018_fixture_') as tmp:
        root=Path(tmp);sf=root/'science/l1_n3_s1';opdir=root/'operations/l1_n3_s1'
        op=dict(research_id='C010',scientific_research_id='C001',stage='train',name='l1_n3_s1',
                returncode=0,child_terminal=True,timed_out=False,actions=[],errors=[],inputs_unchanged=True,
                raw_status='awaiting_literature_review',closed_interval_ids=['outer'])
        om=dict(research_id='C010',stage='train',name='l1_n3_s1',pid=os.getpid(),creation_time=0.,cwd=str(root),outer_interval_id='outer')
        sm=dict(run_id='inner')
        rows=[dict(id='outer',research_id='C001',supervisor_research_id='C010',start=1,end=3,out=str(opdir)),
              dict(id='inner',research_id='C001',start=2,end=3,out=str(sf))]
        terminal_checks(op,om,sm,rows,sf,opdir)
        bad=[]
        for key,val in [('returncode',1),('child_terminal',False),('timed_out',True),('actions',['TERM']),('errors',['error']),('inputs_unchanged',False),('name','l1_n3_s2'),('raw_status','budget_incomplete')]:
            v=copy.deepcopy(op);v[key]=val;bad.append((v,om,sm,rows))
        v=copy.deepcopy(om);v['creation_time']=psutil.Process(os.getpid()).create_time();bad.append((op,v,sm,rows))
        for key,val in [('end',None),('research_id','C005'),('supervisor_research_id','C009'),('out','foreign')]:
            v=copy.deepcopy(rows);v[0][key]=val;bad.append((op,om,sm,v))
        bad.append((op,om,sm,rows+[copy.deepcopy(rows[0])]))
        for args in bad:rejects(lambda:terminal_checks(*args,sf,opdir))
        checks['terminal_owned_closed_rows_and_fourteen_invalid_operations']=len(bad)
        runtime=root/'runtime';audit_root=root/'audit'
        for base in [runtime,audit_root]:
            (base/'go2nav').mkdir(parents=True);(base/'go2nav/code.py').write_text('unchanged computation')
            (base/'docs').mkdir();(base/'docs/research.md').write_text('old docs' if base==runtime else 'new docs')
        bound={str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*') if f.is_file()}
        source_checks(audit_root,runtime,bound)
        (runtime/'docs/research.md').write_text('changed runtime docs');rejects(lambda:source_checks(audit_root,runtime,bound))
        (runtime/'docs/research.md').write_text('old docs');(audit_root/'go2nav/code.py').write_text('changed computation')
        rejects(lambda:source_checks(audit_root,runtime,bound))
        checks['pinned_runtime_inventory_and_current_computational_source_routing']=True
    try:admission('l0_n3_s1')
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve()==Path('outputs/gated/C010/runs/l0_n3_s1/result.json').resolve()
        checks['current_missing_actual_terminal_refusal_before_output']=str(Path('outputs/gated/C010/runs/l0_n3_s1/result.json'))
    else:raise AssertionError('Expected missing actual terminal refusal')
    assert not (BASE/'runs/l0_n3_s1').exists()
    assert not torch.cuda.is_initialized() and digest(ledger)==before and all(digest(f)==h for f,h in unchanged.items())
    checks['CPU_only_no_existing_helpers_or_shared_ledger_changed']=True
    result=dict(research_id='A018',research_sha256=rh,status='awaiting_literature_review',checks=checks,
        elapsed_seconds=time.monotonic()-started,actual_completed_extra_runs=0,cuda_initialized=False,
        source_sha256={f:digest(BASE/f) for f in ['state_contract.py','audit.py','check.py']},
        repository_source_sha256={str(f):digest(f) for f in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{rid}.json':digest(f'docs/research/records/{rid}.json') for rid in ['C001','C010']},
        limitation='Four synthetic default-state fixtures, deliberate corruptions and missing-terminal refusal only. No actual extra-seed checkpoint, GPU RNG restoration or numerical inference validated.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='A018',research_sha256=rh,status='technical_blocked',checks=checks,error=repr(exc)))
    raise
