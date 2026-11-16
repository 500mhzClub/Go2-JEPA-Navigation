"""Read-only terminal C001 training audit. No inference, updates or auto-review."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path.cwd()))
import numpy as np
import psutil
import torch
from go2nav.models.balls_slots import RGBSlotConfig,RGBSlots,ObjectMaskedPredictor
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

BASE=Path('outputs/gated/C008')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def need(condition,message):
    if not condition:raise ValueError(message)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def finite(value):
    if isinstance(value,torch.Tensor):need(value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Nonfinite/non-CPU tensor')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,float):need(math.isfinite(value),'Nonfinite scalar')


def make_model(kind,n):
    need(kind in ['slots','o0','o1'],'Unplanned model kind')
    return RGBSlots(RGBSlotConfig(slots=n+2)) if kind=='slots' else ObjectMaskedPredictor()


def counts(kind,protocol,shape):
    episodes,frames=shape[:2]
    need(episodes>0 and frames>=6,'Invalid training shape')
    count=episodes*(frames-5 if kind=='slots' else frames-3)
    batches=math.ceil(count/protocol['batch'])
    return count,batches,batches*protocol['epochs']


def state_checks(identity,rows,final,selected,raw,shape):
    kind,n=identity['kind'],identity['n'];protocol=identity['protocol'];epochs=protocol['epochs']
    count,batches,steps=counts(kind,protocol,shape)
    need(final['identity']==selected['identity']==identity,'Checkpoint identity mismatch')
    need((selected['kind'],selected['n'],selected['seed'],selected['research_id'])==(kind,n,identity['seed'],'C001'),'Selected identity mismatch')
    need((raw['kind'],raw['n'],raw['seed'],raw['research_id'])==(kind,n,identity['seed'],'C001'),'Raw identity mismatch')
    need(len(rows)==epochs and [r['epoch'] for r in rows]==list(range(epochs)),'Incomplete/duplicate epoch log')
    best=math.inf;best_epoch=None
    for r in rows:
        need(r['research_id']=='C001','Wrong log research identity')
        need(all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) and r[k]>=0 for k in ['train_loss','validation','best']),'Invalid epoch metric')
        if r['validation']<best:best,best_epoch=r['validation'],r['epoch']
        need(r['best']==best,'Incorrect running best')
    need(selected['epoch']==best_epoch,'Selected epoch is not first strict validation minimum')
    need(final['best']==raw['best_validation']==best,'Final best mismatch')
    need(final['epoch']==epochs and final['batch_index']==0 and final['epoch_loss_sum']==0,'Training cursor incomplete')
    finite(final);finite(selected)
    model=make_model(kind,n);expected=asdict(model.config) if kind=='slots' else model.config
    need(selected['config']==expected,'Model architecture changed')
    model.load_state_dict(final['model'],strict=True)
    need(equal(model.state_dict(),final['model']),'Final model restore changed state')
    best_model=make_model(kind,n);best_model.load_state_dict(selected['model'],strict=True)
    need(equal(best_model.state_dict(),selected['model']),'Selected model restore changed state')
    if best_epoch==epochs-1:need(equal(final['model'],selected['model']),'Last selected weights disagree with final state')
    opt=final['optimizer'];params=list(model.parameters());groups=opt['param_groups']
    need(len(groups)==1 and len(groups[0]['params'])==len(params),'Optimizer parameter coverage mismatch')
    ids=groups[0]['params'];need(len(set(ids))==len(ids) and set(ids)==set(opt['state']),'Missing/duplicate optimizer state')
    for key,param in zip(ids,params):
        item=opt['state'][key]
        need(set(item)=={'step','exp_avg','exp_avg_sq'},'Unexpected Adam state fields')
        need(item['step'].numel()==1 and float(item['step'])==steps,'Optimizer update count mismatch')
        for name in ['exp_avg','exp_avg_sq']:need(item[name].shape==param.shape and item[name].dtype==param.dtype,'Optimizer moment shape/dtype mismatch')
        need(bool((item['exp_avg_sq']>=0).all()),'Negative second moment')
    expected_lr=0. if kind=='slots' else protocol['lr']
    scheduler=final['scheduler']
    need(scheduler['last_epoch']==steps and scheduler['_step_count']==steps+1,'Scheduler update count mismatch')
    need(scheduler['base_lrs']==[protocol['lr']] and scheduler['_last_lr']==[expected_lr],'Scheduler learning rate mismatch')
    need(groups[0]['lr']==expected_lr and groups[0]['initial_lr']==protocol['lr'],'Optimizer learning rate mismatch')
    need(groups[0]['betas']==(.9,.999) and groups[0]['eps']==1e-8 and groups[0]['weight_decay']==0 and not groups[0]['amsgrad'],'Adam recipe mismatch')
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['lr'])
    restored_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    restored_scheduler.load_state_dict(copy.deepcopy(scheduler))
    optimizer.load_state_dict(copy.deepcopy(opt))
    need(equal(optimizer.state_dict(),opt) and equal(restored_scheduler.state_dict(),scheduler),'Optimizer/scheduler restoration mismatch')
    rng=final['rng'];cuda_rng=final['cuda_rng']
    need(rng.dtype==cuda_rng.dtype==torch.uint8 and rng.ndim==cuda_rng.ndim==1 and rng.numel()>0 and cuda_rng.numel()>0,'Invalid RNG payload')
    generator=torch.Generator(device='cpu');generator.set_state(rng)
    need(torch.equal(generator.get_state(),rng),'CPU RNG restoration mismatch')
    need(not torch.cuda.is_initialized(),'CUDA initialized')
    return dict(epochs=epochs,windows_per_epoch=count,batches_per_epoch=batches,total_updates=steps,
                optimizer_parameters=len(params),selected_epoch=best_epoch,best_validation=best,
                final_epoch=final['epoch'],selected_is_final=best_epoch==epochs-1,
                strict_model_and_optimizer_restore=True,cpu_rng_restored=True,gpu_rng_bytes_only=True)


def terminal_checks(operation,manifest,science_manifest,rows,science_path,operations_path):
    need(operation['child_terminal'] and operation['returncode']==0 and not operation['timed_out'] and not operation['actions'] and not operation['errors'],'Operation incomplete/unsuccessful')
    need(operation['raw_status']=='awaiting_literature_review','Incomplete scientific stage')
    try:need(psutil.Process(manifest['pid']).create_time()!=manifest['creation_time'],'Exact trainer remains live')
    except psutil.NoSuchProcess:pass
    ids=[manifest['outer_interval_id'],science_manifest['run_id']]
    closed=set(operation['closed_interval_ids'])
    need(len(set(ids))==2 and ids[0] in closed and closed.issubset(ids),'Incorrect owned closure evidence')
    selected=[r for r in rows if r['id'] in ids]
    need(len(selected)==2 and all(r['end'] is not None and r['end']>=r['start'] for r in selected),'Owned accounting remains open/invalid')
    for identifier,research,path in [(ids[0],'C005',operations_path),(ids[1],'C001',science_path)]:
        row=next(r for r in selected if r['id']==identifier)
        need(row['research_id']==research and (Path(manifest['cwd'])/row['out']).resolve()==path.resolve(),'Ledger identity/path mismatch')


def run(name):
    started=time.monotonic();_,rh=require_record('C008');record,ch=require_record('C001')
    need(require_review(BASE/'software/review.json')['status']=='passed','Audit software not reviewed')
    sw=json.loads((BASE/'software/result.json').read_text())
    need(sw['research_sha256']==rh and sw['c001_record_sha256']==ch,'Audit protocol changed')
    verify_files(BASE,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256'])
    science=Path('outputs/gated/C001')/name
    candidates=[Path('outputs/gated/C005')/prefix/'runs'/name for prefix in ['', 'v2']]
    candidates=[p for p in candidates if (p/'result.json').exists()];need(len(candidates)==1,'Actual terminal operation missing/ambiguous')
    opdir=candidates[0];operation=json.loads((opdir/'result.json').read_text());om=json.loads((opdir/'manifest.json').read_text())
    sm=json.loads((science/'manifest.json').read_text());raw=json.loads((science/'result.json').read_text());identity=json.loads((science/'config.json').read_text())
    n,seed,kind=identity['n'],identity['seed'],identity['kind']
    need(name==f'{kind}_n{n}_s{seed}' and ((n==3 and seed in [0,1,2] and kind in ['slots','o0','o1']) or (n in [1,6] and seed==0 and kind in ['slots','o1'])),'Unplanned C001 case')
    need(raw['status']=='awaiting_literature_review' and sm['config']==identity and sm['research_sha256']==identity['research_sha256']==ch,'Scientific identity/source mismatch')
    need(identity['protocol']==record['protocol']['slot_training' if kind=='slots' else 'predictor_training'],'Frozen recipe mismatch')
    need(operation['manifest_sha256']==digest(opdir/'manifest.json'),'Operational manifest changed')
    terminal_checks(operation,om,sm,json.loads(Path('outputs/gated/gpu_intervals.json').read_text()),science,opdir)
    opbase=opdir.parent.parent;need(require_review(opbase/'software/review.json')['status']=='passed','Supervision software not reviewed')
    need(digest(opbase/'software/review.json')==om['software_review_sha256'],'Supervision review changed')
    verify_files(opbase,om['source_sha256'])
    verify_files(science,operation['child_artifact_sha256']);verify_files(Path.cwd(),sm['source_sha256']);verify_files(Path.cwd(),sm['inputs']);verify_files(Path.cwd(),identity['parents'])
    data=om['preconsultation']['data_sha256']
    expected={f'outputs/data/balls/n{n}/{split}/{file}' for split in ['train','val'] for file in ['meta.json','labels.npz','frames.npy']}
    need(set(data)==expected,'Original six-file data binding incomplete');verify_files(Path.cwd(),data)
    files={science/f for f in ['result.json','manifest.json','config.json','metrics.jsonl','model.pt','checkpoint.pt']}
    files.update([opdir/'result.json',opdir/'manifest.json']);files.update(Path(f) for f in data)
    rows=[json.loads(line) for line in (science/'metrics.jsonl').read_text().splitlines()]
    selected=torch.load(science/'model.pt',map_location='cpu',weights_only=False)
    final=torch.load(science/'checkpoint.pt',map_location='cpu',weights_only=False)
    need(raw['checkpoint_sha256']==digest(science/'model.pt'),'Selected artifact digest mismatch')
    if kind=='slots':
        need(selected['cache'] is None and selected['cache_sha256'] is None,'Unexpected encoder cache parent')
        shape=np.load(f'outputs/data/balls/n{n}/train/frames.npy',mmap_mode='r').shape
    else:
        cache=Path(selected['cache']);index=json.loads((cache/'index.json').read_text())
        need(digest(cache/'index.json')==selected['cache_sha256'] and (index['n'],index['seed'])==(n,seed),'Frozen cache identity changed')
        verify_files(cache,{f'{s}.npy':v for s,v in index['splits'].items()})
        need(digest(index['encoder'])==index['encoder_sha256'],'Frozen encoder changed')
        verify_files(Path(f'outputs/data/balls/n{n}'),index['data_metadata'])
        need(any(Path(f).resolve()==(cache/'index.json').resolve() and value==selected['cache_sha256'] for f,value in identity['parents'].items()),'Cache not bound to training parent')
        shape=np.load(cache/'train.npy',mmap_mode='r').shape
        files.update([cache/'index.json',cache/'train.npy',cache/'val.npy',Path(index['encoder'])])
    before={str(f):digest(f) for f in files}
    measured=state_checks(identity,rows,final,selected,raw,shape)
    verify_files(Path.cwd(),before)
    verify_files(science,operation['child_artifact_sha256']);verify_files(Path.cwd(),data)
    verify_files(Path.cwd(),sm['source_sha256']);verify_files(Path.cwd(),sm['inputs'])
    need(not torch.cuda.is_initialized(),'CUDA initialized')
    return dict(research_id='C008',name=name,research_sha256=rh,c001_record_sha256=ch,status='awaiting_literature_review',
        checks=measured,input_sha256=before,source_sha256=digest(__file__),software_review_sha256=digest(BASE/'software/review.json'),
        scientific_result_sha256=digest(science/'result.json'),operational_result_sha256=digest(opdir/'result.json'),
        elapsed_seconds=time.monotonic()-started,cuda_initialized=False,
        limitation='Artifact consistency and CPU restoration only; no forward/gradient checks or scientific quality claim. Earlier selected epoch provenance relies on source-bound save logic and metadata, not retained per-epoch weight snapshots.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    out=BASE/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    try:result=run(args.run)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C008',name=args.run,status='technical_blocked',error=repr(exc),source_sha256=digest(__file__)))
        raise
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)
