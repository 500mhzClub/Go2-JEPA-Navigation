"""Independent C014/C002 raw binding-case inspection; no model inference."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import psutil
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
from common import operation_metadata
BASE=Path('outputs/gated/C009/v2')


def need(value,message):
    if not value:raise ValueError(message)


def same(a,b):
    if isinstance(a,dict):
        need(isinstance(b,dict) and a.keys()==b.keys(),'Summary keys differ')
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        need(isinstance(b,(list,tuple)) and len(a)==len(b),'Summary shape differs')
        for x,y in zip(a,b):same(x,y)
    elif a is None or isinstance(a,(bool,str,int)):need(a==b,'Summary value differs')
    else:need(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,rtol=1e-12,atol=1e-12),'Summary float differs')


def inspect_rows(rows,n,episodes,starts,labels):
    need(len(rows)==len(episodes)*(n+1),'Incomplete binding cases')
    columns=['episode','frame','ball','iou','bound','best_slot','reused','visible_fraction']
    need(all(set(r)==set(columns) for r in rows),'Wrong raw fields')
    for j,r in enumerate(rows):
        w,k=divmod(j,n+1)
        need((r['episode'],r['frame'],r['ball'])==(int(episodes[w]),int(starts[w]),k),'Wrong sample order or identity')
        need(type(r['bound']) is bool and type(r['reused']) is bool,'Invalid boolean')
        need(type(r['best_slot']) is int and 0<=r['best_slot']<n+2,'Invalid slot index')
        need(np.isfinite(r['iou']) and 0<=r['iou']<=1 and np.isfinite(r['visible_fraction']) and 0<=r['visible_fraction']<=1,'Invalid raw metric')
        need(r['bound']==(r['iou']>=.5),'Threshold decision differs')
    slot=np.array([r['best_slot'] for r in rows]).reshape(-1,n+1)
    reused=(slot[:,:,None]==slot[:,None,:]).sum(2)>1
    need(np.array_equal(reused.ravel(),[r['reused'] for r in rows]),'Slot reuse differs')
    axis=np.arange(64,dtype=float)+.5
    for begin in range(0,len(episodes),32):
        e,t=episodes[begin:begin+32],starts[begin:begin+32]
        points=np.concatenate((labels['red'][e,t,None,:],labels['white'][e,t]),1)
        dx=axis[None,None,None,:]-points[:,:,0,None,None]
        dy=axis[None,None,:,None]-points[:,:,1,None,None]
        discs=dx*dx+dy*dy<=9.
        # Reproduce the painter in forward order, independently of production's
        # backwards occlusion accumulation: whites in order, then red.
        owner=np.full((len(e),64,64),-1,dtype=int)
        for k in [*range(1,n+1),0]:owner[discs[:,k]]=k
        for w in range(len(e)):
            for k in range(n+1):
                fraction=np.count_nonzero(owner[w]==k)/np.count_nonzero(discs[w,k])
                need(abs(fraction-rows[(begin+w)*(n+1)+k]['visible_fraction'])<1e-15,'Geometric visibility differs')
    ep=np.array([r['episode'] for r in rows]);balls=np.array([r['ball'] for r in rows]);bound=np.array([r['bound'] for r in rows]);visible=np.array([r['visible_fraction'] for r in rows]);ious=np.array([r['iou'] for r in rows]);reuse=np.array([r['reused'] for r in rows])
    unique=np.unique(ep);inverse=np.searchsorted(unique,ep);report={}
    selectors={'all':np.ones(len(rows),bool),'red':balls==0,'white':balls>0,'at_least_90_percent_visible':visible>=.9}
    for name,sel in selectors.items():
        count=int(sel.sum())
        if not count:
            report[name]=dict(count=0,bound_share=None,bound_ci95=None,mean_best_iou=None);continue
        counts=np.bincount(inverse[sel],minlength=len(unique));success=np.bincount(inverse[sel],weights=bound[sel],minlength=len(unique))
        choices=np.random.default_rng(2026111512).integers(len(unique),size=(2000,len(unique)))
        den=counts[choices].sum(1);num=success[choices].sum(1);bootstrap=num[den>0]/den[den>0]
        report[name]=dict(count=count,bound_share=float(bound[sel].mean()),bound_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),mean_best_iou=float(ious[sel].mean()),mean_visible_fraction=float(visible[sel].mean()),best_slot_reused_share=float(reuse[sel].mean()))
    return report


def check():
    from go2nav.eval.balls_binding import discs,measure,summarize
    import torch
    started=time.monotonic();out=BASE/'binding_inspection_software';out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'inspect_binding.py');groups=[]
    def rejects(call):
        try:call()
        except ValueError:return
        raise AssertionError('Expected corrupt case refusal')
    try:
        for n,mode in [(1,'perfect'),(3,'perfect'),(6,'perfect'),(3,'merged'),(1,'occluded')]:
            points=np.array([[10+15*(k%3),10+15*(k//3)] for k in range(n+1)],dtype=float)
            if mode=='occluded':points[1]=points[0]
            red=np.broadcast_to(points[0],(3,5,2));white=np.broadcast_to(points[1:],(3,5,n,2));labels=dict(red=red,white=white)
            episodes=np.repeat(np.arange(3),2);starts=np.tile([2,3],3);full,_=discs(red[episodes,starts],white[episodes,starts])
            foreground=full.any(1,keepdims=True);masks=np.concatenate((foreground if mode=='merged' else full,~foreground),1).astype(float)
            values=measure(masks,red[episodes,starts],white[episodes,starts]);rows=[dict(episode=int(e),frame=int(t),ball=k,**{key:value[w,k].item() for key,value in values.items()}) for w,(e,t) in enumerate(zip(episodes,starts)) for k in range(n+1)]
            actual=inspect_rows(rows,n,episodes,starts,labels);same(actual,summarize(rows));groups.append(f'n{n}_{mode}_complete_summary_geometry_and_bootstrap')
        for field,value in [('episode',99),('frame',99),('ball',99),('iou',float('nan')),('bound',False),('best_slot',9),('reused',False),('visible_fraction',.1)]:
            bad=copy.deepcopy(rows);bad[0][field]=value;rejects(lambda:inspect_rows(bad,n,episodes,starts,labels))
        rejects(lambda:inspect_rows(rows[:-1],n,episodes,starts,labels));groups.append('nine_raw_case_corruptions_refused')
        wrong=copy.deepcopy(actual);wrong['all']['bound_ci95'][0]+=.1;rejects(lambda:same(actual,wrong));groups.append('corrupt_summary_refused')
        need(not torch.cuda.is_initialized(),'CUDA initialized');groups.append('no_CUDA')
        result=dict(research_id='C014',scientific_research_id='C002',status='awaiting_literature_review',checks=groups,elapsed_seconds=time.monotonic()-started,source_sha256=digest(__file__),research_sha256=require_record('C014')[1],scientific_research_sha256=require_record('C002')[1],preconsultation_sha256=digest(BASE/'pre_binding_inspection_implementation.json'),repository_source_sha256={'go2nav/eval/balls_binding.py':digest('go2nav/eval/balls_binding.py')},limitation='Synthetic masks and raw rows only. Actual inspection will reconstruct raw-case summaries and geometry, not rerun the unsaved decoder masks or their raw IoUs.')
        atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C014',status='technical_blocked',checks=groups,error=repr(exc)));raise


def actual(name):
    started=time.monotonic();record,rh=require_record('C014');cfg,sh=require_record('C002');need(name in record['protocol']['encoder_cases'],'Unplanned binding case')
    sw=BASE/'binding_inspection_software';need(require_review(sw/'review.json')['status']=='passed','Unreviewed inspection software');sr=json.loads((sw/'result.json').read_text());need(sr['source_sha256']==digest(BASE/'inspect_binding.py') and sr['research_sha256']==rh and sr['scientific_research_sha256']==sh,'Changed inspection source/protocol');verify_files(Path.cwd(),sr['repository_source_sha256'])
    out=Path('outputs/gated/binding')/name;op=BASE/'runs/binding'/name
    r=json.loads((out/'result.json').read_text());o=json.loads((op/'result.json').read_text());m=json.loads((op/'manifest.json').read_text());sm=json.loads((out/'manifest.json').read_text())
    operation_metadata('binding',name,o,m,sm)
    need(o['status']=='awaiting_literature_review' and o['returncode']==0 and o['child_terminal'] and o['inputs_unchanged'] and not any(o[k] for k in ['timed_out','actions','errors']),'Operation incomplete or failed')
    need(o['manifest_sha256']==digest(op/'manifest.json'),'Operation manifest changed');verify_files(out,o['child_artifact_sha256']);verify_files(Path.cwd(),m['preconsultation']['inputs_sha256'])
    need(not (psutil.pid_exists(m['pid']) and psutil.Process(m['pid']).create_time()==m['creation_time']),'Owned process still live')
    _,count,seed=name.split('_');n=int(count[1:]);seed=int(seed[1:]);need((r['research_id'],r['kind'],r['n'],r['seed'],r['status'])==('C002','slots',n,seed,'awaiting_literature_review'),'Wrong scientific identity')
    need(r['encoder_sha256']==digest(Path('outputs/gated/C001')/name/'model.pt') and r['cases_sha256']==digest(out/'cases.jsonl'),'Wrong selected weights or cases')
    need(sm['research_sha256']==sh and sm['config']['protocol']==cfg['protocol']['binding'],'Scientific protocol differs')
    need(m['source_sha256']==json.loads((BASE/'software/result.json').read_text())['source_sha256'] and m['software_review_sha256']==digest(BASE/'software/review.json'),'Runner software differs');verify_files(BASE,m['source_sha256'])
    need(sm['research_id']=='C002' and m['scientific_research_id']=='C002','Wrong operation record')
    ledger=json.loads(Path('outputs/gated/gpu_intervals.json').read_text());owned=[]
    for rid in [m['outer_interval_id'],sm['run_id']]:
        rows=[v for v in ledger if v['id']==rid];need(len(rows)==1 and rows[0]['research_id']=='C002' and rows[0]['end'] is not None,'Owned interval incomplete');owned.append(rows[0])
    frames=np.load(f'outputs/data/balls/n{n}/test/frames.npy',mmap_mode='r');config=cfg['protocol']['binding'];need(len(frames)==config['episodes'],'Test cohort differs')
    episodes=np.repeat(np.arange(len(frames)),config['windows_per_episode']);starts=np.random.default_rng([config['seed'],n,True]).integers(2,frames.shape[1]-4,size=len(episodes))
    rows=[json.loads(line) for line in (out/'cases.jsonl').read_text().splitlines()]
    with np.load(f'outputs/data/balls/n{n}/test/labels.npz') as labels:measures=inspect_rows(rows,n,episodes,starts,labels)
    same(measures,r['measures']);result=dict(research_id='C014',execution_integration=m['execution_integration'],scientific_research_id='C002',name=name,status='awaiting_literature_review',result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'),source_sha256=digest(__file__),software_review_sha256=digest(sw/'review.json'),cases=len(rows),windows=len(episodes),episodes=len(frames),measures=measures,owned_closed_intervals=owned,torch_imported='torch' in sys.modules,elapsed_seconds=time.monotonic()-started,limitation='Independent sample/threshold/slot-reuse/geometry and complete summary/bootstrap reconstruction. Decoder masks were not saved, so their IoU values are not independently recomputed; no new prediction/planning claim.')
    need(not (out/'inspection.json').exists(),'Inspection already exists');atomic_json(out/'inspection.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    try:actual(a.run)
    except BaseException as exc:
        failure=Path('outputs/gated/U012/failures/binding')/a.run
        need(not failure.exists(),'Prior failed inspection must be reviewed before a new attempt')
        failure.mkdir(parents=True);shutil.copy2(__file__,failure/'inspect_binding.py')
        atomic_json(failure/'result.json',dict(research_id='C014',execution_research_id='U012',name=a.run,status='technical_blocked',error=repr(exc),source_sha256=digest(__file__)))
        raise
