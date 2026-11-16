"""C014 independent C001 object-probe cases, statistics and lineage inspection."""
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
BASE=Path('outputs/gated/C009/v2')
MODES=('current','rollout','shuffled','no_rollout')


def need(value,message):
    if not value:raise ValueError(message)


def same(a,b):
    if isinstance(a,dict):
        need(isinstance(b,dict) and a.keys()==b.keys(),'Summary keys differ')
        return max([same(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):
        need(isinstance(b,(list,tuple)) and len(a)==len(b),'Summary shape differs')
        return max([same(x,y) for x,y in zip(a,b)] or [0.])
    need(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,rtol=1e-10,atol=1e-12),'Summary value differs')
    return abs(float(a)-float(b))


def physical_targets(red,white):
    ordered=np.asarray([sorted(row.tolist()) for row in white],dtype=white.dtype)
    return np.concatenate((red,ordered.reshape(len(red),-1)),axis=1)


def summary(predicted,truth,episodes):
    need(predicted.ndim==2 and predicted.shape==truth.shape and len(episodes)==len(truth),'Invalid statistic shape')
    need(np.isfinite(predicted).all() and np.isfinite(truth).all(),'Nonfinite statistic input')
    def corr(a,b):
        # Preserve each input dtype: actual labels are float32, predictions float64.
        x=a-np.mean(a,axis=0);y=b-np.mean(b,axis=0)
        covariance=np.sum(x*y,axis=0)
        denominator=np.sqrt(np.sum(x*x,axis=0)*np.sum(y*y,axis=0))
        return np.divide(covariance,denominator,out=np.zeros_like(denominator),where=denominator>0)
    point=corr(predicted,truth)
    groups=[np.flatnonzero(episodes==e) for e in np.unique(episodes)]
    draws=np.random.default_rng(2026111510).integers(len(groups),size=(2000,len(groups)))
    boot=[]
    for draw in draws:
        rows=np.concatenate([groups[i] for i in draw])
        boot.append(corr(predicted[rows],truth[rows]))
    return dict(pearson=point.tolist(),pearson_ci95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),rmse_px=np.sqrt(np.mean((predicted-truth)**2,axis=0)).tolist(),minimum_pearson=float(point.min()),mean_pearson=float(point.mean()))


def inspect_arrays(arrays,n,episodes,current,future):
    need(set(arrays)=={'episodes','current_truth','future_truth',*MODES},'Wrong array inventory')
    need(np.array_equal(arrays['episodes'],episodes),'Wrong episode cohort')
    for key in ['current_truth','future_truth',*MODES]:
        a=arrays[key];need(a.shape==(len(episodes),2*(n+1)) and a.dtype.kind=='f' and np.isfinite(a).all(),'Invalid case shape/type or nonfinite')
    need(np.array_equal(arrays['current_truth'],current) and np.array_equal(arrays['future_truth'],future),'Targets differ from original heldout labels/sample ordering')
    need(np.array_equal(arrays['current'],arrays['no_rollout']),'Held-current predictions differ')
    reports={};colors={}
    for mode in MODES:
        predicted=arrays[mode];truth=current if mode=='current' else future
        reports[mode]=summary(predicted,truth,episodes)
        colors[mode]={}
        for color,start,stop,count in [('red',0,2,1),('white',2,2*(n+1),n)]:
            error=(predicted[:,start:stop]-truth[:,start:stop]).reshape(len(episodes),count,2)
            colors[mode][color]=dict(mean_pearson=float(np.mean(reports[mode]['pearson'][start:stop])),position_rmse_px=float(np.sqrt(np.mean(np.sum(error*error,axis=2)))))
    return reports,colors


def check():
    from go2nav.eval.balls_probes import probe_summary,targets
    import torch
    start=time.monotonic();out=BASE/'probe_inspection_software';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'inspect_probes.py');checks=[]
    def refuses(call):
        try:call()
        except ValueError:return
        raise AssertionError('Expected refusal')
    try:
        rng=np.random.default_rng(2026111511)
        episodes=np.repeat(np.arange(7),[2,3,5,4,6,3,5])
        for n,dtype in [(n,dtype) for n in [1,3,6] for dtype in [np.float32,np.float64]]:
            red=rng.uniform(3,61,(len(episodes),2)).astype(dtype);white=rng.uniform(3,61,(len(episodes),n,2)).astype(dtype)
            current=physical_targets(red,white);need(np.array_equal(current,targets(red,white)),'Independent sorted target differs')
            need(np.array_equal(current,physical_targets(red,white[:,::-1])),'Target depends on white identity ordering')
            future=(current+rng.normal(0,2,current.shape)).astype(dtype);pred=current+rng.normal(0,.5,current.shape)
            arrays=dict(episodes=episodes,current_truth=current,future_truth=future,current=pred,no_rollout=pred.copy(),rollout=future+rng.normal(0,1,future.shape),shuffled=future+rng.normal(0,3,future.shape))
            reports,colors=inspect_arrays(arrays,n,episodes,current,future)
            for mode in MODES:same(reports[mode],probe_summary(arrays[mode],current if mode=='current' else future,episodes))
            checks.append(f'n{n}_{np.dtype(dtype).name}_all4modes_unequal_episode_bootstrap_targets_and_summaries')
        x=np.column_stack((np.arange(len(episodes)),np.ones(len(episodes))*4)).astype(float)
        y=np.column_stack((-2*x[:,0]+7,np.ones(len(episodes))*9));s=summary(x,y,episodes);same(s,probe_summary(x,y,episodes));need(np.allclose(s['pearson'],[-1,0]),'Analytic correlation differs');checks.append('analytic_anticorrelation_and_constant_coordinates')
        for key in ['episodes','current_truth','future_truth','current','rollout','shuffled','no_rollout']:
            bad={k:v.copy() for k,v in arrays.items()}
            if key=='episodes':bad[key][0]=99
            elif key in ['current_truth','future_truth']:bad[key][0,0]+=.1
            elif key=='no_rollout':bad[key][0,0]+=.1
            else:bad[key][0,0]=np.nan
            refuses(lambda:inspect_arrays(bad,n,episodes,current,future))
        bad=dict(arrays);bad.pop('shuffled');refuses(lambda:inspect_arrays(bad,n,episodes,current,future));checks.append('eight_array_identity_coverage_or_nonfinite_corruptions_refused')
        bad=copy.deepcopy(reports);bad['rollout']['pearson_ci95'][0][0]+=.1;refuses(lambda:same(reports,bad));checks.append('changed_report_interval_refused')
        large_truth=rng.uniform(3,61,(5000,8)).astype(np.float32)
        large_pred=large_truth+rng.normal(0,3,large_truth.shape);large_episodes=np.repeat(np.arange(1000),5)
        same(summary(large_pred,large_truth,large_episodes),probe_summary(large_pred,large_truth,large_episodes));checks.append('5000rows_1000episodes_float32targets_float64predictions')
        need(not torch.cuda.is_initialized(),'CUDA initialized');checks.append('no_CUDA')
        r=dict(research_id='C014',scientific_research_id='C001',research_sha256=require_record('C014')[1],scientific_research_sha256=require_record('C001')[1],status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-start,source_sha256=digest(__file__),repository_source_sha256={'go2nav/eval/balls_probes.py':digest('go2nav/eval/balls_probes.py')},preconsultation_sha256=digest(BASE/'pre_probe_inspection_implementation.json'),statistics_consultation_sha256=digest(BASE/'pre_probe_inspector_statistics.json'),limitation='CPU synthetic statistic and corruption fixtures, no GPU or model inference. Actual inspection cannot reconstruct absent training features, ridge residual or latent predictions.')
        r['dtype_correction_consultation_sha256']=digest(BASE/'pre_probe_inspector_dtype_correction.json')
        atomic_json(out/'result.json',r);print(json.dumps(r),flush=True)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C014',status='technical_blocked',checks=checks,error=repr(exc),source_sha256=digest(__file__)));raise


def actual(name):
    import torch
    from go2nav.eval.slots_inference import require_numerical_review
    start=time.monotonic();record,rh=require_record('C014');_,sh=require_record('C001');need(name in record['protocol']['predictor_cases'],'Unplanned probe case')
    torch.set_num_threads(2);sw=BASE/'probe_inspection_software';need(require_review(sw/'review.json')['status']=='passed','Unreviewed inspector')
    sr=json.loads((sw/'result.json').read_text());need(sr['source_sha256']==digest(__file__) and sr['research_sha256']==rh and sr['scientific_research_sha256']==sh,'Changed source/protocol');verify_files(Path.cwd(),sr['repository_source_sha256'])
    out=Path('outputs/gated/probes')/name;op=BASE/'runs/probes'/name
    r=json.loads((out/'result.json').read_text());o=json.loads((op/'result.json').read_text());m=json.loads((op/'manifest.json').read_text());sm=json.loads((out/'manifest.json').read_text())
    need((o['research_id'],o['scientific_research_id'],o['stage'],o['name'])==('C009','C001','probes',name),'Wrong operation identity')
    need(o['status']=='awaiting_literature_review' and o['returncode']==0 and o['child_terminal'] and o['inputs_unchanged'] and not any(o[k] for k in ['timed_out','actions','errors']),'Failed or incomplete operation')
    need(o['manifest_sha256']==digest(op/'manifest.json'),'Operation manifest changed');verify_files(out,o['child_artifact_sha256']);verify_files(Path.cwd(),m['preconsultation']['inputs_sha256'])
    try:live=psutil.Process(m['pid']).create_time()==m['creation_time']
    except psutil.NoSuchProcess:live=False
    need(not live,'Owned child live')
    kind,count,seed=name.split('_');n=int(count[1:]);seed=int(seed[1:]);need((r['research_id'],r['kind'],r['n'],r['status'])==('C001',kind,n,'awaiting_literature_review'),'Wrong science identity')
    checkpoint=Path('outputs/gated/C001')/name/'model.pt';num=Path('outputs/gated/C014')/name/'review.json';numerics=require_numerical_review(checkpoint,num);need((numerics['kind'],numerics['n'],numerics['seed'])==(kind,n,seed),'Wrong numerical parent')
    need(sm['research_id']=='C001' and sm['research_sha256']==sh and sm['code_commit']==m['source_commit'],'Science source/protocol differs')
    need(sm['config']==dict(stage='probes',n=n,train_windows=4096,test_windows_per_episode=5,ridge_alpha=.001),'Probe settings differ')
    runner=json.loads((BASE/'software/result.json').read_text());need(m['source_sha256']==runner['source_sha256'] and m['software_review_sha256']==digest(BASE/'software/review.json'),'Runner software differs');verify_files(BASE,m['source_sha256']);verify_files(Path.cwd(),runner['repository_source_sha256'])
    need(m['integration_research_sha256']==rh,'Integration protocol differs')
    ledger=json.loads(Path('outputs/gated/gpu_intervals.json').read_text());owned=[]
    for rid in [m['outer_interval_id'],sm['run_id']]:
        rows=[x for x in ledger if x['id']==rid];need(len(rows)==1 and rows[0]['research_id']=='C001' and rows[0]['end'] is not None,'Owned interval incomplete');owned.append(rows[0])
    need(r['cases_sha256']==digest(out/'cases.npz'),'Case bytes changed')
    with np.load(out/'cases.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    frames=np.load(f'outputs/data/balls/n{n}/test/frames.npy',mmap_mode='r');need(frames.shape[:2]==(1000,100),'Wrong heldout dataset')
    episodes=np.repeat(np.arange(1000),5);starts=np.random.default_rng([2026111511,n,True]).integers(2,96,size=5000)
    with np.load(f'outputs/data/balls/n{n}/test/labels.npz') as labels:
        current=physical_targets(labels['red'][episodes,starts],labels['white'][episodes,starts]);future=physical_targets(labels['red'][episodes,starts+4],labels['white'][episodes,starts+4])
    report,colors=inspect_arrays(arrays,n,episodes,current,future);difference=same(report,r['measures'])
    probe=torch.load(out/'probe.pt',map_location='cpu',weights_only=False);need(probe['research_id']=='C001' and probe['checkpoint_sha256']==digest(checkpoint),'Probe checkpoint binding differs')
    width=(n+2)*128
    for key,shape in [('mean',(width,)),('std',(width,)),('weights',(width+1,2*(n+1)))]:
        value=probe[key];need(value.shape==shape and value.dtype==torch.float64 and torch.isfinite(value).all(),'Invalid readout tensor')
    need((probe['std']>=1e-8).all() and not torch.cuda.is_initialized(),'Invalid standardization or CUDA initialized')
    result=dict(research_id='C014',scientific_research_id='C001',name=name,status='awaiting_literature_review',run_identity=dict(n=n,kind=kind,seed=seed,checkpoint_sha256=digest(checkpoint)),result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'),source_sha256=digest(__file__),software_review_sha256=digest(sw/'review.json'),cases=5000,episodes=1000,bootstrap_repeats=2000,summary_max_abs_difference=difference,measures=report,color_summaries=colors,probe_sha256=digest(out/'probe.pt'),owned_closed_intervals=owned,cuda_initialized=False,elapsed_seconds=time.monotonic()-start,limitation='Independent exact target/cohort verification and all raw-prediction summary/episode-bootstrap reconstruction. Finite saved readout parameters and checkpoint identity verify. Original fit features and latent features are absent, so no independent fit residual, coefficient reconstruction or model-output regeneration is claimed.')
    need(not (out/'inspection.json').exists(),'Inspection already exists');atomic_json(out/'inspection.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check',action='store_true');g.add_argument('--run');a=p.parse_args()
    if a.check:check()
    else:
        try:actual(a.run)
        except BaseException as exc:
            failure=BASE/'probe_inspection_failures'/a.run
            need(not failure.exists(),'Prior failed inspection must be reviewed before a new attempt')
            failure.mkdir(parents=True);shutil.copy2(__file__,failure/'inspect_probes.py');atomic_json(failure/'result.json',dict(research_id='C014',name=a.run,status='technical_blocked',error=repr(exc),source_sha256=digest(__file__)));raise
