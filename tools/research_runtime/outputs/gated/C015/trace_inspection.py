"""Independent C015 exhaustive assignment and CEM trace inspection."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import psutil
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
BASE=Path('outputs/gated/C015')
MODES=('original','bulk_cost_copy_original_per_matrix_GPU_sum','bulk_cost_copy_batched_indices_GPU_sum')


def need(v,msg):
    if not v:raise ValueError(msg)


def assignment(matrices):
    need(matrices.shape[-2:]==(5,5) and np.isfinite(matrices).all(),'Invalid matrices')
    shape=matrices.shape[:-2];flat=matrices.reshape(-1,5,5)
    permutations=np.asarray(list(itertools.permutations(range(5))))
    return flat[:,np.arange(5)[None,:],permutations].sum(-1).min(-1).reshape(shape)


def cap(a):
    return a*np.minimum(1,3/np.maximum(np.sqrt(np.sum(a*a,axis=-1,keepdims=True)),1e-12))


def audit_trace(a,h,seed):
    expected=dict(proposals=(20,100,h,2),costs=(20,100),order=(20,100),matrices=(20,100,5,5),actions=(h,2))
    need(set(a)==set(expected),'Wrong trace array inventory')
    for k,shape in expected.items():need(a[k].shape==shape and np.isfinite(a[k]).all(),'Invalid trace shape/nonfinite')
    computed=assignment(a['matrices']);need(np.allclose(computed,a['costs'],rtol=1e-12,atol=1e-12),'Assignment objective differs')
    rng=np.random.default_rng(seed);mean=np.zeros((h,2),np.float32);scale=np.full((h,2),2.,np.float32)
    for step in range(20):
        proposals=cap(mean+scale*rng.standard_normal((100,h,2)).astype(np.float32));need(np.array_equal(proposals,a['proposals'][step]),'Proposal stream differs')
        order=np.argsort(a['costs'][step],kind='stable');need(np.array_equal(order,a['order'][step]),'Ranking differs')
        selected=proposals[order[:10]];mean=selected.mean(0);scale=np.maximum(selected.std(0),1e-3)
    need(np.array_equal(cap(mean),a['actions']),'Final elite mean differs')
    return float(np.max(np.abs(computed-a['costs'])))


def check():
    import torch
    from go2nav.eval.balls_planning import cem
    start=time.monotonic();out=BASE/'inspection_software';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'inspect.py');checks=[]
    try:
        m=np.full((5,5),100.);np.fill_diagonal(m,1.)
        need(np.array_equal(assignment(np.stack([m,m[:,::-1],m*0])),[5.,5.,0.]),'Analytic assignment differs');checks.append('analytic_permuted_and_tied_complete_assignments')
        a={k:[] for k in ['proposals','costs','order','matrices']}
        def cost(proposals):
            values=np.sum(proposals**2,axis=(1,2)).astype(np.float64);m=np.full((100,5,5),1e6);m[:,np.arange(5),np.arange(5)]=0;m[:,0,0]=values
            for k,v in [('proposals',proposals),('costs',values),('order',np.argsort(values,kind='stable')),('matrices',m)]:a[k].append(v.copy())
            return values
        actions=cem(cost,4,[2026111504,3,0,4]);a={k:np.stack(v) for k,v in a.items()};a['actions']=actions;need(audit_trace(a,4,[2026111504,3,0,4])==0.,'Synthetic trace differs');checks.append('original_full20iteration_CEM_proposals_assignments_ranks_actions')
        def refuses(call):
            try:call()
            except ValueError:return
            raise AssertionError('Expected refusal')
        for key in ['proposals','costs','order','matrices','actions']:
            bad={k:v.copy() for k,v in a.items()};bad[key].flat[0]+=.1 if key!='order' else 1;refuses(lambda:audit_trace(bad,4,[2026111504,3,0,4]))
        bad=dict(a);bad.pop('matrices');refuses(lambda:audit_trace(bad,4,[2026111504,3,0,4]));refuses(lambda:audit_trace(a,4,[0]));checks.append('seven_matrix_proposal_cost_rank_action_coverage_seed_refusals')
        need(not torch.cuda.is_initialized(),'CUDA initialized');checks.append('no_CUDA')
        r=dict(research_id='C015',research_sha256=require_record('C015')[1],status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-start,source_sha256=digest(__file__),preconsultation_sha256=digest(BASE/'pre_trace_inspection.json'),limitation='Synthetic exhaustive-cost and complete CEM trace reconstruction only; no neural replay or actual GPU result interpretation.')
        atomic_json(out/'result.json',r);print(json.dumps(r),flush=True)
    except BaseException as exc:atomic_json(out/'result.json',dict(research_id='C015',status='technical_blocked',checks=checks,error=repr(exc)));raise


def actual():
    start=time.monotonic();sw=BASE/'inspection_software';need(require_review(sw/'review.json')['status']=='passed','Inspector unreviewed');s=json.loads((sw/'result.json').read_text());need(s['source_sha256']==digest(__file__) and s['research_sha256']==require_record('C015')[1],'Inspector source/protocol changed')
    out=BASE/'diagnostic';op=BASE/'operations';r=json.loads((out/'result.json').read_text());o=json.loads((op/'result.json').read_text());m=json.loads((op/'manifest.json').read_text());dm=json.loads((out/'manifest.json').read_text())
    need(o['research_id']=='C015' and o['returncode']==0 and o['child_terminal'] and not any(o[k] for k in ['timed_out','actions','errors']),'Incomplete operation');need(r['complete'] and r['inputs_unchanged'] and len(r['cases'])==12,'Incomplete diagnostic')
    need(o['manifest_sha256']==digest(op/'manifest.json') and r['manifest_sha256']==digest(out/'manifest.json') and dm['operation_manifest_sha256']==digest(op/'manifest.json'),'Manifest changed');verify_files(out,o['child_artifact_sha256']);verify_files(out,r['arrays_sha256']);verify_files(Path.cwd(),m['inputs']);verify_files(BASE,m['helper_sha256'])
    runtime=json.loads((BASE/'software/result.json').read_text());need(require_review(BASE/'software/review.json')['status']=='passed' and m['software_review_sha256']==digest(BASE/'software/review.json') and m['helper_sha256']==runtime['helper_sha256'],'Runtime software changed');verify_files(Path.cwd(),runtime['repository_source_sha256'])
    try:live=psutil.Process(m['pid']).create_time()==m['creation_time']
    except psutil.NoSuchProcess:live=False
    need(not live,'Child still live');owned=[x for x in json.loads(Path('outputs/gated/gpu_intervals.json').read_text()) if x['id']==m['outer_interval_id']];need(len(owned)==1 and owned[0]['end'] is not None and owned[0]['supervisor']=='C015','Interval incomplete')
    expected={(name,split,h) for name in ['o0_n3_s0','o1_n3_s0'] for split in ['train','val'] for h in [4,6,8]};keys=[(x['name'],x['split'],x['horizon']) for x in r['cases']];need(len(keys)==len(set(keys)) and set(keys)==expected,'Case identities differ')
    expected_files={f'{name}_{split}_h{h}_{mode}.npz' for name,split,h in expected for mode in MODES};need(set(r['arrays_sha256'])==expected_files,'Trace inventory differs')
    qualified={k:True for k in MODES[1:]};largest=0.;verified=[]
    for row in r['cases']:
        name,split,h=row['name'],row['split'],row['horizon'];need(row['episode']==0 and row['seed']==[2026111504,3,0,h],'Wrong episode/seed');traces={}
        with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as z:labels={k:z[k] for k in ['red','white']}
        t=int(np.random.default_rng([2026111503,3,0,h]).integers(2,len(labels['red'][0])-h));need(row['start']==t,'Wrong start')
        for mode in MODES:
            with np.load(out/f'{name}_{split}_h{h}_{mode}.npz') as z:a={k:z[k] for k in z.files}
            largest=max(largest,audit_trace(a,h,row['seed']));traces[mode]=a
            xy=labels['red'][0,t].astype(np.float64);path=[]
            for command in cap(a['actions'].astype(np.float64)):
                xy=np.clip(xy+command,3.,61.);path.append(xy.copy())
            error=float(np.sqrt(np.sum((xy-labels['red'][0,t+h])**2)));collision=bool(np.any(np.sqrt(np.sum((np.asarray(path)[:,None]-labels['white'][0,t+1:t+h+1])**2,axis=-1))<6.));score=row['measurements'][mode]['score']
            need(np.isclose(error,score['endpoint_error'],rtol=1e-4,atol=1e-5) and score['success']==(error<=1.) and score['collision']==collision and score['collision_free_success']==(error<=1. and not collision),'Physical score differs');need(np.isfinite(row['measurements'][mode]['elapsed_seconds']) and row['measurements'][mode]['elapsed_seconds']>0,'Invalid latency')
        for mode in MODES[1:]:
            a,ref=traces[mode],traces['original'];checks={k:bool(np.array_equal(a[k],ref[k])) for k in ['proposals','order','actions']};checks['costs']=bool(np.allclose(a['costs'],ref['costs'],rtol=1e-12,atol=1e-12));checks['physical_score']=row['measurements'][mode]['score']==row['measurements']['original']['score'];checks['finite']=all(np.isfinite(v).all() for v in a.values());need(checks==row['comparisons'][mode]['checks'],'Reported qualification differs');qualified[mode]&=all(checks.values());need(np.isclose(float(np.max(np.abs(a['costs']-ref['costs']))),row['comparisons'][mode]['max_cost_abs_difference'],rtol=1e-12,atol=1e-12),'Cost difference differs');need(np.isclose(row['measurements']['original']['elapsed_seconds']/row['measurements'][mode]['elapsed_seconds'],row['comparisons'][mode]['speedup'],rtol=1e-12,atol=1e-12),'Speed ratio differs')
        verified.append(dict(name=name,split=split,horizon=h))
    need(qualified==r['qualified'],'Final qualification differs');result=dict(research_id='C015',status='awaiting_literature_review',cases=verified,trace_files=36,assignment_costs=72000,exhaustive_permutations=120,max_assignment_abs_difference=largest,qualified=qualified,scientific_result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'),source_sha256=digest(__file__),software_review_sha256=digest(sw/'review.json'),elapsed_seconds=time.monotonic()-start,owned_closed_interval=owned[0],limitation='Independent raw matrix/seed/CEM/physical-score and reported qualification reconstruction. Neural outputs and synchronized timing hardware are not independently replayed; no scientific planning success claim.')
    need(not (out/'inspection.json').exists(),'Inspection exists');atomic_json(out/'inspection.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--actual',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.actual:actual()
    else:p.error('Choose check or actual')
