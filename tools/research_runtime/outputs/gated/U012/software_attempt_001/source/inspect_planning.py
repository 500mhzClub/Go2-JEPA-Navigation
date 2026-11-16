"""C014 independent inspection of unchanged B001 object-planning raw cases."""
import argparse
import ast
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
INHERITED=Path('outputs/gated/B003/inspection/v2')


def need(value,message):
    if not value:raise ValueError(message)


def audit_rows(rows, controls, labels, n, kind, episodes):
    expected = {(e, h) for e in range(episodes) for h in [4, 6, 8]}
    need(len(rows) == len(expected), 'Incomplete case count')
    keys = [(r['episode'], r['horizon']) for r in rows]
    need(len(set(keys)) == len(keys) and set(keys) == expected, 'Missing or duplicate task')
    error_deltas, margins = [], []
    for r in rows:
        e, h, t = r['episode'], r['horizon'], r['start_frame']
        control = controls[e, h]
        need(r['research_id'] == 'B001' and r['n'] == n and r['arm'] == kind, 'Run identity mismatch')
        need(r['group'] == e//200 and 2 <= t < labels['red'].shape[1]-h, 'Group or bounds mismatch')
        need(r['seed'] == [2026111504, n, e, h], 'CEM seed mismatch')
        for key in ['n', 'episode', 'horizon', 'start_frame', 'seed']:
            need(r[key] == control[key], 'Control task mismatch: '+key)
        need(r['goal_frame'] == t+h, 'Goal indexing mismatch')
        need(np.array_equal(r['start_position'], labels['red'][e,t]) and
             np.array_equal(r['goal_position'], labels['red'][e,t+h]), 'Truth positions mismatch')
        a = np.asarray(r['actions'], dtype=np.float64)
        need(a.shape == (h,2) and np.isfinite(a).all(), 'Invalid actions')
        norms = np.sqrt(np.sum(a*a, axis=1))
        need(float(norms.max()) <= 3+1e-6, 'Action cap violation')
        xy = np.asarray(r['start_position'], dtype=np.float64); path=[]
        for action, norm in zip(a, norms):
            xy = np.maximum(3., np.minimum(61., xy+action*min(1.,3/max(norm,1e-12))))
            path.append(xy.copy())
        error = float(np.sqrt(np.sum((xy-np.asarray(r['goal_position']))**2)))
        reported = r['endpoint_error']
        need(np.isfinite(reported) and np.isclose(error, reported, rtol=1e-4, atol=1e-5), 'Endpoint score mismatch')
        separation = np.sqrt(np.sum((np.array(path)[:,None]-labels['white'][e,t+1:t+h+1])**2, axis=-1))
        collision = bool(np.any(separation < 6.)); success = error <= 1.
        need(all(type(r[k]) is bool for k in ['success','collision','collision_free_success']), 'Invalid flag type')
        need(r['success'] == success and r['collision'] == collision and
             r['collision_free_success'] == (success and not collision), 'Independent classification mismatch')
        need(np.isfinite(r['latency_s']) and r['latency_s'] >= 0, 'Invalid latency')
        error_deltas.append(abs(error-reported))
        margins.append(min(abs(error-1), float(np.min(np.abs(separation-6)))))
    return dict(cases=len(rows), episodes=episodes, max_endpoint_discrepancy=max(error_deltas),
                minimum_classification_boundary_distance=min(margins))

def summary_from_rows(rows, kind):
    out={}
    for h in [4,6,8]:
        selected=sorted((r for r in rows if r['horizon']==h),key=lambda r:r['episode'])
        values=np.array([r['success'] for r in selected],dtype=float)
        bootstrap=np.random.default_rng(2026111505).integers(len(values),size=(2000,len(values)))
        ci=np.quantile(values[bootstrap].mean(1),[.025,.975],method='linear')
        errors=np.array([r['endpoint_error'] for r in selected])
        out[str(h)]=dict(count=len(selected),success=float(values.mean()),success_ci95=ci.tolist(),
            endpoint_median=float(np.quantile(errors,.5,method='linear')),endpoint_mean=float(errors.mean()),
            collision_free_success=float(np.mean([r['collision_free_success'] for r in selected])),
            latency_mean_s=float(np.mean([r['latency_s'] for r in selected])),
            groups={str(g):float(np.mean([r['success'] for r in selected if r['group']==g])) for g in sorted({r['group'] for r in selected})})
    return {kind:out}

def compare_summary(actual, expected):
    need(actual.keys()==expected.keys(),'Summary keys mismatch')
    for key in expected:
        if isinstance(expected[key],dict): compare_summary(actual[key],expected[key])
        else:
            a,b=np.asarray(actual[key]),np.asarray(expected[key])
            need(a.shape==b.shape and np.isfinite(a).all() and np.allclose(a,b,rtol=1e-12,atol=1e-12),'Summary mismatch: '+key)


def operation_identity(o,name):
    need((o['research_id'],o['scientific_research_id'],o['stage'],o['name'])==('C009','B001','planning',name),'Wrong operation identity')
    need(o['status']=='awaiting_literature_review' and o['returncode']==0 and o['child_terminal'] and o['inputs_unchanged'] and not any(o[k] for k in ['timed_out','actions','errors']),'Failed or incomplete operation')


def inherited_guard():
    r=require_review(INHERITED/'software/review.json');s=json.loads((INHERITED/'software/result.json').read_text());pre=json.loads((BASE/'pre_planning_inspection_implementation.json').read_text())
    need(r['status']=='passed' and s['source_sha256']==digest(INHERITED/'verify_cases.py')==pre['inherited_source_sha256'] and digest(INHERITED/'software/review.json')==pre['inherited_review_sha256'],'Original inspector changed or unreviewed')


def check():
    import torch
    from go2nav.eval.balls_planning import score,summarize
    torch.set_num_threads(2);start=time.monotonic();out=BASE/'planning_inspection_software';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'inspect_planning.py');checks=[]
    def refuses(call):
        try:call()
        except ValueError:return
        raise AssertionError('Expected refusal')
    try:
        inherited_guard();checks.append('reviewed_original_inspector_source_bound')
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='B001':node.value='B003'
                return node
        old=ast.parse((INHERITED/'verify_cases.py').read_text());new=ast.parse(Path(__file__).read_text())
        for name in ['audit_rows','summary_from_rows','compare_summary']:
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
            b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
            need(ast.dump(a)==ast.dump(Normalize().visit(b)),'Inherited computation changed')
        checks.append('three_independent_scoring_functions_AST_identical_except_research_identity')
        for n in [1,3,6]:
            for kind in ['o0','o1']:
                labels=dict(red=np.full((2,100,2),20.,np.float32),white=np.full((2,100,n,2),40.,np.float32))
                labels['white'][0,41:45,0]=20.;labels['red'][1,40]=61.;labels['red'][1,44]=61.;labels['red'][1,66]=30.
                rows=[];controls={}
                for e in range(2):
                    for h in [4,6,8]:
                        t=h*10;a=np.zeros((h,2),np.float32)
                        if e==1 and h==4:a[:,0]=3
                        row=dict(research_id='B001',n=n,arm=kind,episode=e,horizon=h,start_frame=t,goal_frame=t+h,start_position=labels['red'][e,t].tolist(),goal_position=labels['red'][e,t+h].tolist(),seed=[2026111504,n,e,h],group=0,actions=a.tolist(),latency_s=.5,**score(labels['red'][e,t],labels['red'][e,t+h],a,labels['white'][e,t+1:t+h+1]))
                        rows.append(row);controls[e,h]={k:row[k] for k in ['n','episode','horizon','start_frame','seed']}
                audit_rows(rows,controls,labels,n,kind,2);summary=summary_from_rows(rows,kind);compare_summary(summary,summarize(rows)['arms'])
                need(summary[kind]['4']['success']==1 and summary[kind]['4']['collision_free_success']==.5 and summary[kind]['6']['success']==.5,'Known fixture scores differ')
                checks.append(f'n{n}_{kind}_float32_success_failure_wall_collision_and_bootstrap')
        # Exact one-pixel success and six-pixel noncollision boundaries.
        labels=dict(red=np.full((1,100,2),20.,np.float32),white=np.full((1,100,3,2),40.,np.float32));rows=[];controls={}
        for h in [4,6,8]:
            t=h*10;labels['red'][0,t+h]=[21.,20.];labels['white'][0,t+1:t+h+1]=[26.,20.];a=np.zeros((h,2),np.float32)
            row=dict(research_id='B001',n=3,arm='o1',episode=0,horizon=h,start_frame=t,goal_frame=t+h,start_position=[20.,20.],goal_position=[21.,20.],seed=[2026111504,3,0,h],group=0,actions=a.tolist(),latency_s=0.,**score(labels['red'][0,t],labels['red'][0,t+h],a,labels['white'][0,t+1:t+h+1]));rows.append(row);controls[0,h]=row.copy()
        checked=audit_rows(rows,controls,labels,3,'o1',1);need(all(r['success'] and not r['collision'] for r in rows) and checked['minimum_classification_boundary_distance']==0,'Boundary classification differs');checks.append('exact_onepixel_success_sixpixel_noncollision_boundaries')
        for key,value in [('research_id','wrong'),('arm','o0'),('n',1),('endpoint_error',.25),('success',False),('collision',True),('collision_free_success',False),('seed',[0]),('group',1),('goal_frame',99),('latency_s',-1),('start_position',[21.,20.]),('goal_position',[20.,20.])]:
            bad=copy.deepcopy(rows);bad[0][key]=value;refuses(lambda:audit_rows(bad,controls,labels,3,'o1',1))
        for v in [float('nan'),4.]:
            bad=copy.deepcopy(rows);bad[0]['actions'][0][0]=v;refuses(lambda:audit_rows(bad,controls,labels,3,'o1',1))
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]]:refuses(lambda:audit_rows(bad,controls,labels,3,'o1',1))
        checks.append('eighteen_wrong_task_identity_score_flags_actions_count_refusals')
        summary=summary_from_rows(rows,'o1')
        for key,value in [('success',.25),('success_ci95',[.1,.9]),('groups',{'0':.5}),('count',3)]:
            bad=copy.deepcopy(summary);bad['o1']['4'][key]=value;refuses(lambda:compare_summary(bad,summary))
        checks.append('four_changed_summary_refusals')
        good=dict(research_id='C009',scientific_research_id='B001',stage='planning',name='o1_n3_s0',status='awaiting_literature_review',returncode=0,child_terminal=True,inputs_unchanged=True,timed_out=False,actions=[],errors=[]);operation_identity(good,'o1_n3_s0')
        for k,v in [('research_id','wrong'),('scientific_research_id','wrong'),('stage','probes'),('name','o0_n3_s0'),('status','budget_incomplete'),('returncode',1),('child_terminal',False),('inputs_unchanged',False),('timed_out',True),('actions',['signal']),('errors',['error'])]:
            bad=copy.deepcopy(good);bad[k]=v;refuses(lambda:operation_identity(bad,'o1_n3_s0'))
        checks.append('eleven_incomplete_wrong_or_failed_operation_refusals')
        need(not torch.cuda.is_initialized(),'CUDA initialized');checks.append('no_CUDA')
        result=dict(research_id='C014',scientific_research_id='B001',research_sha256=require_record('C014')[1],scientific_research_sha256=require_record('B001')[1],status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-start,source_sha256=digest(__file__),repository_source_sha256={'go2nav/eval/balls_planning.py':digest('go2nav/eval/balls_planning.py')},preconsultation_sha256=digest(BASE/'pre_planning_inspection_implementation.json'),inherited_source_sha256=digest(INHERITED/'verify_cases.py'),limitation='Synthetic physical scoring and schema/refusal fixtures, no real planning or CEM rerun. Actual inspection verifies all raw cases and evidence identity; saved actions cannot independently reconstruct unsaved optimization latents/costs.')
        atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C014',status='technical_blocked',checks=checks,error=repr(exc),source_sha256=digest(__file__)));raise


def actual(name):
    import torch
    from go2nav.eval.slots_inference import require_numerical_review
    start=time.monotonic();record,rh=require_record('C014');science,sh=require_record('B001');need(name in record['protocol']['predictor_cases'],'Unplanned planning case')
    torch.set_num_threads(2);sw=BASE/'planning_inspection_software';need(require_review(sw/'review.json')['status']=='passed','Unreviewed inspector');inherited_guard()
    sr=json.loads((sw/'result.json').read_text());need(sr['source_sha256']==digest(BASE/'inspect_planning.py') and sr['research_sha256']==rh and sr['scientific_research_sha256']==sh,'Changed source/protocol');verify_files(Path.cwd(),sr['repository_source_sha256'])
    out=Path('outputs/gated/planning')/name;op=BASE/'runs/planning'/name
    r=json.loads((out/'result.json').read_text());o=json.loads((op/'result.json').read_text());m=json.loads((op/'manifest.json').read_text());sm=json.loads((out/'manifest.json').read_text())
    operation_metadata('planning',name,o,m,sm);operation_identity(o,name)
    need(o['manifest_sha256']==digest(op/'manifest.json'),'Operation manifest changed');verify_files(out,o['child_artifact_sha256']);verify_files(Path.cwd(),m['preconsultation']['inputs_sha256'])
    try:live=psutil.Process(m['pid']).create_time()==m['creation_time']
    except psutil.NoSuchProcess:live=False
    need(not live,'Owned child live')
    kind,count,seed=name.split('_');n=int(count[1:]);seed=int(seed[1:]);need((r['research_id'],r['n'],r['status'],r['task_seed'],r['cem_seed'])==('B001',n,'awaiting_literature_review',2026111503,2026111504),'Wrong science identity')
    checkpoint=Path('outputs/gated/C001')/name/'model.pt';numerics=require_numerical_review(checkpoint,Path('outputs/gated/C014')/name/'review.json');need((numerics['kind'],numerics['n'],numerics['seed'])==(kind,n,seed),'Wrong numerical parent')
    need(sm['research_id']=='B001' and sm['research_sha256']==sh and sm['code_commit']==m['source_commit'] and sm['config']==science['protocol'],'Science source/protocol differs');verify_files(Path.cwd(),sm['inputs'])
    runner=json.loads((BASE/'software/result.json').read_text());need(m['source_sha256']==runner['source_sha256'] and m['software_review_sha256']==digest(BASE/'software/review.json'),'Runner software differs');verify_files(BASE,m['source_sha256']);verify_files(Path.cwd(),runner['repository_source_sha256']);need(m['integration_research_sha256']==rh,'Integration protocol differs')
    probe=Path('outputs/gated/probes')/name/'review.json';pr=require_review(probe);need(pr['status'] in ['passed','measured_negative'] and pr['run_identity']==dict(n=n,kind=kind,seed=seed,checkpoint_sha256=digest(checkpoint)),'Incomplete or wrong preceding probe')
    ledger=json.loads(Path('outputs/gated/gpu_intervals.json').read_text());owned=[]
    for rid in [m['outer_interval_id'],sm['run_id']]:
        rows=[x for x in ledger if x['id']==rid];need(len(rows)==1 and rows[0]['research_id']=='B001' and rows[0]['end'] is not None,'Owned interval incomplete');owned.append(rows[0])
    need(r['cases_sha256']==digest(out/'cases.jsonl'),'Case bytes changed')
    control=Path(f'outputs/gated/B001/n{n}');need(require_review(control/'review.json')['status']=='passed','Controls not reviewed');cr=json.loads((control/'result.json').read_text());need(cr['controls_passed'] and cr['cases_sha256']==digest(control/'cases.jsonl'),'Control cases changed')
    controls={(row['episode'],row['horizon']):row for row in map(json.loads,(control/'cases.jsonl').read_text().splitlines()) if row['arm']=='recorded'};need(len(controls)==3000,'Control task coverage mismatch')
    with np.load(f'outputs/data/balls/n{n}/test/labels.npz') as z:labels={k:z[k] for k in ['red','white']}
    rows=[json.loads(line) for line in (out/'cases.jsonl').read_text().splitlines()];checked=audit_rows(rows,controls,labels,n,kind,1000)
    summary=summary_from_rows(rows,kind);compare_summary(r['arms'],summary);need(not torch.cuda.is_initialized(),'CUDA initialized')
    result=dict(research_id='C014',execution_integration=m['execution_integration'],scientific_research_id='B001',name=name,status='awaiting_literature_review',run_identity=dict(n=n,kind=kind,seed=seed,checkpoint_sha256=digest(checkpoint)),result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'),source_sha256=digest(__file__),software_review_sha256=digest(sw/'review.json'),probe_review_sha256=digest(probe),checks=checked,summary=summary,all_task_scoring_summary_checks_passed=True,bootstrap_repeats=2000,owned_closed_intervals=owned,cuda_initialized=False,elapsed_seconds=time.monotonic()-start,limitation='Independent complete saved-action integration, task/label/classification and all summary/bootstrap checks. Original evaluator did not retain all CEM candidate latents/costs, so optimization is not independently reconstructed. Checkpoint-specific C014 numerical admission remains separate evidence.')
    need(not (out/'inspection.json').exists(),'Inspection already exists');atomic_json(out/'inspection.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    try:actual(a.run)
    except BaseException as exc:
        failure=Path('outputs/gated/U012/failures/planning')/a.run
        need(not failure.exists(),'Prior failed inspection must be reviewed before a new attempt')
        failure.mkdir(parents=True);shutil.copy2(__file__,failure/'inspect_planning.py')
        atomic_json(failure/'result.json',dict(research_id='C014',execution_research_id='U012',name=a.run,status='technical_blocked',error=repr(exc),source_sha256=digest(__file__)))
        raise
