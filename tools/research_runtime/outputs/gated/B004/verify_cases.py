"""Independent B004 raw-case inspection; never executes a model or approves a review."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import psutil
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/B004/inspection')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def audit_rows(rows, controls, labels, n, kind, episodes):
    expected = {(e, h) for e in range(episodes) for h in [4, 6, 8]}
    need(len(rows) == len(expected), 'Incomplete case count')
    keys = [(r['episode'], r['horizon']) for r in rows]
    need(len(set(keys)) == len(keys) and set(keys) == expected, 'Missing or duplicate task')
    error_deltas, margins = [], []
    for r in rows:
        e, h, t = r['episode'], r['horizon'], r['start_frame']
        control = controls[e, h]
        need(r['research_id'] == 'B004' and r['n'] == n and r['arm'] == kind, 'Run identity mismatch')
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


def inspect(name):
    started=time.monotonic();record,rh=require_record('B004');need(name in record['protocol']['order'],'Unplanned name')
    need(require_review(BASE.parent/'software/review.json')['status']=='passed','Inspector not reviewed')
    sw=json.loads((BASE.parent/'software/result.json').read_text())
    need(sw['source_sha256']['verify_cases.py']==digest(__file__) and sw['research_sha256']==rh,'Inspector source/protocol changed')
    d=Path('outputs/gated/planning')/name;op=Path('outputs/gated/B004/operations')/name
    result=json.loads((d/'result.json').read_text());operations=json.loads((op/'result.json').read_text())
    need(result['status']=='awaiting_literature_review' and result['complete_cases']==3000,'Incomplete scientific result')
    need(operations['child_terminal'] and operations['returncode']==0 and not operations['errors'] and not operations['signals'],'Unsuccessful CPU operation')
    execution=json.loads((op/'execution.json').read_text())
    try:need(psutil.Process(execution['pid']).create_time()!=execution['creation_time'],'Exact child still live')
    except psutil.NoSuchProcess:pass
    verify_files(d,operations['child_artifact_sha256'])
    need(operations['execution_sha256']==digest(op/'execution.json'),'Execution changed')
    manifest=json.loads((d/'manifest.json').read_text());verify_files(Path.cwd(),manifest['inputs'])
    verify_files(Path.cwd(),json.loads(Path('outputs/gated/B004/pre_'+name+'.json').read_text())['inputs_sha256'])
    need(result['manifest_sha256']==digest(d/'manifest.json') and manifest['research_sha256']==rh,'Manifest binding mismatch')
    runtime=json.loads(Path('outputs/gated/B004/software/result.json').read_text())
    verify_files(Path('outputs/gated/B004'),runtime['source_sha256']);verify_files(Path.cwd(),runtime['repository_source_sha256'])
    n,kind,seed=result['n'],result['kind'],result['seed']
    need(name==f'{kind}_n{n}_s{seed}' and seed in [1,2] and n==3 and result['device']=='cpu','Run identity mismatch')
    need(result['weights_unchanged'] and not result['cuda_initialized'] and operations['no_B004_gpu_intervals'],'Invalid runtime evidence')
    need(result['cases_sha256']==digest(d/'cases.jsonl') and result['numerics_sha256']==digest(d/'numerics.npz'),'Raw data changed')
    controls_path=Path(f'outputs/gated/B001/n{n}')
    need(require_review(controls_path/'review.json')['status']=='passed','Controls not reviewed')
    control_rows=[json.loads(x) for x in (controls_path/'cases.jsonl').read_text().splitlines()]
    controls={(r['episode'],r['horizon']):r for r in control_rows if r['arm']=='recorded'}
    need(len(controls)==3000,'Control coverage mismatch')
    labels=dict(np.load(f'outputs/data/balls/n{n}/test/labels.npz'))
    rows=[json.loads(x) for x in (d/'cases.jsonl').read_text().splitlines()]
    checked=audit_rows(rows,controls,labels,n,kind,1000)
    summary=summary_from_rows(rows,kind);compare_summary(result['arms'],summary)
    a=np.load(d/'numerics.npz');need(a['proposals'].shape==(100,8,2),'Wrong numerical batch')
    need(all(np.isfinite(a[k]).all() for k in a.files),'Nonfinite numerical diagnostic')
    np.testing.assert_allclose(a['latents'][:8],a['singleton_latents'],rtol=1e-4,atol=1e-5)
    np.testing.assert_allclose(a['costs'][:8],a['singleton_costs'],rtol=1e-4,atol=1e-5)
    need('torch' not in sys.modules,'Inspector imported torch')
    out=BASE/name;out.mkdir(parents=True,exist_ok=False)
    inspected=dict(research_id='B004',name=name,status='awaiting_literature_review',checks=checked,
        all_task_scoring_summary_checks_passed=True,summary=summary,saved_singleton_numerics_passed=True,
        scientific_result_sha256=digest(d/'result.json'),operational_result_sha256=digest(op/'result.json'),
        inspection_source_sha256=digest(__file__),software_review_sha256=digest(BASE.parent/'software/review.json'),
        elapsed_seconds=time.monotonic()-started,torch_imported=False,
        limitation='Independent saved-case reconstruction and summary checks only. Original in-memory frozen-weight checks and reversed-batch checks remain source-bound run evidence, not independently re-executed here. No review is generated.')
    atomic_json(out/'result.json',inspected);print(json.dumps(inspected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);inspect(p.parse_args().run)
