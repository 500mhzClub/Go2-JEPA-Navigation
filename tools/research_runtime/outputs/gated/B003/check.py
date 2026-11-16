"""B003 adapter verification against unchanged B001; CPU fixtures only."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import time

import planning as p
from go2nav.eval.balls_planning import _evaluate
from go2nav.research import atomic_json, digest, require_record
import numpy as np
import torch

p.guard(); started = time.monotonic(); record, rh = require_record('B003')
base = p.BASE; out = base/'software'; out.mkdir(exist_ok=False)
ledger_before = digest('outputs/gated/gpu_intervals.json')
inherited_before = {str(f):digest(f) for b in ['B002','C005','C007'] for f in Path(f'outputs/gated/{b}').glob('*.py')}
checks = {}
for f in ['planning.py','run_one.py','check.py']: ast.parse((base/f).read_text())


class Fake(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def encode(self, x):
        return x.mean((-3,-2,-1))[...,None].expand(-1,-1,2)

    def rollout(self, z, a, steps):
        return z[:,-1:]+a[:,-steps:].cumsum(1)


frames = np.broadcast_to(np.arange(100,dtype=np.uint8)[None,:,None,None,None],(2,100,4,4,3)).copy()
labels = dict(actions=np.full((2,100,2),.2,np.float32),red=np.broadcast_to([20.,20.],(2,100,2)).copy(),
              white=np.broadcast_to([40.,40.],(2,100,3,2)).copy())
with tempfile.TemporaryDirectory(prefix='b003_fixture_') as tmp:
    root=Path(tmp); original=root/'original'; original.mkdir()
    ledger=root/'ledger.json'; atomic_json(ledger,[])
    _evaluate(frames,labels,3,original,'B003',Fake(),dict(budget_hours=48,gpu_ledger=str(ledger)),'l0')
    expected=[json.loads(line) for line in (original/'cases.jsonl').read_text().splitlines()]
    actual=list(p.cases(Fake(),frames,labels,3,'l0'))
    assert len(actual)==len(expected)==6
    assert len({(r['episode'],r['horizon']) for r in actual})==6
    for a,b in zip(actual,expected):
        assert {k:v for k,v in a.items() if k!='latency_s'}=={k:v for k,v in b.items() if k!='latency_s'}
        xy=np.array(a['start_position']); path=[]
        for action in a['actions']:
            xy=np.clip(xy+action,3.,61.); path.append(xy.copy())
        error=float(np.linalg.norm(xy-np.array(a['goal_position'])))
        assert abs(error-a['endpoint_error'])<1e-12 and a['success']==(error<=1)
        collision=bool((np.linalg.norm(np.array(path)[:,None]-labels['white'][a['episode'],a['start_frame']+1:a['goal_frame']+1],axis=-1)<6).any())
        assert a['collision']==collision and a['collision_free_success']==(error<=1 and not collision)
    checks['all_case_fields_match_original_except_latency']=True
    checks['independent_endpoint_collision_flags_and_unique_tasks']=True
    summary=p.summarize(actual); reference=json.loads((original/'result.json').read_text())
    for h in ['4','6','8']:
        a=summary['arms']['l0'][h]; b=reference['arms']['l0'][h]
        assert {k:v for k,v in a.items() if k!='latency_mean_s'}=={k:v for k,v in b.items() if k!='latency_mean_s'}
    checks['unchanged_summary_bootstrap_groups']=True
    inputs=p.task_inputs(frames,labels['actions'],0,p.task_start(3,0,8),8)
    _,old=p.prior.candidate_check(Fake(),inputs)
    for n in [1,3,6]:
        metrics,arrays=p.numerical_check(Fake(),inputs,n)
        assert all(v<1e-12 for v in metrics.values())
        expected_proposals=p.cap(2*np.random.default_rng([p.CEM_SEED,n,0,8]).standard_normal((100,8,2)).astype(np.float32))
        np.testing.assert_array_equal(arrays['proposals'],expected_proposals)
        if n==3:
            for key in arrays: np.testing.assert_array_equal(arrays[key],old[key])
    checks['candidate_diagnostic_matches_reviewed_B002_and_count_seed']=True
    p.BASE=root/'consult';p.BASE.mkdir()
    checkpoint=root/'parent/calibrated.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'fixture')
    (checkpoint.parent/'review.json').write_text('fixture only')
    dh={'fixture':'hash'}
    value=dict(name='l0_n3_s0',research_sha256=rh,b001_record_sha256=digest('docs/research/records/B001.json'),
               checkpoint_sha256=digest(checkpoint),parent_review_sha256=digest(checkpoint.parent/'review.json'),data_sha256=dh,
               consulted_at='fixture',consequences='test only',sources=[dict(url='fixture',version='fixture',sections='fixture',consequence='fixture')])
    pre=p.BASE/'pre_l0_n3_s0.json';atomic_json(pre,value);p.consultation('l0_n3_s0',checkpoint,dh,rh)
    for key in ['name','research_sha256','b001_record_sha256','checkpoint_sha256','parent_review_sha256','data_sha256','sources']:
        wrong=copy.deepcopy(value);wrong[key]=None;atomic_json(pre,wrong)
        try:p.consultation('l0_n3_s0',checkpoint,dh,rh)
        except ValueError:pass
        else:raise AssertionError('Stale consultation accepted')
    p.BASE=base
    checks['stale_consultation_parent_protocol_and_data_rejected']=True

parents={}
for name in record['protocol']['order']:
    _,_,checkpoint,_,_,cfg=p.lineage(name)
    parents[name]=dict(checkpoint_sha256=digest(checkpoint),n=cfg['n'],kind=cfg['kind'],seed=cfg['seed'])
checks['all_six_actual_final_parents_and_controls_bound']=True
assert digest('go2nav/eval/balls_planning.py')=='737da767014a0e524373764346edf4de6f4e071b374d8f2f5c27b2b4a4d235b3'
assert ledger_before==digest('outputs/gated/gpu_intervals.json')
assert all(digest(f)==h for f,h in inherited_before.items());p.guard()
checks['no_GPU_interval_or_inherited_source_mutation']=True
result=dict(research_id='B003',research_sha256=rh,status='awaiting_literature_review',checks=checks,
            elapsed_seconds=time.monotonic()-started,cuda_initialized=False,actual_scientific_planning_cases=0,
            actual_parent_bindings=parents,source_sha256={f:digest(base/f) for f in ['planning.py','run_one.py','check.py']},
            repository_source_sha256={str(f):digest(f) for f in sorted(Path('go2nav').rglob('*.py'))},
            a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'),
            inherited_b002_software_review_sha256=digest('outputs/gated/B002/software/review.json'),
            limitation='Synthetic adapter/candidate checks and real lineage only; actual complete planning results remain pending.')
atomic_json(out/'result.json',result)
print(json.dumps(dict(checks=checks,elapsed_seconds=result['elapsed_seconds'],actual_parent_count=len(parents))),flush=True)
