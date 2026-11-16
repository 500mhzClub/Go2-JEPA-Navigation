"""Independent C013 coverage/NumPy-array verification; no Torch or GPU execution."""
import itertools
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import psutil
from go2nav.research import atomic_json,digest,verify_files,require_review

BASE=Path('outputs/gated/C013');OUT=BASE/'retry_001/o0_n3_s0';OP=BASE/'retry_001/operations/o0_n3_s0'
POLICIES=['encoder32_predictor64','encoder64_predictor64']
counter=0;array_checks=0;cost_checks=0;failures=[];maxima={}


def reported(value,path):
    global counter
    assert isinstance(value['passed'],bool)
    assert all(math.isfinite(value[k]) and value[k]>=0 for k in ['max_abs','rms'])
    counter+=1
    category='/'.join(path.split('/')[:2]);maxima[category]=max(maxima.get(category,0),value['max_abs'])
    if not value['passed']:failures.append(dict(path=path,**value))
    return value['passed']


def recompute(a,b,value):
    global array_checks
    dtype=np.result_type(a.dtype,b.dtype);a,b=a.astype(dtype),b.astype(dtype)
    close=np.isclose(a,b,rtol=1e-4,atol=1e-5);diff=np.abs(a-b)
    assert bool(close.all())==value['passed'] and int((~close).sum())==value['failed_elements']
    assert math.isclose(float(diff.max()),value['max_abs'],rel_tol=1e-12,abs_tol=1e-18)
    assert math.isclose(float(np.sqrt(np.mean(diff.astype('float64')**2))),value['rms'],rel_tol=1e-12,abs_tol=1e-18)
    array_checks+=1


def run():
    global cost_checks
    r=json.loads((OUT/'result.json').read_text());o=json.loads((OP/'result.json').read_text())
    m=json.loads((OUT/'manifest.json').read_text());om=json.loads((OP/'manifest.json').read_text())
    assert r['complete'] and r['inputs_unchanged'] and set(r['policies'])==set(POLICIES)
    assert o['returncode']==0 and o['child_terminal'] and not o['timed_out'] and not o['actions'] and not o['errors']
    try:assert psutil.Process(om['pid']).create_time()!=om['creation_time']
    except psutil.NoSuchProcess:pass
    verify_files(OUT,o['child_artifact_sha256']);assert digest(OP/'manifest.json')==o['manifest_sha256']
    verify_files(Path.cwd(),m['inputs']);verify_files(Path.cwd(),om['inputs']);verify_files(Path.cwd(),m['source_sha256']);verify_files(BASE,m['helper_sha256'])
    require_review(BASE/'software/review.json')
    rows=json.loads(Path('outputs/gated/gpu_intervals.json').read_text());owned=[x for x in rows if x['id']==om['outer_interval_id']]
    assert len(owned)==1 and owned[0]['end'] is not None and owned[0]['research_id']=='C004' and owned[0]['supervisor']=='C013'
    old=json.loads(Path('outputs/gated/C004/o0_n3_s0/result.json').read_text())
    old_encoder=json.loads(Path('outputs/gated/C004/slots_n3_s0/result.json').read_text())
    expected_gradients={k:set(v['windows']['train'][0]['unmasked']['gradients']) for k,v in [('encoder',old_encoder),('predictor',old)]}
    counts={};verdicts={};array_count=0;max_cost_delta=0.
    for policy in POLICIES:
        p=r['policies'][policy];assert p['complete'] and p['weights_unchanged']
        checks=[];before=counter
        assert set(p['windows'])==set(p['planning'])=={'train','val'}
        for split,items in p['windows'].items():
            assert len(items)==8
            for j,row in enumerate(items):
                assert row['episodes']==old['windows'][split][j]['episodes'] and row['starts']==old['windows'][split][j]['starts']
                for kind,names in [('encoder',{'slots','reconstruction','masks'}),('predictor',{'predicted'})]:
                    v=row[kind];assert set(v['outputs'])==set(v['grouping'])==names
                    assert set(v['gradients'])==expected_gradients[kind]
                    assert v['missing_gradients']=={'cpu':[],'cuda:0':[]}
                    local=[]
                    for field in ['outputs','gradients']:
                        for name,value in v[field].items():local.append(reported(value,f'{policy}/{kind}/{split}/{j}/{field}/{name}'))
                    for name,groups in v['grouping'].items():
                        assert set(groups)=={'repeat','reversed','separate'}
                        for mode,value in groups.items():local.append(reported(value,f'{policy}/{kind}/{split}/{j}/{name}/{mode}'))
                    assert v['passed']==all(local);checks+=local
                v=row['pipeline'];assert set(v['checks'])=={'cpu','repeat','reversed','separate'};local=[]
                for mode,values in v['checks'].items():
                    assert set(values)=={'history','goal','predicted','cost'}
                    for name,value in values.items():local.append(reported(value,f'{policy}/pipeline/{split}/{j}/{mode}/{name}'))
                assert v['passed']==all(local);checks+=local
        for split,v in p['planning'].items():
            assert (v['episode'],v['start'])==(old['windows'][split][0]['episodes'][0],old['windows'][split][0]['starts'][0])
            npz=OUT/f'{policy}_{split}.npz';assert digest(npz)==v['arrays_sha256']
            with np.load(npz,allow_pickle=False) as payload:a={k:payload[k] for k in payload.files}
            expected={'cpu_commands','gpu_commands','cpu_history','gpu_history','cpu_goal','gpu_goal'}|{f'{mode}_{field}' for mode in ['cpu','batched','repeat','reversed','separate'] for field in ['rollout','cost']}
            assert set(a)==expected;array_count+=len(a)
            for name,value in a.items():
                assert np.isfinite(value).all() and list(value.shape)==v['array_shapes'][name]
                shape=(100,10,2) if name.endswith('commands') else (1,3,5,128) if name.endswith('history') else (1,5,128) if name.endswith('goal') else (100,8,5,128) if name.endswith('rollout') else (100,)
                assert value.shape==shape
                assert value.dtype==('float32' if policy==POLICIES[0] and name.endswith('commands') else 'float64')
            with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as f:actions=f['actions']
            rng=np.random.default_rng((2026111516,3,['train','val'].index(split),100));future=(2*rng.standard_normal((100,8,2))).astype('float32')
            future=future*np.minimum(1,3/np.maximum(np.linalg.norm(future,axis=-1,keepdims=True),1e-12))
            e,t=v['episode'],v['start'];commands=np.concatenate((np.broadcast_to(actions[e,t:t+2],(100,2,2)),future),1).astype(a['cpu_commands'].dtype)/3
            assert np.array_equal(commands,a['cpu_commands'])
            # GPU division may differ by rounding; candidate identity is checked at machine precision, separately from rollout tolerance.
            assert np.allclose(commands,a['gpu_commands'],rtol=np.finfo(commands.dtype).eps,atol=0)
            local=[];assert set(v['checks'])=={'cpu','repeat','reversed','separate'}
            for mode,values in v['checks'].items():
                assert set(values)=={'rollout','cost'}
                for field,value in values.items():
                    recompute(a[f'batched_{field}'],a[f'{mode}_{field}'],value)
                    local.append(reported(value,f'{policy}/planning/{split}/{mode}/{field}'))
                    if field=='rollout':
                        assert len(value['by_step'])==8
                        for step in range(8):recompute(a['batched_rollout'][:,step],a[f'{mode}_rollout'][:,step],value['by_step'][step])
            assert set(v['encoding'])=={'history','goal'}
            for name,value in v['encoding'].items():
                recompute(a[f'gpu_{name}'],a[f'cpu_{name}'],value)
                local.append(reported(value,f'{policy}/planning/{split}/encoding/{name}'))
            assert v['passed']==all(local);checks+=local
            # Enumerate all5! assignments: independent of SciPy's matching implementation.
            permutations=np.array(list(itertools.permutations(range(5))))
            for mode in ['cpu','batched','repeat','reversed','separate']:
                goal=a['cpu_goal' if mode=='cpu' else 'gpu_goal'];predicted=a[f'{mode}_rollout'][:,-1]
                matrix=((predicted[:,:,None,:]-goal[:,None,:,:])**2).sum(-1)
                exhaustive=matrix[:,np.arange(5),permutations].sum(-1).min(-1)
                assert np.allclose(exhaustive,a[f'{mode}_cost'],rtol=1e-12,atol=1e-12)
                max_cost_delta=max(max_cost_delta,float(np.max(np.abs(exhaustive-a[f'{mode}_cost']))));cost_checks+=len(exhaustive)
        verdicts[policy]=all(checks);assert p['checks_passed']==verdicts[policy]
        counts[policy]=counter-before;assert counts[policy]==2772
    expected=next((p for p in POLICIES if verdicts[p]),None)
    assert r['selected_policy']==expected and r['checks_passed']==(expected is not None)
    assert array_checks==168 and counter==5544 and array_count==64 and cost_checks==2000 and 'torch' not in sys.modules
    result=dict(research_id='C013',status='awaiting_literature_review',name=r['name'],result_sha256=digest(OUT/'result.json'),
        operational_result_sha256=digest(OP/'result.json'),inspection_source_sha256=digest(__file__),
        comparisons=counter,counts_by_policy=counts,verdicts=verdicts,selected_policy=expected,failures=failures,
        maxima=maxima,independent_array_summaries=array_checks,raw_arrays=array_count,exhaustive_assignment_costs=cost_checks,
        maximum_exhaustive_cost_difference=max_cost_delta,owned_closed_interval=owned[0],torch_imported=False,
        limitation='Complete coverage, names, finite summaries, identities and every saved planning-array comparison independently checked. Raw encoder/predictor gradients and small-batch outputs were not retained, so their numerical values are not recomputed here. Sampled validity is not physical/scientific performance or all-input proof.')
    assert not (OUT/'inspection.json').exists();atomic_json(OUT/'inspection.json',result)
    print(json.dumps(result))


if __name__=='__main__':run()
