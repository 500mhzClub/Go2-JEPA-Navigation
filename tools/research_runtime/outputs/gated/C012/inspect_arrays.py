"""C012 independent NumPy inspection of saved arrays, scores and input identity."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import psutil
from go2nav.research import atomic_json,digest,verify_files

base=Path('outputs/gated/C012');out=base/'o0_n3_s0';op=base/'operations/o0_n3_s0'
r=json.loads((out/'result.json').read_text());o=json.loads((op/'result.json').read_text());om=json.loads((op/'manifest.json').read_text());m=json.loads((out/'manifest.json').read_text())
assert r['complete'] and r['weights_unchanged'] and r['inputs_unchanged']
assert o['returncode']==0 and o['child_terminal'] and not o['errors'] and not o['actions'] and not o['timed_out']
try:assert psutil.Process(om['pid']).create_time()!=om['creation_time']
except psutil.NoSuchProcess:pass
verify_files(out,o['child_artifact_sha256']);verify_files(Path.cwd(),m['inputs']);verify_files(Path.cwd(),om['inputs']);verify_files(Path.cwd(),m['source_sha256']);verify_files(base,m['helper_sha256'])
assert digest(op/'manifest.json')==o['manifest_sha256']
row=[x for x in json.loads(Path('outputs/gated/gpu_intervals.json').read_text()) if x['id']==om['outer_interval_id']]
assert len(row)==1 and row[0]['research_id']=='C004' and row[0]['supervisor']=='C012' and row[0]['end'] is not None
old=json.loads(Path('outputs/gated/C004/o0_n3_s0/result.json').read_text());checks=0;summary={}

def check_summary(a,b,s):
    global checks
    dtype=np.result_type(a.dtype,b.dtype);a,b=a.astype(dtype),b.astype(dtype)
    delta=np.abs(a-b);close=np.isclose(a,b,rtol=1e-4,atol=1e-5)
    ratio=np.abs(a.astype('float64')-b.astype('float64'))/(1e-5+1e-4*np.abs(b.astype('float64')))
    assert bool(close.all())==s['passed'] and int((~close).sum())==s['failed_elements']
    for key,value in [('max_abs',float(delta.max())),('rms',float(np.sqrt(np.mean(delta.astype('float64')**2)))),('max_tolerance_ratio',float(ratio.max()))]:
        assert math.isclose(value,s[key],rel_tol=1e-12,abs_tol=1e-18),(key,value,s[key])
    checks+=1

for split,v in r['splits'].items():
    p=out/f'{split}_arrays.npz';assert digest(p)==v['arrays_sha256']
    with np.load(p,allow_pickle=False) as payload:arrays={k:payload[k] for k in payload.files}
    expected={'history','actions','cpu64_batched'}|{f'{g}_{mode}' for g in ['gpu32','gpu64'] for mode in ['batched','repeat','reversed','separate_pairs']}|{f'common32_{mode}' for mode in ['batched','reversed','separate_pairs']}
    assert set(arrays)==expected
    for k,a in arrays.items():
        assert np.isfinite(a).all() and list(a.shape)==v['array_shapes'][k]
        assert a.shape==({'history':(100,3,5,128),'actions':(100,10,2)}.get(k,(100,8,5,128)))
        assert a.dtype==('float64' if k.startswith(('cpu64','gpu64')) else 'float32')
    assert np.array_equal(arrays['history'],np.broadcast_to(arrays['history'][:1],arrays['history'].shape))
    e,t=v['episode'],v['start'];assert (e,t)==(old['windows'][split][0]['episodes'][0],old['windows'][split][0]['starts'][0])
    with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as labels:actions=labels['actions']
    rng=np.random.default_rng((2026111516,3,['train','val'].index(split),100))
    future=(2*rng.standard_normal((100,8,2))).astype(np.float32)
    future*=np.minimum(1,3/np.maximum(np.linalg.norm(future,axis=-1,keepdims=True),1e-12))
    expected_actions=np.concatenate((np.broadcast_to(actions[e,t:t+2],(100,2,2)),future),1)/3
    assert np.array_equal(expected_actions,arrays['actions'])
    summary[split]={}
    for group,items in v['comparisons'].items():
        a=arrays[f'{group}_batched']
        for kind,stats in items.items():
            b=arrays['cpu64_batched' if kind=='cpu64_reference' else f'{group}_{kind}']
            check_summary(a,b,stats)
            assert len(stats['by_step'])==8
            for step in range(8):check_summary(a[:,step],b[:,step],stats['by_step'][step])
            summary[split][f'{group}/{kind}']={k:val for k,val in stats.items() if k!='by_step'}
assert checks==180 and 'torch' not in sys.modules
result=dict(research_id='C012',name='o0_n3_s0',status='awaiting_literature_review',result_sha256=digest(out/'result.json'),operational_result_sha256=digest(op/'result.json'),inspection_source_sha256=digest(__file__),checks=checks,comparisons=20,raw_arrays=28,summary=summary,owned_closed_interval=row[0],torch_imported=False,limitation='All saved numerical summaries, raw array shapes/finiteness and command identities verified independently. Encoder generation and model immutability checks remain source-bound execution evidence, not rerun here.')
assert not (out/'inspection.json').exists();atomic_json(out/'inspection.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='summary'}))
