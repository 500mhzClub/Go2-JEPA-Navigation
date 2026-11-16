"""C009v2 fixed evaluations with explicit C014 full64 admission."""
import json
from pathlib import Path
from go2nav.research import digest,require_review,reviewed_artifact,verify_files
from go2nav.eval.slots_inference import require_numerical_review

SCIENCE={'planning':'B001','probes':'C001','binding':'C002'}


def route(root,stage,name):
    if stage not in SCIENCE:raise ValueError('Unplanned stage')
    parts=name.split('_')
    if len(parts)!=3 or not parts[1].startswith('n') or not parts[2].startswith('s'):raise ValueError('Invalid model name')
    kind=parts[0];n=int(parts[1][1:]);seed=int(parts[2][1:])
    allowed=(n==3 and seed in [0,1,2]) or (n in [1,6] and seed==0)
    allowed=allowed and (kind=='slots' if stage=='binding' else kind=='o1' or (kind=='o0' and n==3))
    if not allowed:raise ValueError('Unplanned stage/model/count/seed')
    root=Path(root)
    return dict(root=root,stage=stage,name=name,kind=kind,n=n,seed=seed,science_id=SCIENCE[stage],
        data=root/'outputs/data/balls',checkpoint=root/'outputs/gated/C001'/name/'model.pt',
        review=root/'outputs/gated/C014'/name/'review.json',
        science=root/'outputs/gated'/stage/name,operations=root/'outputs/gated/C009/v2/runs'/stage/name)


def arguments(python,p):
    argv=[str(python),'-m',f'go2nav.eval.balls_{p["stage"]}','--data',str(p['data']),'--n',str(p['n']),
          '--checkpoint',str(p['checkpoint']),'--out',str(p['science']),'--research',p['science_id']]
    argv+=['--inference-review' if p['stage']=='planning' else '--parent-review',str(p['review'])]
    if p['stage']=='planning':argv+=['--controls',str(p['root']/f'outputs/gated/B001/n{p["n"]}/result.json')]
    return argv


def identity(evidence,kind,n,seed):
    if evidence.get('research_id')!='C014' or not evidence.get('checks_passed') or (evidence.get('kind'),evidence.get('n'),evidence.get('seed'))!=(kind,n,seed):raise ValueError('Wrong selected-checkpoint evidence')


def inputs(p):
    root=p['root'];name=p['name'];n=p['n'];seed=p['seed'];kind=p['kind']
    evidence=require_numerical_review(p['checkpoint'],p['review']);identity(evidence,kind,n,seed)
    if require_review(p['review'])['status']!='passed':raise ValueError('Numerical review not passed')
    op=root/'outputs/gated/C014/operations'/name
    opr=require_review(op/'review.json')
    if opr['research_id']!='C014' or opr['status']!='passed':raise ValueError('C014 operation not reviewed')
    raw=json.loads((op/'result.json').read_text())
    if raw['research_id']!='C014' or raw['name']!=name or not raw['child_terminal'] or raw['returncode']!=0 or any(raw[k] for k in ['timed_out','actions','errors']) or raw['manifest_sha256']!=digest(op/'manifest.json'):raise ValueError('C014 operation incomplete/identity mismatch')
    verify_files(p['review'].parent,raw['child_artifact_sha256'])
    files={p['checkpoint'],p['review'],p['review'].parent/'result.json',p['review'].parent/'manifest.json',
           op/'review.json',op/'result.json',op/'manifest.json',p['checkpoint'].with_name('config.json')}
    m=json.loads((p['review'].parent/'manifest.json').read_text());verify_files(root,m['inputs'])
    files.update(root/f for f in m['inputs'])
    verify_files(root,m['computational_source_sha256'])
    numerical_base=root/'outputs/gated/C014'
    verify_files(numerical_base,m['helper_sha256'])
    files.update(numerical_base/f for f in m['helper_sha256'])
    swr=require_review(numerical_base/'software_numerics/review.json')
    sw=json.loads((numerical_base/'software_numerics/result.json').read_text())
    if swr['research_id']!='C014' or swr['status']!='passed' or sw['research_id']!='C014' or sw['research_sha256']!=evidence['research_sha256']:raise ValueError('Numerical software unreviewed or wrong protocol')
    om=json.loads((op/'manifest.json').read_text())
    if om['software_review_sha256']!=digest(numerical_base/'software_numerics/review.json'):raise ValueError('Numerical software review changed')
    if m['helper_sha256']!=om['helper_sha256'] or m['helper_sha256']!=sw['helper_sha256']:raise ValueError('Numerical helper inventories differ')
    files.update(numerical_base/f for f in ['software_numerics/result.json','software_numerics/review.json'])
    inspected=p['review'].parent/'inspection_review.json';ir=require_review(inspected)
    inspection=p['review'].parent/ir['result'];iv=json.loads(inspection.read_text())
    if ir['research_id']!='C014' or ir['status']!='passed' or not iv['numerical_passed'] or iv['result_sha256']!=digest(p['review'].parent/'result.json') or iv['operational_result_sha256']!=digest(op/'result.json'):raise ValueError('Numerical inspection missing or stale')
    if require_review(p['review']).get('inspection_review_sha256')!=digest(inspected):raise ValueError('Scientific admission not bound to its inspection')
    files.update([inspected,inspection])
    config=json.loads(p['checkpoint'].with_name('config.json').read_text())
    if (config['kind'],config['n'],config['seed'])!=(kind,n,seed):raise ValueError('Training identity mismatch')
    verify_files(root,config['parents']);files.update(root/f for f in config['parents'])
    for split in (['train','test'] if p['stage']=='probes' else ['test']):
        files.update(p['data']/f'n{n}'/split/f for f in ['meta.json','labels.npz','frames.npy'])
    baseline=root/f'outputs/gated/probes/l0_n{n}_s0';br=require_review(baseline/'review.json')
    baseline_result=json.loads((baseline/br['result']).read_text());bm_path=baseline/'manifest.json'
    if br['status'] not in ['passed','measured_negative'] or (baseline_result['n'],baseline_result['kind'],baseline_result['seed'])!=(n,'l0',0) or baseline_result['manifest_sha256']!=digest(bm_path):raise ValueError('Original dataset evidence invalid')
    baseline_pre=root/f'outputs/gated/C006/pre_l0_n{n}_s0.json';bm=json.loads(bm_path.read_text())
    if not any((root/f).resolve()==baseline_pre.resolve() and h==digest(baseline_pre) for f,h in bm['inputs'].items()):raise ValueError('Original dataset consultation changed')
    original=json.loads(baseline_pre.read_text())['data_sha256']
    requested=[f'outputs/data/balls/n{n}/{split}/{file}' for split in (['train','test'] if p['stage']=='probes' else ['test']) for file in ['meta.json','labels.npz','frames.npy']]
    verify_files(root,{f:original[f] for f in requested})
    files.update([baseline/'review.json',baseline/br['result'],bm_path,baseline_pre])
    if kind!='slots':
        indexes=[root/f for f in config['parents'] if Path(f).name=='index.json']
        if len(indexes)!=1:raise ValueError('Ambiguous frozen cache')
        index_path=indexes[0];index=json.loads(index_path.read_text());cache=index_path.parent
        if (index['n'],index['seed'])!=(n,seed):raise ValueError('Wrong cache identity')
        verify_files(cache,{f'{split}.npy':h for split,h in index['splits'].items()})
        encoder=root/index['encoder'];er=root/f'outputs/gated/C014/slots_n{n}_s{seed}/review.json'
        identity(require_numerical_review(encoder,er),'slots',n,seed)
        if require_review(er)['status']!='passed' or digest(encoder)!=index['encoder_sha256']:raise ValueError('Frozen encoder changed')
        files.update([index_path,cache/'train.npy',cache/'val.npy',encoder,er])
    if p['stage']=='planning':
        control=root/f'outputs/gated/B001/n{n}';review=require_review(control/'review.json');raw=json.loads((control/'result.json').read_text())
        if review['status']!='passed' or not raw['controls_passed'] or raw['n']!=n:raise ValueError('Planning controls invalid')
        if digest(control/'cases.jsonl')!=raw['cases_sha256']:raise ValueError('Control cases changed')
        files.update(control/f for f in ['review.json','result.json','manifest.json','cases.jsonl'])
    if any(not f.is_file() for f in files):raise ValueError('Required input absent')
    return sorted(files)


def consultation(path,p,files,rh,sh):
    value=json.loads(Path(path).read_text())
    expected=dict(stage=p['stage'],name=p['name'],research_sha256=rh,scientific_research_sha256=sh)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong consultation identity/protocol')
    hashes=value.get('inputs_sha256')
    if not isinstance(hashes,dict) or set(hashes)!={str(f.resolve()) for f in files}:raise ValueError('Incomplete consultation input binding')
    if not value.get('consulted_at') or not value.get('sources') or not value.get('consequences'):raise ValueError('Missing substantive consultation')
    if any(not all(s.get(k) for k in ['url','version','sections','consequence']) for s in value['sources']):raise ValueError('Incomplete source consultation')
    verify_files(p['root'],hashes)
    return value
