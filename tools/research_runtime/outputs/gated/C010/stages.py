"""Only the four approved fresh baseline seeds; no Torch or automatic reviews."""
import importlib.util
import json
from pathlib import Path
from go2nav.research import digest, require_review, verify_files

ORDER = ['l0_n3_s1', 'l1_n3_s1', 'l0_n3_s2', 'l1_n3_s2']


def route(root, name):
    if name not in ORDER:
        raise ValueError('Unplanned baseline/count/seed')
    root = Path(root)
    return dict(root=root, stage='train', name=name, kind=name[:2], n=3,
                seed=int(name[-1]), science_id='C001', data=root/'outputs/data/balls',
                science=root/'outputs/gated/C001'/name,
                operations=root/'outputs/gated/C010/runs'/name)


def arguments(python, p):
    return [str(python), '-m', 'go2nav.training.balls', 'train', '--data', str(p['data']),
            '--n', '3', '--model', p['kind'], '--seed', str(p['seed']),
            '--research', 'C001', '--out', str(p['science'])]


def reviewed(folder, research_id, statuses=('passed',)):
    folder = Path(folder)
    review = require_review(folder/'review.json')
    result = folder/review['result']
    raw = json.loads(result.read_text())
    if review['research_id'] != research_id or raw['research_id'] != research_id or review['status'] not in statuses:
        raise ValueError('Wrong prerequisite research identity/status')
    if raw.get('status') in ['budget_incomplete', 'technical_blocked']:
        raise ValueError('Incomplete prerequisite result')
    return review, raw, {folder/'review.json', result}


def original_data(root):
    folder = root/'outputs/gated/A014/preparation'
    _, raw, files = reviewed(folder, 'A014')
    names = [f'outputs/data/balls/n3/{split}/{name}'
             for split in ['train', 'val'] for name in ['meta.json', 'labels.npz', 'frames.npy']]
    verify_files(root, {f:raw['data_sha256'][f] for f in names})
    return files | {root/f for f in names}


def p26_reviews(root):
    files = set()
    for stage, science in [('export', 'corpus/track_run'), ('s1', 's1'), ('compare', 'comparison')]:
        op = root/'outputs/gated/D003/runs'/stage
        _, raw, bound = reviewed(op, 'D003')
        if (raw['stage'] != stage or not raw['child_terminal'] or raw['returncode'] != 0
                or raw['timed_out'] or raw['signals'] or raw['errors'] or not raw['inputs_unchanged']):
            raise ValueError('P26 operation not complete')
        if raw['manifest_sha256'] != digest(op/'manifest.json'):
            raise ValueError('P26 manifest changed')
        verify_files(root, raw['child_artifact_sha256'])
        sf = root/'outputs/gated/D001'/science
        _, sr, sb = reviewed(sf, 'D001', ('passed', 'measured_negative') if stage == 'compare' else ('passed',))
        if not any((root/f).resolve() == (sf/'result.json').resolve() and h == digest(sf/'result.json')
                   for f,h in raw['child_artifact_sha256'].items()):
            raise ValueError('P26 scientific result absent from operation')
        if stage == 's1' and sr['step'] != 20000:
            raise ValueError('Incomplete S1 training')
        files |= bound | sb | {op/'manifest.json'} | {root/f for f in raw['child_artifact_sha256']}
    return files


def core_reviews(root, seed):
    files = set()
    for kind in ['l0', 'l1', 'o0', 'o1']:
        for phase in ['planning', 'probes']:
            folder = root/'outputs/gated'/phase/f'{kind}_n3_s{seed}'
            review = require_review(folder/'review.json')
            raw = json.loads((folder/review['result']).read_text())
            identity = review.get('run_identity', {})
            actual = tuple(raw.get(k, identity.get(k)) for k in ['n', 'kind', 'seed'])
            if review['status'] not in ['passed', 'measured_negative'] or actual != (3,kind,seed):
                raise ValueError('Incomplete core identity/review')
            if raw.get('status') in ['budget_incomplete','technical_blocked']:
                raise ValueError('Incomplete core result')
            cases = folder/('cases.jsonl' if phase == 'planning' else 'cases.npz')
            if digest(cases) != raw['cases_sha256']:
                raise ValueError('Core cases changed')
            if phase == 'planning' and any(raw['arms'][kind][str(h)]['count'] != 1000 for h in [4,6,8]):
                raise ValueError('Partial core planning')
            files |= {folder/'review.json', folder/review['result'], cases}
    folder = root/f'outputs/gated/binding/slots_n3_s{seed}'
    _, raw, bound = reviewed(folder, 'C002', ('passed','measured_negative'))
    if (raw['n'], raw['kind'], raw['seed']) != (3,'slots',seed):
        raise ValueError('Wrong core binding identity')
    cases = folder/'cases.jsonl'
    if digest(cases) != raw['cases_sha256']:
        raise ValueError('Core binding cases changed')
    return files | bound | {cases}


def prior_training(root, name):
    sf = root/'outputs/gated/C001'/name
    review, raw, files = reviewed(sf, 'C001')
    if (raw['kind'],raw['n'],raw['seed'],raw['step']) != (name[:2],3,int(name[-1]),303000):
        raise ValueError('Wrong/unfinished prior baseline')
    if raw['checkpoint_sha256'] != digest(sf/'model.pt'):
        raise ValueError('Prior model changed')
    audit_path = root/review['final_audit_review']
    if digest(audit_path) != review['final_audit_review_sha256']:
        raise ValueError('Prior completion audit changed')
    ar = require_review(audit_path)
    audit = json.loads((audit_path.parent/ar['result']).read_text())
    if (ar['status'] != 'passed' or audit['step'] != 303000 or
            audit['model_sha256'] != digest(sf/'model.pt') or
            audit['checkpoint_sha256'] != digest(sf/'checkpoint.pt')):
        raise ValueError('Prior full artifact audit invalid')
    op = root/'outputs/gated/C010/runs'/name
    _, operation, bound = reviewed(op, 'C010')
    if (operation['name'] != name or operation['returncode'] != 0 or not operation['child_terminal']
            or operation['timed_out'] or operation['actions'] or operation['errors'] or not operation['inputs_unchanged']):
        raise ValueError('Prior training operation failed')
    if operation['manifest_sha256'] != digest(op/'manifest.json'):
        raise ValueError('Prior operation changed')
    verify_files(sf,operation['child_artifact_sha256'])
    if any(f not in operation['child_artifact_sha256'] for f in ['model.pt','checkpoint.pt','result.json','manifest.json','config.json','metrics.jsonl']):
        raise ValueError('Incomplete prior training artifacts')
    return files | bound | {op/'manifest.json',audit_path,audit_path.parent/ar['result']} | {sf/f for f in operation['child_artifact_sha256']}


def inputs(p):
    root = p['root']
    files = original_data(root) | p26_reviews(root) | core_reviews(root,0)
    # Retain the reviewed core/legacy gate used by the actual P26 supervisor.
    d = root/'outputs/gated/D003'
    require_review(d/'software/review.json')
    sw = json.loads((d/'software/result.json').read_text())
    verify_files(d,sw['source_sha256'])
    spec=importlib.util.spec_from_file_location('c010_d003_stages',d/'stages.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files |= set(module.core_reviews(root)) | {d/'software/review.json',d/'software/result.json'}
    if p['seed'] == 2:
        files |= core_reviews(root,1)
    for name in ORDER[:ORDER.index(p['name'])]:
        files |= prior_training(root,name)
    if any(not f.is_file() for f in files):
        raise ValueError('Required input absent')
    return sorted(files)


def consultation(path,p,files,rh,sh):
    value=json.loads(Path(path).read_text())
    expected=dict(stage='train',name=p['name'],research_sha256=rh,scientific_research_sha256=sh)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Wrong consultation identity/protocol')
    hashes=value.get('inputs_sha256')
    if not isinstance(hashes,dict) or set(hashes)!={str(f.resolve()) for f in files}:
        raise ValueError('Incomplete consultation inputs')
    if not value.get('consulted_at') or not value.get('sources') or not value.get('consequences'):
        raise ValueError('Missing substantive consultation')
    if any(not all(s.get(k) for k in ['url','version','sections','consequence']) for s in value['sources']):
        raise ValueError('Incomplete source consultation')
    verify_files(p['root'],hashes)
    return value
