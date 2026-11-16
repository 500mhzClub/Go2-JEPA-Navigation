"""Additional-seed numerical admission; unchanged computation, no automatic reviews."""
import importlib.util
import json
from pathlib import Path
import psutil
from go2nav.research import digest, require_record, require_review, verify_files

ORDER = ['l0_n3_s1', 'l1_n3_s1', 'l0_n3_s2', 'l1_n3_s2']


def route(root, name):
    if name not in ORDER:
        raise ValueError('Unplanned baseline/count/seed')
    root = Path(root)
    return dict(root=root, stage='inference', name=name, kind=name[:2], n=3,
                seed=int(name[-1]), science_id='A019', data=root/'outputs/data/balls',
                checkpoint=root/'outputs/gated/C001'/name/'model.pt',
                science=root/'outputs/gated/A019/inference'/name,
                operations=root/'outputs/gated/A019/runs'/name)


def arguments(python, p):
    return [str(python), '-m', 'go2nav.eval.balls_inference', '--checkpoint', str(p['checkpoint']),
            '--data', str(p['data']), '--out', str(p['science']), '--research', 'A019']


def parent_module():
    base = Path('outputs/gated/C010')
    if require_review(base/'software/review.json')['status'] != 'passed':
        raise ValueError('C010 software not reviewed')
    sw = json.loads((base/'software/result.json').read_text())
    if sw['research_sha256'] != require_record('C010')[1]:
        raise ValueError('C010 protocol changed')
    verify_files(base, sw['source_sha256'])
    spec = importlib.util.spec_from_file_location('a019_training_parents', base/'stages.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inputs(p):
    root, name = p['root'], p['name']
    parent = parent_module()
    files = parent.prior_training(root, name) | parent.original_data(root)
    sf = root/'outputs/gated/C001'/name
    training = require_review(sf/'review.json')
    audit = root/'outputs/gated/A018/runs'/name
    if (root/training['final_audit_review']).resolve() != (audit/'review.json').resolve():
        raise ValueError('Wrong final audit location')
    ar = require_review(audit/'review.json'); raw = json.loads((audit/ar['result']).read_text())
    if (ar['status'], ar['research_id'], raw['research_id'], raw['name'], raw['step']) != ('passed', 'A018', 'A018', name, 303000):
        raise ValueError('Wrong/incomplete A018 identity')
    if raw['manifest_sha256'] != digest(audit/'manifest.json'):
        raise ValueError('A018 manifest changed')
    am = json.loads((audit/'manifest.json').read_text())
    verify_files(root, am['input_sha256']); verify_files(sf, am['snapshot_sha256'])
    files |= {audit/'manifest.json'} | {root/f for f in am['input_sha256']}
    om = json.loads((root/'outputs/gated/C010/runs'/name/'manifest.json').read_text())
    try:
        if psutil.Process(om['pid']).create_time() == om['creation_time']:
            raise ValueError('Exact trainer still exists')
    except psutil.NoSuchProcess:
        pass
    for label in ['A018', 'C010']:
        for f in ['result.json', 'review.json']:
            files.add(root/'outputs/gated'/label/'software'/f)
    if any(not f.is_file() for f in files):
        raise ValueError('Missing numerical prerequisite')
    return sorted(files)


def consultation(path, p, files, rh, sh):
    pre = json.loads(Path(path).read_text())
    expected = dict(name=p['name'], research_sha256=rh, scientific_research_sha256=sh,
                    inputs_sha256={str(f): digest(f) for f in files},
                    software_review_sha256=digest(p['root']/'outputs/gated/A019/software/review.json'))
    if any(pre.get(k) != v for k,v in expected.items()) or not all(pre.get(k) for k in ['sources', 'consequences', 'consulted_at']):
        raise ValueError('Missing/stale numerical consultation')
    return pre
