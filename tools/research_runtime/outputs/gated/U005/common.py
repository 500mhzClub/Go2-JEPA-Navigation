"""Additional baseline operations under U001; original training stays unchanged."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U005')
ORIGINAL = Path('outputs/gated/C010')
HELPERS = ('common.py', 'science.py', 'run_train.py', 'check.py')

def need(condition, message):
    if not condition: raise ValueError(message)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

def parents():
    need(require_review(ORIGINAL/'software/review.json')['status'] == 'passed', 'C010 software unreviewed')
    sw = json.loads((ORIGINAL/'software/result.json').read_text())
    need(sw['research_sha256'] == require_record('C010')[1], 'C010 protocol changed')
    verify_files(ORIGINAL, sw['source_sha256']); verify_files(Path.cwd(), sw['record_sha256'])
    base = Path('outputs/gated/U003')
    need(require_review(base/'software/review.json')['status'] == 'passed', 'U003 software unreviewed')
    core_sw = json.loads((base/'software/result.json').read_text())
    need(core_sw['research_sha256'] == require_record('U003')[1], 'U003 protocol changed')
    verify_files(base, core_sw['source_sha256']); verify_files(Path.cwd(), core_sw['repository_source_sha256'])
    core = module('u005_core', base/'common.py')
    _, _, unlimited, _ = core.parents()
    numerical = json.loads(Path('outputs/gated/C014/o0_n3_s0/result.json').read_text())
    changed = core.reconcile(Path.cwd(), sw['repository_source_sha256'], numerical['implementation_sha256'])
    return (module('u005_original_stages', ORIGINAL/'stages.py'),
            module('u005_original_supervision', ORIGINAL/'supervision.py'), core, unlimited, changed)

def core_files(root, seed, stages, core):
    if seed != 0: return stages.core_reviews(root, seed)
    reviews = core.core_reviews(root)
    files = set(reviews)
    for path in reviews:
        review = require_review(path); files.add(path.parent/review['result'])
    for kind in ['l0','l1','o0','o1']:
        for phase in ['planning','probes']:
            folder = 'planning_full' if phase == 'planning' and kind in ('o0','o1') else phase
            dest = root/'outputs/gated'/folder/f'{kind}_n3_s0'
            review = require_review(dest/'review.json'); raw = json.loads((dest/review['result']).read_text())
            need(raw.get('status') not in ['budget_incomplete','technical_blocked'], 'Incomplete core result')
            if phase == 'planning':
                need(all(raw['arms'][kind][str(h)]['count'] == 1000 for h in [4,6,8]), 'Partial core planning')
            files.add(dest/('cases.jsonl' if phase == 'planning' else 'cases.npz'))
    binding = root/'outputs/gated/binding/slots_n3_s0'
    review = require_review(binding/'review.json'); raw = json.loads((binding/review['result']).read_text())
    need(raw.get('status') not in ['budget_incomplete','technical_blocked'], 'Incomplete binding')
    need(digest(binding/'cases.jsonl') == raw['cases_sha256'], 'Binding cases changed')
    files.add(binding/'cases.jsonl')
    return files

def inputs(p, stages, core):
    root = p['root']
    files = stages.original_data(root) | stages.p26_reviews(root) | core_files(root, 0, stages, core)
    if p['seed'] == 2: files |= core_files(root, 1, stages, core)
    for name in stages.ORDER[:stages.ORDER.index(p['name'])]:
        files |= stages.prior_training(root, name)
    for label in ['C010','D003','U001','U003','U005']:
        files.add(root/f'docs/research/records/{label}.json')
        base = root/'outputs/gated'/label
        for name in ['result.json','review.json']: files.add(base/'software'/name)
        sw = json.loads((base/'software/result.json').read_text())
        files |= {base/name for name in sw['source_sha256']}
    need(all(f.is_file() for f in files), 'Required input absent')
    return sorted({f.resolve() for f in files})

def arguments(python, p, stages):
    original = stages.arguments(python, p)
    return [str(python), str(BASE/'science.py'), original[2], *original[3:]]

def consultation(path, p, files, stages):
    pre = stages.consultation(path, p, files, require_record('C010')[1], require_record('C001')[1])
    need(pre.get('u005_record_sha256') == require_record('U005')[1], 'Wrong integration consultation')
    need(pre.get('u001_authorization_sha256') == require_record('U001')[1], 'Wrong authorization consultation')
    need(pre.get('software_review_sha256') == digest(BASE/'software/review.json'), 'Wrong software consultation')
    return pre

def admit(ledger, out, science):
    need(science == 'C001', 'Wrong scientific allocation')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); rows = json.loads(ledger.read_text())
        need(all(row['end'] is not None for row in rows), 'Open GPU interval')
        row = dict(id=uuid.uuid4().hex, research_id=science, supervisor_research_id='C010',
            budget_policy='U001', integration_research_id='U005', out=str(out), start=time.time(), end=None)
        rows.append(row); atomic_json(ledger, rows)
    return row

def progress_paths(p, out):
    return [out/'training.log'] + [p['science']/name for name in ['metrics.jsonl','checkpoint.pt','model.pt']]
